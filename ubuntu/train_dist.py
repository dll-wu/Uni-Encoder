import os
import math
import torch
import numpy as np
import random
from argparse import ArgumentParser
from pprint import pformat
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel, DataParallel
from torch.optim.lr_scheduler import LambdaLR
import sys
sys.path.append("..")
from uni_encoder import *
from utils.dataloader import build_dataloaders
import time
from ignite.engine import Engine, Events
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.metrics import MetricsLambda, RunningAverage, Loss, Accuracy, Average
from ignite.handlers import (
        Timer,
        ModelCheckpoint,
        EarlyStopping,
        global_step_from_engine,
        )
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    OutputHandler,
    OptimizerParamsHandler,
    )

import logging
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def average_distributed_scalar(scalar, args):
    if args.local_rank == -1:
        return scalar
    scalar_t = (
        torch.tensor(scalar, dtype=torch.float, device=args.device)
        / torch.distributed.get_world_size()
    )
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()


def train():
    parser = ArgumentParser()
    parser.add_argument("--model_size", type=int, default=768, help="Hs of the model")
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="models/",
        help="Path or URL of the model",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="If False train from scratch"
    )
    parser.add_argument(
        "--use_post_training", type=str, default="NoPost", help="Post training"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/dataset.json", help="Path or url of the dataset. "
    )
    parser.add_argument(
        "--dataset_cache",
        action="store_true",
        help="use dataset cache or not",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of subprocesses for data loading",
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="Patience for early stopping"
    )
    parser.add_argument("--n_saved", type=int, default=3, help="Save the best n models")
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=8, help="Batch size for validation"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="noam",
        choices=["noam", "linear"],
        help="method of optim",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=3500, help="Warmup steps for noam"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
        help="Accumulate gradients on several steps",
    )
    parser.add_argument(
        "--max_norm", type=float, default=1.0, help="Clipping gradient norm"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--fp16",
        type=str,
        default="",
        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1: not distributed)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="ubuntu",
        help="The corpus you use. Should be: ubuntu, douban or personachat",
    )
    parser.add_argument(
        "--attn_map",
        type=str,
        default="arrow",
        help="The attention mechanism: arrow, diag, square",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning("Running process %d", args.local_rank)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = args.local_rank != -1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    
    logger.info("Prepare tokenizer, models and optimizer.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    configuration = BertConfig()
    model = UniEncoder(configuration, args)
    if args.pretrained:
        model.from_pretrained(os.path.join(args.model_checkpoint, 'pytorch_model.bin'))
    else:
        model.from_pretrained()
    torch.save(model.state_dict(), 'models/pytorch_model.bin')

    ## for post-training
    if args.use_post_training == 'ums':
        model.bert.resize_token_embeddings(model.config.vocab_size + 1)
        model_dict = model.state_dict()
        post_model = torch.load('post_training_model/bert-post-uncased-pytorch_model.pth', map_location=args.device)
        post_model = post_model['model']
        update_dict = {k.replace('_bert_model','bert'):v for k, v in post_model.items() if k.replace('_bert_model','bert') in model_dict.keys()}
        update_dict.pop('bert.cls.predictions.bias')
        model_dict.update(update_dict)
        model.config = model.bert.config
        model.load_state_dict(model_dict)
    
    elif args.use_post_training == 'bert_fp':
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.bert.resize_token_embeddings(model.config.vocab_size + 1)
        model_dict = model.bert.bert.state_dict()
        post_model = torch.load('post_training_model/bert.pt', map_location=args.device)
        update_dict = {k:v for k, v in post_model.items() if k in model_dict.keys()}
        model_dict.update(update_dict)
        model.config = model.bert.config
        model.bert.bert.load_state_dict(model_dict)
        
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(args.device)
   
    logger.info("Prepare datasets")
    train_loader, val_loader, train_sampler, valid_sampler = build_dataloaders(args, tokenizer, logger)
    
    if args.scheduler == "noam":
        # make the peak lr to be the value we want
        args.warmup_steps = min(
            len(train_loader) // args.gradient_accumulation_steps + 1, args.warmup_steps
        )
        args.lr /= args.model_size ** (-0.5) * args.warmup_steps ** (-0.5)
    optimizer = AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)

    # fp16
    if args.fp16:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    
    # better not use parallel. 
    # if you would like to use parallel, you shouldn't calculate loss in the model, just calculate it in the update and evaluate function
    if args.distributed:
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    
    # trainer
    def update(engine, batch):
        model.train()
        input_ids, input_ids_mask, token_type_ids, attention_mask, labels, true_response_label, position_ids = tuple(
            input_tensor.to(args.device) for input_tensor in batch[:-1]
        )
        output, MLMloss, CEloss = model(
                                        input_ids=input_ids_mask,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels,
                                        true_response_label=true_response_label,
                                        position_ids=position_ids,
                                        bos_locations=batch[-1]
                                    )
        if MLMloss:
            loss = (MLMloss + CEloss)/ args.gradient_accumulation_steps
        else:
            loss = (CEloss) / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        if MLMloss:
            return loss.item()*args.gradient_accumulation_steps, MLMloss.item(), CEloss.item()
        else:
            return loss.item()*args.gradient_accumulation_steps, 0, CEloss.item()
    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def evaluate(engine, batch):
        model.eval()
        with torch.no_grad():
            input_ids, input_ids_mask, token_type_ids, attention_mask, labels, true_response_label, position_ids = tuple(
                input_tensor.to(args.device) for input_tensor in batch[:-1]
            )
            # do not use the mask input
            output, MLMloss, CEloss = model(
                                        input_ids=input_ids,
                                        token_type_ids=token_type_ids,
                                        attention_mask=attention_mask,
                                        labels=labels,
                                        true_response_label=true_response_label,
                                        position_ids=position_ids,
                                        bos_locations=batch[-1]
                                    )
            mean_logits = torch.nn.functional.softmax(output, dim=-1)
            pred_labels = torch.argmax(mean_logits, dim=-1)  # batch_size 
            
            flags = (true_response_label == pred_labels)
            acc = sum(flags).item()/true_response_label.shape[0]
        return acc, CEloss

    evaluator = Engine(evaluate)


    # Attach evaluator to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader)
    )
    
    # Count running time
    timer = Timer(average=False)
    timer.attach(trainer, start=Events.STARTED)

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: train_sampler.set_epoch(engine.state.epoch),
        )
        evaluator.add_event_handler(
            Events.EPOCH_STARTED,
            lambda engine: valid_sampler.set_epoch(engine.state.epoch),
        )
    # learning rate schedule
    def noam_lambda(iteration):
        step = ((iteration - 1) // args.gradient_accumulation_steps) + 1
        # step cannot be zero
        step = max(step, 1)
        # calculate the noam learning rate
        lr = args.model_size ** (-0.5) * min(
            (step) ** (-0.5), (step) * args.warmup_steps ** (-1.5)
        )
        return lr

    if args.scheduler == "noam":
        noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)
        scheduler = LRScheduler(noam_scheduler)
    else:
        scheduler = PiecewiseLinear(
            optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)]
        )

    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)
    
    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "train_loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "MLM_loss")
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, "CE_loss")
    # attach the metrics to the evaluator
    eval_metrics = {}
    eval_metrics["acc"] = Average(output_transform=lambda x: x[0])
    eval_metrics["eval_loss"] = Average(output_transform=lambda x: x[1])
    eval_metrics["avg_acc"] = MetricsLambda(
        average_distributed_scalar, eval_metrics["acc"], args
    )
    eval_metrics["avg_eval_loss"] = MetricsLambda(
        average_distributed_scalar, eval_metrics["eval_loss"], args
    )
  

    for name, metric in eval_metrics.items():
        metric.attach(evaluator, name)
    
    # log training info on the main process only
    if args.local_rank in [-1, 0]:
        # TODO print log, NOTICE: sometimes errors occur, because logdir should be log_dir
        tb_logger = TensorboardLogger(log_dir=None)
        pbar_log_file = open(os.path.join(tb_logger.writer.logdir, "training.log"), "w")
        pbar = ProgressBar(persist=True, file=pbar_log_file, mininterval=4)
        pbar.attach(trainer, metric_names=["train_loss", 'MLM_loss', 'CE_loss'])
        trainer.add_event_handler(
            Events.ITERATION_STARTED,
            lambda _: pbar.log_message(
                "lr: %.5g @ iteration %d step %d\nTime elapsed: %f"
                % (
                    optimizer.param_groups[0]["lr"],
                    trainer.state.iteration,
                    ((trainer.state.iteration - 1) // args.gradient_accumulation_steps) + 1,
                    timer.value(),
                )
            ),
        )
        evaluator.add_event_handler(
            Events.COMPLETED,
            lambda _: pbar.log_message(
                "Validation metrics:\n%s\nTime elapsed: %f" % \
                (pformat(evaluator.state.metrics), timer.value())
            ),
        )
        
        tb_logger.attach(
            trainer,
            log_handler=OutputHandler(tag="training", metric_names=["train_loss", 'MLM_loss', 'CE_loss']),
            event_name=Events.ITERATION_COMPLETED,
        )
        tb_logger.attach(
            trainer,
            log_handler=OptimizerParamsHandler(optimizer),
            event_name=Events.ITERATION_STARTED,
        )
        
        def global_step_transform(*args, **kwargs):
            return trainer.state.iteration

        tb_logger.attach(
            evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=list(eval_metrics.keys()),
                global_step_transform=global_step_transform
            ),
            event_name=Events.EPOCH_COMPLETED,
        )
        
        # we save the model with maximum acc
        def score_function(engine):
            score = engine.state.metrics["avg_acc"]
            return score

        checkpoint_handlers = [
            ModelCheckpoint(
                tb_logger.writer.logdir,
                "checkpoint",
                score_function=score_function,
                n_saved=args.n_saved,
            )
            for _ in range(1)
        ]
        
        # save n best models
        evaluator.add_event_handler(
            Events.EPOCH_COMPLETED,
            checkpoint_handlers[0],
            {"mymodel": getattr(model, "module", model)},
        )

        earlyStopping_handler = EarlyStopping(
            patience=args.patience, score_function=score_function, trainer=trainer
        )
        
        evaluator.add_event_handler(Events.COMPLETED, earlyStopping_handler)
        
        # save args
        torch.save(args, tb_logger.writer.logdir + "/model_training_args.bin")
    
        # save config
        getattr(model, "module", model).config.to_json_file(
            os.path.join(tb_logger.writer.logdir, CONFIG_NAME)
        )
        
        # save vocab
        tokenizer.save_vocabulary(tb_logger.writer.logdir)
    
    # start training
    trainer.run(train_loader, max_epochs=args.n_epochs)
    
    if args.local_rank in [-1, 0]:
        # rename the best model
        saved_name = checkpoint_handlers[0]._saved[-1][1]
        os.rename(
            os.path.join(tb_logger.writer.logdir, saved_name),
            os.path.join(tb_logger.writer.logdir, saved_name+"_"+WEIGHTS_NAME),
        )
        
        pbar_log_file.close()
        tb_logger.close()



if __name__ == "__main__":
    train()    
    
