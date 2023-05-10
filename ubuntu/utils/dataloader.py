import torch
from transformers import cached_path
import json
from utils.dataset import CUSTdataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

def get_data(tokenizer, dataset_path, dataset_cache, logger):
    cache_dir = "cache/dataset_cache_" + type(tokenizer).__name__
    if dataset_cache:
        logger.info("Load tokenized dataset from cache at %s", cache_dir)
        dataset = torch.load(cache_dir)
    else:
        logger.info("Download dataset from %s", dataset_path)
        cache_file = cached_path(dataset_path) 
        with open(cache_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
 
        logger.info("Tokenize and encode the dataset")
        def tokenize(obj):
            if isinstance(obj, int):
                return obj
            if isinstance(obj, str):
                return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            if isinstance(obj, dict):
                return dict((n, tokenize(o)) for n, o in obj.items())
            return list(tokenize(o) for o in obj)
    
        dataset = tokenize(dataset)
        torch.save(dataset, cache_dir)
    return dataset


def build_dataloaders(args, tokenizer, logger): 
    logger.info("Build train and validation dataloaders")
    
    datasets = get_data(tokenizer, args.data_path, args.dataset_cache, logger)
    
    train_dataset, valid_dataset = (
        CUSTdataset(datasets["train"], tokenizer, use_in_batch=True),
        CUSTdataset(datasets["valid"], tokenizer),
    )
    train_sampler = DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=train_dataset.collate,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.train_batch_size,
        shuffle=(not args.distributed),
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=valid_sampler,
        collate_fn=valid_dataset.collate,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_size=args.valid_batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, train_sampler, valid_sampler
