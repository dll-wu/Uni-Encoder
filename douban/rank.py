import random
import numpy as np
from itertools import chain
from argparse import ArgumentParser
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import sys
import os
sys.path.append("..")
from uni_encoder import *
from torch.nn.parallel import DataParallel


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[unused1]", "[unused2]"]


def build_input_from_segments(history, responses, tokenizer, args):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS
    )
    sequence = history  # on douban dataset
    if tokenizer.eos_token_id is None:
        if len(sequence) % 2 == 1:
            sequence = [
                [speaker1 if i % 2 == 1 else speaker2] + s  # history: start with speaker2, end with speaker2
                for i, s in enumerate(sequence)
            ]
        else:
            sequence = [
                [speaker2 if i % 2 == 1 else speaker1] + s  # history: start with speaker1, end with speaker2
                for i, s in enumerate(sequence)
            ]
    else:
        eou = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
        if len(sequence) % 2 == 1:
            sequence = [
                [speaker1 if i % 2 == 1 else speaker2] + s + [eou]  # history: start with speaker2, end with speaker2
                for i, s in enumerate(sequence)
            ]
        else:
            sequence = [
                [speaker2 if i % 2 == 1 else speaker1] + s + [eou]  # history: start with speaker1, end with speaker2
                for i, s in enumerate(sequence)
            ]

    MaxRes = 256
    if args.use_post_training == 'bert_fp':
        MaxRes = 128

    max_response_length = max([len(res)+2 for res in responses])
    max_response_length = min(max_response_length, MaxRes)
    while True:
        sequence_ = [[bos]] + sequence  + [[eos]]
        sequence_his = list(chain(*sequence_))
        if len(sequence_his) <= 512 - MaxRes:
            break

        sequence = sequence[1:]

    instance = {}
    instance["input_ids"] = sequence_his
    instance["token_type_ids"] = [0] * len(instance['input_ids'])  # origin token_type

    instance["position_ids"] = list(range(len(instance['input_ids'])))
    # extend
    bos_locations = []
    start = len(instance['input_ids'])
    for i, response in enumerate(responses):
        # response = [bos] + response[:max_response_length-2] + [eos]
        response = [bos] + response[-MaxRes+2:] + [eos]
        instance["input_ids"].extend(response)
        bos_locations.append(start)
        start += len(response)
        # continue position ids
        instance["position_ids"].extend(list(range(bos_locations[0], bos_locations[0] + len(response))))

    instance["token_type_ids"] += (len(instance['input_ids']) - len(instance["token_type_ids"])) * [1] # origin token_type
    
    
    instance["bos_locations"] = bos_locations
    instance["attention_mask"] = [1] * len(instance["input_ids"])
    return instance


def rank_responses(history, responses, tokenizer, model, args):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)    
    instance = build_input_from_segments(
        history, responses, tokenizer, args
    )
        
    input_ids = pad_sequence(
        [
            torch.tensor(instance["input_ids"], dtype=torch.long)
            
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    attention_mask = pad_sequence(
        [
            torch.tensor(instance["attention_mask"], dtype=torch.long)
            
        ],
        batch_first=True,
        padding_value=0,
    )
    
    token_type_ids = pad_sequence(
        [
            torch.tensor(instance["token_type_ids"], dtype=torch.long)
            
        ],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    position_ids = pad_sequence(
        [
            torch.tensor(instance["position_ids"], dtype=torch.long)

        ],
        batch_first=True,
        padding_value=0,
    )
    bos_locations = [
        instance["bos_locations"] 
    ]
    
    batch = [input_ids, token_type_ids, attention_mask,  position_ids, bos_locations]

    input_ids, token_type_ids, attention_mask, position_ids = tuple(
        input_tensor.to(args.device) for input_tensor in batch[:-1]
    )
    output, MLMloss, CEloss = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    bos_locations=bos_locations
                            )
    mean_logits = torch.nn.functional.softmax(output, dim=-1)
    pred_label = torch.argmax(mean_logits, dim=-1).item()
    # print(mean_logits)
    return pred_label, mean_logits


def rank_label(records, replies, tokenizer, model, args):

    def tokenize(obj):
        if isinstance(obj, int):
            return obj
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    history = tokenize(records)
    responses = tokenize(replies)
    with torch.no_grad():
        pred_label, mean_logits = rank_responses(history, responses, tokenizer, model, args)
    return pred_label, mean_logits

class Ranker:
    def __init__(self, model_path, args):
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=True)
        if args.use_post_training == 'bert_fp':
            special_tokens_dict = {'eos_token': '[eos]'}
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        config = BertConfig.from_pretrained(model_path)
        model = UniEncoder(config, args)
        model_files = os.listdir(model_path)
        if 'pytorch_model.bin' in model_files:
            model_file = 'pytorch_model.bin'
        else:
            model_file = [f for f in model_files if 'checkpoint' in f]
            model_file = sorted(model_file)[-1]    
        
        specific_path = os.path.join(model_path, model_file)
        print('Choose model:', specific_path)
        model.load_state_dict(torch.load(specific_path, map_location=args.device))
        # model.from_pretrained(specific_path) 
        model.to(args.device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model
        self.args = args

    def rank(self, records, responses):
        return rank_label(records, responses, self.tokenizer, self.model, self.args)