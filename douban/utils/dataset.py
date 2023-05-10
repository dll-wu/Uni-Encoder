import torch
from torch.utils.data import Dataset
from itertools import chain
from transformers import DataCollatorForLanguageModeling
from torch.nn.utils.rnn import pad_sequence


SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[unused1]", "[unused2]"]


class CUSTdataset(Dataset):
    def __init__(self, data, tokenizer, use_in_batch=False, batch_first=True):
        self.data = data
        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.batch_first = batch_first
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
        self.use_in_batch= use_in_batch

    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        history = self.data[index]["history"]
        responses = self.data[index]["responses"]
        res_labels = self.data[index]["labels"]
        
        return self.process( history, responses, res_labels)
    
    def process(self, history, responses, res_labels):
        bos, eos, speaker1, speaker2 = self.tokenizer.convert_tokens_to_ids(
            SPECIAL_TOKENS
        )

        sequence = history 
        if self.tokenizer.eos_token_id is None:
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
            eou = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
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
        if self.tokenizer.eos_token_id is not None:
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
        instance["token_type_ids"] = [0] * len(instance['input_ids']) 
        instance["position_ids"] = list(range(len(instance['input_ids'])))
        # extend
        bos_locations = []
        start = len(instance['input_ids'])
        for i, response in enumerate(responses):
            response = [bos] + response[-MaxRes+2:] + [eos]
            instance["input_ids"].extend(response)
            bos_locations.append(start)
            start += len(response)
            # continue position ids
            instance["position_ids"].extend(list(range(bos_locations[0], bos_locations[0] + len(response))))
    
        instance["token_type_ids"] += (len(instance['input_ids']) - len(instance["token_type_ids"])) * [1] 
        
        # true response index
        # notice that in the train and dev set, there is only 1 golden response for 1 history
        instance["true_response_label"] = res_labels.index(1)

        instance["bos_locations"] = bos_locations
        instance["attention_mask"] = [1] * len(instance["input_ids"])
        
        inputs_left, labels_left = self.mlm_collator.mask_tokens(torch.LongTensor([instance["input_ids"]]))
        instance["input_ids_mask"] = inputs_left[0]
        instance["labels"] = labels_left[0]
        
        # without MLM
        instance["input_ids"] = torch.LongTensor(instance["input_ids"])

        return instance
    
    
    def use_in_batch_negative_samples(self, batch):
        input_ids_in_batch = []
        input_ids_mask_in_batch = []
        labels_in_batch = []
        length_of_golden = []
        for i in range(len(batch)):
            instance = batch[i]
            input_ids = instance["input_ids"]
            input_ids_mask = instance["input_ids_mask"]
            bos_locations = instance["bos_locations"]
            labels = instance["labels"]
            label = instance["true_response_label"]


            if label + 1 == len(bos_locations):
                input_ids_in_batch.append(input_ids[bos_locations[label]:])
                input_ids_mask_in_batch.append(input_ids_mask[bos_locations[label]:])
                labels_in_batch.append(labels[bos_locations[label]:])
            else:
                input_ids_in_batch.append(input_ids[bos_locations[label]:bos_locations[label+1]])
                input_ids_mask_in_batch.append(input_ids_mask[bos_locations[label]:bos_locations[label+1]])
                labels_in_batch.append(labels[bos_locations[label]:bos_locations[label+1]])
            
            length_of_golden.append(len(input_ids_in_batch[-1]))


        golden_responses_cat = torch.cat(input_ids_in_batch, dim=0)
        golden_responses_mask_cat = torch.cat(input_ids_mask_in_batch, dim=0)
        labels_cat = torch.cat(labels_in_batch, dim=0)
        for i in range(len(batch)):
            instance = batch[i]

            # ======== add hard negatives as additional negative samples =========
            input_ids = instance["input_ids"]
            input_ids_mask = instance["input_ids_mask"]
            bos_locations = instance["bos_locations"]
            labels = instance["labels"]
            label = instance["true_response_label"]

            hard_negatives_in_batch = []
            hard_negatives_mask_in_batch = []
            hard_negatives_labels_in_batch = []
            length_of_hard_negatives = []

            for index, location in enumerate(bos_locations):
                current_bos = bos_locations[index]
                if index == label:
                    continue
                else:
                    if location == bos_locations[-1]:
                        hard_negatives_in_batch.append(input_ids[current_bos:])
                        hard_negatives_mask_in_batch.append(input_ids_mask[current_bos:])
                        hard_negatives_labels_in_batch.append(labels[current_bos:])
                    else:
                        current_eos = bos_locations[index+1]
                        hard_negatives_in_batch.append(input_ids[current_bos: current_eos])
                        hard_negatives_mask_in_batch.append(input_ids_mask[current_bos: current_eos])
                        hard_negatives_labels_in_batch.append(labels[current_bos: current_eos])

                length_of_hard_negatives.append(len(hard_negatives_in_batch[-1]))

            hard_negatives_in_batch = torch.cat(hard_negatives_in_batch, dim=0)
            hard_negatives_mask_in_batch = torch.cat(hard_negatives_mask_in_batch, dim=0)
            hard_negatives_labels_in_batch = torch.cat(hard_negatives_labels_in_batch, dim=0)
            # ==================================================================


            start = instance["bos_locations"][0]
            instance["input_ids"] = torch.cat([instance["input_ids"][:start], golden_responses_cat, hard_negatives_in_batch], dim=0)
            instance["input_ids_mask"] = torch.cat([instance["input_ids_mask"][:start], golden_responses_mask_cat, hard_negatives_mask_in_batch], dim=0)   
            instance["labels"] = torch.cat([instance["labels"][:start], labels_cat, hard_negatives_labels_in_batch], dim=0)            
            instance["attention_mask"] = [1] * len(instance["input_ids"])      
            instance["token_type_ids"] = [0] * start + [1] * (len(instance["input_ids"]) - start)

            # print(self.tokenizer.decode(instance["input_ids"]))
            # print(self.tokenizer.decode(instance["input_ids_mask"]))
            # print(self.tokenizer.decode(instance["labels"]))

            instance["bos_locations"] = []
            instance["position_ids"] = list(range(start))
            for L in length_of_golden:
                instance["bos_locations"].append(start)
                start += L
                instance["position_ids"].extend(range(instance["bos_locations"][0], instance["bos_locations"][0] + L))

            # ======== add hard negatives as extra negative samples ==========
            for L in length_of_hard_negatives:
                instance["bos_locations"].append(start)
                start += L
                instance["position_ids"].extend(range(instance["bos_locations"][0], instance["bos_locations"][0] + L))
            # ================================================================
            # print(len(instance["bos_locations"]))
            instance['true_response_label'] = i
            assert len(instance["position_ids"]) == len(instance["input_ids"])
    
    def collate(self, batch):
        if self.use_in_batch:
            self.use_in_batch_negative_samples(batch)

        input_ids = pad_sequence(
            [
                instance["input_ids"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        input_ids_mask = pad_sequence(
            [
                instance["input_ids_mask"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        
        attention_mask = pad_sequence(
            [
                torch.tensor(instance["attention_mask"], dtype=torch.long)
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )
        
        labels = pad_sequence(
            [
                instance["labels"]
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=-100,
        )
        
        token_type_ids = pad_sequence(
            [
                torch.tensor(instance["token_type_ids"], dtype=torch.long)
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=self.pad,
        )
        
        true_response_label = torch.tensor(
            [
                instance["true_response_label"]
                for instance in batch
            ]
            
        )
        
        position_ids = pad_sequence(
            [
                torch.tensor(instance["position_ids"], dtype=torch.long)
                for instance in batch
            ],
            batch_first=self.batch_first,
            padding_value=0,
        )
        bos_locations = [
            instance["bos_locations"] for instance in batch
        ]


        return input_ids, input_ids_mask, token_type_ids, attention_mask, labels, true_response_label, \
               position_ids, bos_locations

