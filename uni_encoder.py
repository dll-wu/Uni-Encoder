import torch
import torch.nn as nn
import os
import math
from transformers import (
        BertTokenizer, 
        BertConfig,
        BertForPreTraining,
        BertModel, 
        AdamW,
        WEIGHTS_NAME,
        CONFIG_NAME,
)
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class UniEncoder(nn.Module):
    def __init__(self, config, args):
        super(UniEncoder, self).__init__()
        
        self.config = config
        self.args = args
        self.bert = BertForPreTraining(self.config)
        self.fc = nn.Linear(self.config.hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None,
                labels=None,
                true_response_label=None,
                position_ids=None,
                bos_locations=None
    ):
        # get info
        dtype = input_ids.dtype
        device = input_ids.device
        Bs, L = input_ids.shape

        if self.args.attn_map == 'square':
            # just use attention_mask: that is Square attention 
            output = self.bert(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=position_ids,
                labels=labels, return_dict=True, output_hidden_states=True
            )
        else:
            # compute candidates encoder masks
            cand_attn_mask = []
            for i in range(Bs):
                valid_length = (attention_mask[i] == 1).sum().item()
                history_length = bos_locations[i][0]
                idx = torch.arange(valid_length).unsqueeze(0).expand(len(bos_locations[i]), -1).to(device) 
                bos_pos = torch.LongTensor(bos_locations[i]).repeat(2, 1).to(device)
                bos_pos[1] = torch.roll(bos_pos[1], -1)
                bos_pos[-1, -1] = valid_length
                bos_pos = torch.transpose(bos_pos, 0, 1)  
                idx = (idx >= bos_pos[:, 0].unsqueeze(1)) & (idx < bos_pos[:, 1].unsqueeze(1)) 
                idx = torch.repeat_interleave(idx, torch.squeeze(bos_pos[:, 1] - bos_pos[:, 0]).long(), dim=0) 
                idx = idx[:, history_length:]  # responses_length * responses_length
                ## Arrow attention
                if self.args.attn_map == 'arrow':
                    idx = F.pad(idx, (history_length, 0, history_length, 0), "constant", 1) # valid_length * valid_length
                    idx = F.pad(idx, (0, L-valid_length, 0, L-valid_length), "constant", 0) # L * L
                
                ## Diagnoal attention, history attn to itself,  responses attn to themselves
                elif self.args.attn_map == 'diag':
                    idx = F.pad(idx, (history_length, 0, history_length, 0), "constant", 0) # valid_length * valid_length
                    idx_append = torch.ones(history_length, history_length).to(device)
                    idx_append = F.pad(idx_append, (0, L-history_length, 0, L-history_length), "constant", 0)
                    idx = F.pad(idx, (0, L-valid_length, 0, L-valid_length), "constant", 0) # L * L
                    idx = idx + idx_append
            
                cand_attn_mask.append(idx.long())

            cand_attn_mask = torch.stack(cand_attn_mask, dim=0) # batch_size * L * L
            
            output = self.bert(
                input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=cand_attn_mask, position_ids=position_ids,
                labels=labels, return_dict=True, output_hidden_states=True
            )

        tokens = output.hidden_states[-1] # batch_size * L * 768

        ## You can decide for yourself if you would like to add one attention step. TODO
        # attn_mask = (1 - attention_mask) * -10000
        # tokens = self.attn_with_mask(tokens, tokens, tokens, attn_mask[:,None,:])

        mean_reps = []
        # parallel, calculate the mean representation of each response 
        for i in range(len(tokens)):
            valid_length = (attention_mask[i] == 1).sum().item()
            idx = torch.arange(L).unsqueeze(0).expand(len(bos_locations[i]), -1).to(device)  
            bos_pos = torch.LongTensor(bos_locations[i]).repeat(2, 1).to(device)
            bos_pos[1] = torch.roll(bos_pos[1], -1)
            bos_pos[-1, -1] = valid_length
            bos_pos = torch.transpose(bos_pos, 0 ,1)
            idx = (idx >= bos_pos[:,0].unsqueeze(1)) & (idx < bos_pos[:,1].unsqueeze(1))
            mean_rep_per_sent = torch.matmul(idx.float(), tokens[i]) / idx.sum(1).unsqueeze(1)
            mean_reps.append(mean_rep_per_sent)
    
        attention_output = torch.stack(mean_reps, dim=0) 
        mean_logits = torch.squeeze(self.fc(attention_output))
        if labels != None : 
            maskedLMLoss = self.lossfn(output.prediction_logits.view(-1, self.config.vocab_size), labels.view(-1))
        else:
            maskedLMLoss = None

        if true_response_label != None:
            CEloss = self.lossfn(mean_logits, true_response_label)
        else:
            CEloss = None
        
        
        return mean_logits, maskedLMLoss, CEloss

    def from_pretrained(self, model_checkpoint=None):
        if model_checkpoint:
            self.load_state_dict(torch.load(model_checkpoint))
        else:
            if self.args.corpus == 'douban':
                self.bert = BertForPreTraining.from_pretrained('bert-base-chinese')
            else:
                self.bert = BertForPreTraining.from_pretrained('bert-base-uncased')

    def attn_with_mask(self, Q, K, V, mask=0):
        K_trans = torch.transpose(K, 1, 2)
        attention_scores = torch.bmm(Q, K_trans)  # batch_size * L2 * L1
        attention_scores /= math.sqrt(self.config.hidden_size)
        # add attention mask
        attention_scores = attention_scores + mask
        attention_probs = self.softmax(attention_scores)
        attention_output = torch.bmm(attention_probs, V) # batch_size * L2 * h
        return attention_output
