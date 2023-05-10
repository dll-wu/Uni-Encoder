from argparse import ArgumentParser
from rank import Ranker
import torch
import json
import math
import os
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = ArgumentParser()
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
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device (cuda or cpu)",
)
parser.add_argument(
    "--corpus",
    type=str,
    default="personachat",
    help="The corpus you use. Should be: ubuntu, douban or persona-chat",
)
parser.add_argument(
    "--attn_map",
    type=str,
    default="arrow",
    help="The attention mechanism: arrow, diag, square",
)
args = parser.parse_args()
ranker = Ranker(args.model_checkpoint, args) 

data_path = 'data/PersonaChat.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

test_data = data['test']
print('test samples:', len(test_data))
total_samples = len(test_data)
R10_1 = 0
R10_5 = 0
MRR = 0
for i in tqdm(range(len(test_data))):
    turn = test_data[i]
    persona = turn["your_persona"]
    history = turn["history"]
    responses = turn["responses"]
    label = turn["label"]
    pred_label, descend_labels = ranker.rank(persona, history, responses)
    if pred_label == label:
        R10_1 += 1
    if label in descend_labels[:5]:
        R10_5 += 1
    MRR += 1.0 / (descend_labels.index(label) + 1)

print('R10@1:', R10_1/total_samples)
print('R10@5:', R10_5/total_samples)
print('MRR:', MRR/total_samples)