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
    "--parallel",
    action="store_true",
    help="Use DataParallel or not",
)
parser.add_argument(
    "--use_post_training", type=str, default="NoPost", help="Post training"
)
parser.add_argument(
    "--corpus",
    type=str,
    default="douban",
    help="The corpus you use. Should be: ubuntu, douban or personachat",
)
parser.add_argument(
    "--attn_map",
    type=str,
    default="arrow",
    help="The attention mechanism: arrow, diag, square",
)
args = parser.parse_args()
ranker = Ranker(args.model_checkpoint, args) 
data_path = 'data/douban.json'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.loads(f.read())

test_data = data['test']
print('test samples:', len(test_data))
total_samples = len(test_data)
acc = 0

def mean_average_precision(sort_data):
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index+1)
    return sum_precision / count_1

def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))

def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0

def recall_at_position_k_in_10(sort_data, k):
    sort_label = [s_d[1] for s_d in sort_data]
    select_label = sort_label[:k]
    return 1.0 * select_label.count(1) / sort_label.count(1)



sum_m_a_p = []
sum_m_r_r = []
sum_p_1 = []
sum_r_1 = []
sum_r_2 = []
sum_r_5 = []
for i in tqdm(range(len(test_data))):
    turn = test_data[i]
    history = turn["history"]
    responses = turn["responses"]
    labels = turn["labels"]
    _, mean_logits = ranker.rank(history, responses)
    mean_logits = mean_logits.data.cpu().numpy().tolist()
    map_list = zip(mean_logits, labels)
    map_list = list(map_list)
    sort_data = sorted(map_list, key=lambda x: x[0], reverse=True)

    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1   = precision_at_position_1(sort_data)
    r_1   = recall_at_position_k_in_10(sort_data, 1)
    r_2   = recall_at_position_k_in_10(sort_data, 2)
    r_5   = recall_at_position_k_in_10(sort_data, 5)

    sum_m_a_p.append(m_a_p)
    sum_m_r_r.append(m_r_r)
    sum_p_1.append(p_1)
    sum_r_1.append(r_1)
    sum_r_2.append(r_2)
    sum_r_5.append(r_5)

sum_m_a_p = sum(sum_m_a_p) / len(sum_m_a_p)
sum_m_r_r = sum(sum_m_r_r) / len(sum_m_r_r)
sum_p_1 = sum(sum_p_1) / len(sum_p_1)
sum_r_1 = sum(sum_r_1) / len(sum_r_1)
sum_r_2 = sum(sum_r_2) / len(sum_r_2)
sum_r_5 = sum(sum_r_5) / len(sum_r_5)

print("MAP: {}".format(sum_m_a_p))
print("MRR: {}".format(sum_m_r_r))
print("P@1: {}".format(sum_p_1))
print("R10@1: {}".format(sum_r_1))
print("R10@2: {}".format(sum_r_2))
print("R10@5: {}".format(sum_r_5))