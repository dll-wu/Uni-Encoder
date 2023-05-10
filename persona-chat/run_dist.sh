#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
nohup python -m torch.distributed.launch --master_port 1234 --nproc_per_node=6 train_dist.py \
       --data_path data/PersonaChat.json \
       --train_batch_size 14 \
       --valid_batch_size 12 \
       --gradient_accumulation_steps 1 \
       --num_workers 24 \
       --scheduler noam \
       --n_epochs 50 \
       --lr 5e-5 > logs.out &