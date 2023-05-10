export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
nohup python -m torch.distributed.launch --nproc_per_node=6 --master_port 1234 train_dist.py \
       --data_path data/douban.json \
       --train_batch_size 7 \
       --valid_batch_size 20 \
       --gradient_accumulation_steps 1 \
       --num_workers 24 \
       --scheduler noam \
       --n_epochs 50 \
       --dataset_cache \
       --lr 3e-5 > logs.out &