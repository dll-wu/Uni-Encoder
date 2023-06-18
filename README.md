# Uni-Encoder: A Fast and Accurate Response Selection Paradigm for Generation-Based Dialogue Systems

*Uni-Encoder* is a state-of-the-art response selection paradigm for generation-based dialogue systems. It has been tested on PersonaChat, Ubuntu V1, Ubuntu V2, and Douban datasets, achieving high accuracy and fast response times. Our [paper](https://arxiv.org/abs/2106.01263) has been accepted to the Findings of ACL 2023. This repo contains the implementation of Uni-Encoder and procedures to reproduce the experimental results.

![alt text](PNG/uni_results.png)

## Dependencies
The code is implemented using python 3.8 and PyTorch v1.8.1 (please choose the correct command that match your CUDA version from [PyTorch](https://pytorch.org/get-started/previous-versions/))

Anaconda / Miniconda is recommended to set up this codebase.

You may use the command below(cuda 11+):
```shell
conda create -n UniEncoder python=3.8

conda activate UniEncoder
cd Uni-Encoder

pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

## Training
Training codes for the Ubuntu (v1 and v2), Persona-Chat and Douban datasets are available in separate folders, all sharing a common model file, `uni_ecoder.py`.

Standardized data for each dataset can be downloaded from the following links:
- [Ubuntu v1](https://pan.baidu.com/s/10MeOilheRRfxdy-5LajQrA?pwd=3rgc) (Please put the dataset in the "ubuntu/data".)
- [Ubuntu v2](https://pan.baidu.com/s/1rtsigkgmjm-A5lYAyw6Nyw?pwd=pp5v) (Please put the dataset in the "ubuntu/data".)
- [Persona Chat](https://pan.baidu.com/s/1EEvQGn5nS5VoKTV54HTlDw?pwd=j8de) (Please put the dataset in the "persona-chat/data".)
- [Douban](https://pan.baidu.com/s/1EEvQGn5nS5VoKTV54HTlDw?pwd=j8de) (Please put the dataset in the "douban/data".)

Each dataset folder includes its corresponding training script and other utilities. For example, in the `ubuntu` folder, there is a `run_dist.sh` file:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
nohup python -m torch.distributed.launch --master_port 1234 --nproc_per_node=6 train_dist.py \
       --data_path data/UbuntuV2_data.json \ # change to V1 if needed
       --train_batch_size 8 \
       --valid_batch_size 8 \
       --gradient_accumulation_steps 1 \
       --num_workers 24 \
       --scheduler noam \
       --n_epochs 20 \
       --lr 2e-4 > logs.out &
```

To train Uni-Encoder, run the following command:
```bash
cd ubuntu
bash run_dist.sh
```


You may adjust the batch size and the number of GPUs based on your device. Training data will be tokenized and cached after the first run, and you can accelerate subsequent runs by adding the ```--dataset_cache``` option.

If you want to finetune the post-trained [BERT-FP(Han et al.)](https://aclanthology.org/2021.naacl-main.122/) model, you need to first download its checkpoint from [here](https://github.com/hanjanghoon/BERT_FP) and place it in the "post_training_model" folder. Then use ```--use_post_training bert_fp```.

During training, the model is saved after each epoch, and you can find the "xxx.pt" model and the "training.log" file containing the details of the training in the "runs" directory. If an error occurs, the error message will be displayed in the "logs.out" file.

## Inference 
- We have released all the finetuned checkpoints for evaluation. Download instructions are provided below.

- Inference command are provided in the following, ```--model_checkpoint``` specifies the path of the checkpoint.

- Please also refer to the "Training" section to download the standardized dataset.

### PersonChat

- Please download the processed data and [model checkpoint (about 389M)](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/EVB9Gi_cmXdEvQNE_YN7w-MBAv751am1G-zmRlfr2xIqeQ?e=XhfgFc).
- Inference
    ```shell
    unzip checkpoint_for_PersonaChat.zip -d persona-chat/
    cd persona-chat/
    python test.py --model_checkpoint checkpoint_for_PersonaChat
    ```

### Ubuntu V1 and V2

- Please download the processed data and [model checkpoints (about 1.14G)](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/EVtRu4j7HCpGhAiCCn8v8acB8ANGsSwvAmyybWSkaEb0SA?e=o9G6EM).

- Inference
    ```shell
    unzip checkpoint_for_Ubuntu.zip -d ubuntu/
    cd ubuntu/

    # inference on Ubuntu V1
    python test.py --data_path data/UbuntuV1_data.json --model_checkpoint checkpoint_for_UbuntuV1

    # inference on Ubuntu V2
    python test.py --data_path data/UbuntuV2_data.json --model_checkpoint checkpoint_for_UbuntuV2

    # inference on Ubuntu V1(finetune from the checkpoint given by BERT-FP)
    python test.py --data_path data/UbuntuV1_data.json --use_post_training bert_fp --model_checkpoint checkpoint_for_UbuntuV1_use_bert_FP
    ```

### Douban

- Please download the processed data and [model checkpoint (about 729M)](https://westlakeu-my.sharepoint.com/:u:/g/personal/hehongliang_westlake_edu_cn/EVF2RYknM6NBrl0_yhKf-moBCIq97Jg_qHHrrhqjqJKIIQ?e=FQjG3r).
- Inference
    ```shell
    unzip checkpoint_for_Douban.zip -d douban/
    cd douban/

    # inference on Douban
    python test.py --model_checkpoint checkpoint_for_Douban

    # inference on Douban(finetune from the checkpoint given by BERT-FP)
    python test.py --use_post_training bert_fp --model_checkpoint checkpoint_for_Douban_use_bert_FP
    ```
