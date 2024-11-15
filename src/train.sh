#!/bin/bash

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

export WANDB_PROJECT=TCL_DataHw

output_dir=output/debug # siglip_lr1e-4_bs16_8gpu_40epo_28k # debug # siglip_lr1e-4_bs16_8gpu_20epo_ctn15620 # debug # 
resnet_model=models/siglip-so400m-patch14-384

torchrun --nproc_per_node=8 --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port=42586 \
    train.py \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 40 \
    --output_dir $output_dir \
    --eval_steps 4000 \
    --save_steps 4000 \
    --resnet_model $resnet_model \
    --model_type siglip_subfig \
    --freeze_vm True \
    --train_txt data/train_qtcom.txt \
    --train_data_root data/Train_qtc \
    --eval_txt data/val_qtcom.txt \
    --eval_data_root data/val \
    --test_txt data/test_qtcom.txt \
    --test_data_root data/test_new \
    --do_test False \
    --do_eval True \
    --ckpt output/siglipSubfig_lr1e-4_bs16_8gpu_40epo/checkpoint-16000
