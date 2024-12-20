#!/bin/bash

PROJECT_ROOT=$(cd $(dirname $0); pwd)
cd $PROJECT_ROOT

export WANDB_PROJECT=TCL_DataHw

# output_dir="output/debug"
#    --ckpt /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/resnet_lr2e-3_bs16_8gpu_20epo/checkpoint-15000 \
    # --ckpt /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/siglip_lr1e-4_bs16_8gpu_20epo/checkpoint-15620

    # --lr_scheduler_type "cosine" \

export CUDA_VISIBLE_DEVICES=0
output_dir=output/siglip_lora_lr1e-4_bs16_1gpu_20epo # val/siglip_lr1e-4_bs16_8gpu_40epo_28k # debug # siglip_lr1e-4_bs16_8gpu_20epo_ctn15620 # debug # output/debug # 
resnet_model=/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-patch14-384 # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-50 # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-14-980-flash-attn2-navit # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/clip-vit-large-patch14 # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-152 # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-34 # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-50

torchrun --nproc_per_node=1 --nnodes="${ARNOLD_WORKER_NUM}" --node_rank="${ARNOLD_ID}" --master_addr="${METIS_WORKER_0_HOST}" --master_port=22286 \
    train.py \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --num_train_epochs 20 \
    --output_dir $output_dir \
    --eval_steps 4000 \
    --save_steps 4000 \
    --resnet_model $resnet_model \
    --model_type siglip_lora \
    --freeze_vm True \
    --train_txt /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/train_qtcom.txt \
    --train_data_root /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/Train_qtc \
    --eval_txt /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/val_qtcom.txt \
    --eval_data_root /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/val \
    --test_txt /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/test_qtcom.txt \
    --test_data_root /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/test_new \
    --do_eval True

    # --do_test True --ckpt /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/siglip_lora_augm_lr1e-4_bs16_8gpu_20epo_useFakeTest/checkpoint-6000
