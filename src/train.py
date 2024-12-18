import torch
from torch import nn
from transformers import TrainingArguments, HfArgumentParser
from transformers import AutoImageProcessor
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass, field
import random
import torch.distributed as dist
import os
import numpy as np
from img_trainer import ImgTrainer
from model.resnet_classify import ResnetClassify
from model.siglip_classify import SiglipClassify
from model.siglip_subfig_classify import SiglipSubfigClassify
from model.navit_classify import NavitClassify
from model.resnet_embed import ResnetEmbed
from model.siglip_embed import SiglipEmbed
from model.siglip_lora_classify import SiglipLoRAClassify
from dataset.img_dataset import ImgDataset, collate_img, collate_img_test
from dataset.img_contrast_dataset import ImgContrastDataset, collate_pair_img
import json

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir: str = "./output/results"
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 1000
    learning_rate: float = 2e-5
    logging_steps: int = 10
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 5
    dataloader_num_workers: int = 16
    seed: int = 2024
    metric_for_best_model: str = "accuracy"
    resnet_model: str = "microsoft/resnet-50"
    train_txt: str = "data/train_qtcom.txt"
    train_data_root: str = "data/Train_qtc"
    eval_txt: str = "data/val_qtcom.txt"
    eval_data_root: str = "data/val"
    do_test: bool = False
    test_txt: str = "data/test_qtcom.txt"
    test_data_root: str = "data/test_new"
    ckpt: str = None
    model_type: str = "resnet"
    freeze_vm: bool = False
    use_subfig: bool = False
    do_eval: bool = False

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    accuracy = (preds == labels).mean()
    return {'accuracy': accuracy}

def main():
    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    training_args.logging_dir = os.path.join(training_args.output_dir, "logs")

    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    assert training_args.model_type in ["resnet", "siglip", "siglip_subfig", "navit", "resnet_embed", "siglip_embed", "siglip_lora"]
    if training_args.model_type == "siglip_subfig":
        training_args.use_subfig = True
        
    if training_args.model_type == "resnet":
        model = ResnetClassify(training_args.resnet_model)
    elif training_args.model_type == "siglip":
        model = SiglipClassify(training_args.resnet_model)
    elif training_args.model_type == "siglip_subfig":
        model = SiglipSubfigClassify(training_args.resnet_model)
    elif training_args.model_type == "navit":
        model = NavitClassify(training_args.resnet_model)
    elif training_args.model_type == "resnet_embed":
        model = ResnetEmbed(training_args.resnet_model)
    elif training_args.model_type == "siglip_embed":
        model = SiglipEmbed(training_args.resnet_model)
    elif training_args.model_type == "siglip_lora":
        model = SiglipLoRAClassify(training_args.resnet_model)

    train_embed = "embed" in training_args.model_type
    if not training_args.do_test:
        for k, p in model.named_parameters():
            p.requires_grad = True
        
        if training_args.freeze_vm:
            if "siglip" in training_args.model_type or "navit" in training_args.model_type:
                for k, p in model.model.named_parameters():
                    p.requires_grad = False
            
        for k, p in model.model.named_parameters():
            if 'lora' in k:
                p.requires_grad = True

        # try:
        #     processor = AutoImageProcessor.from_pretrained(training_args.resnet_model)
        # except:
        processor = AutoImageProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-patch14-384")
        if training_args.model_type == "navit":
            processor.size = {"height": 980, "width": 980}
        
        if train_embed:
            train_dataset = ImgContrastDataset(training_args.train_txt, training_args.train_data_root, processor, csv_file="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/val/siglip_lr1e-4_bs16_8gpu_40epo_28k/test/results_final.csv")
            eval_dataset = ImgContrastDataset(training_args.eval_txt, training_args.eval_data_root, processor, csv_file="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/val/siglip_lr1e-4_bs16_8gpu_40epo_28k/test/results_final.csv", test=True)
            data_collator = collate_pair_img
        else:
            train_dataset = ImgDataset(training_args.train_txt, training_args.train_data_root, processor, use_subfig=training_args.use_subfig)
            eval_dataset = ImgDataset(training_args.eval_txt, training_args.eval_data_root, processor, use_subfig=training_args.use_subfig)
            data_collator = collate_img

        temp_cnt, temp_total = 0, 0
        if dist.get_rank() == 0:
            for k, p in model.named_parameters():
                temp_total += 1
                if p.requires_grad:
                    print(k)
                    temp_cnt += 1
            print(temp_cnt, temp_total)
        
        trainer = ImgTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            train_embed=train_embed
        )

        if training_args.ckpt is not None:
            trainer._load_from_checkpoint(training_args.ckpt)

        # if training_args.do_eval:
        #     trainer.evaluate()

        # else:
        trainer.train()
        trainer.save_model("final_model")

    else:
        # processor = AutoImageProcessor.from_pretrained(training_args.resnet_model)
        processor = AutoImageProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-patch14-384")
        if training_args.model_type == "navit":
            processor.size = {"height": 980, "width": 980}
        test_dataset = ImgDataset(training_args.test_txt, training_args.test_data_root, processor, test=True)
        trainer = ImgTrainer(
            model=model,
            args=training_args,
            train_dataset=test_dataset,
            eval_dataset=test_dataset,
            data_collator=collate_img_test,
            train_embed=train_embed
        )

        trainer._load_from_checkpoint(training_args.ckpt)
        output_data = trainer.predict(test_dataset)
        if dist.get_rank() == 0:
            os.makedirs(os.path.join(training_args.output_dir, "test"), exist_ok=True)

        dist.barrier()
        with open(os.path.join(training_args.output_dir, "test", f"results_{dist.get_rank()}.json"), 'w') as fp:
            json.dump(output_data, fp)

        dist.barrier()

        if dist.get_rank() == 0:
            res = []
            print("Start Merging")
            for i in range(dist.get_world_size()):
                with open(os.path.join(training_args.output_dir, "test", f"results_{i}.json"), 'r') as fp:
                    data_i = json.load(fp)
                res += data_i
            with open(os.path.join(training_args.output_dir, "test", f"results_final.json"), 'w') as fp:
                json.dump(res, fp, indent=4)
            print(os.path.join(training_args.output_dir, "test", f"results_final.json"))

        dist.barrier()

if __name__ == "__main__":
    main()
