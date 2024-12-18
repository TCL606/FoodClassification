from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pandas as pd
import random

class ImgContrastDataset(Dataset):
    def __init__(self, data_path, data_root, processor, csv_file="/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/val/siglip_lr1e-4_bs16_8gpu_40epo_28k/test/results_final.csv", test=False):
        super().__init__()
        self.data_root = data_root
        self.processor = processor
        self.data = []
        self.test = test

        csv_data = pd.read_csv(csv_file, header=None)
        if self.test:
            if "test_qtcom.txt" in data_path:
                with open(data_path, 'r') as fp:
                    for line in fp:
                        line = line.strip()
                        self.data.append(os.path.join(self.data_root, line))
            else:
                map_dic = {}   
                with open(data_path, 'r') as fp:
                    for line in fp:
                        line = line.strip()
                        line = line.split()
                        if int(line[1]) not in map_dic:
                            map_dic[int(line[1])] = []
                        map_dic[int(line[1])].append(os.path.join(self.data_root, line[0]))

                self.map_dic = map_dic

                self.data = []
                for i in range(len(csv_data[0])):
                    crt = int(csv_data[0][i].split("val/")[1].split("/")[0])
                    top1, top2 = csv_data[1][i], csv_data[2][i]
                    if crt == top1:
                        label = 0
                    elif crt == top2:
                        label = 1
                    else:
                        label = 2
                    self.data.append((label, csv_data[0][i], top1, top2))

        else:
            self.pairs = []
            for i in range(len(csv_data[0])):
                top1, top2, top3 = csv_data[1][i], csv_data[2][i], csv_data[3][i]
                perm_lst = [[top1, top2], [top1, top3], [top2, top3]]
                perm_lst += [[x[1], x[0]] for x  in perm_lst]
                self.pairs += perm_lst

            map_dic = {}   
            with open(data_path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    line = line.split()
                    if int(line[1]) not in map_dic:
                        map_dic[int(line[1])] = []
                    map_dic[int(line[1])].append(os.path.join(self.data_root, line[0]))

            self.map_dic = map_dic
            
            # with open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/val_qtcom.txt", 'r') as fp:
            #     for line in fp:
            #         line = line.strip()
            #         line = line.split()
            #         self.data.append((os.path.join("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/val", line[0]), int(line[1])))

    def __len__(self):
        if self.test:
            return len(self.data)
        else:
            return len(self.pairs)

    def __getitem__(self, index):
        if self.test:
            label, img_path, pos, neg = self.data[index]

            img = Image.open(img_path)
            inputs = self.processor(img, return_tensors="pt")
            
            data_dict = dict()
            data_dict["pos_values"] = pos
            data_dict["neg_values"] = neg
            data_dict["label"] = label
        else:
            pair = self.pairs[index]
            pos, neg = pair[0], pair[1]
            img_path = random.choice(self.map_dic[pos])

            img = Image.open(img_path)
            inputs = self.processor(img, return_tensors="pt")

            data_dict = dict()
            data_dict["pos_values"] = pos
            data_dict["neg_values"] = neg

        data_dict["pixel_values"] = inputs["pixel_values"]
        data_dict["ids"] = [img_path, pos, neg]
        return data_dict

def collate_pair_img(img_lst):
    pixel_values_lst = [img["pixel_values"] for img in img_lst]
    pos_values = torch.tensor([img["pos_values"] for img in img_lst])
    neg_values = torch.tensor([img["neg_values"] for img in img_lst])
    ids_lst = [img["ids"] for img in img_lst]
    if "label" in img_lst[0]:
        label_lst = [img["label"] for img in img_lst]
        label_tensor = torch.tensor(label_lst, dtype=torch.int64)

    img_tensor = torch.cat(pixel_values_lst, dim=0)
    
    if "label" in img_lst[0]:
        inputs = {
            "pixel_values": img_tensor,
            "pos_values": pos_values,
            "neg_values": neg_values,
            "label": label_tensor,
            "ids": ids_lst
        }
    else:
        inputs = {
            "pixel_values": img_tensor,
            "pos_values": pos_values,
            "neg_values": neg_values,
            "ids": ids_lst
        }

    return inputs

if __name__ == "__main__":
    data_path = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/train_qtcom.txt"
    data_root = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/Train_qtc"
    from transformers import AutoImageProcessor
    processor = AutoImageProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-patch14-384")
    ds = ImgContrastDataset(data_path, data_root, processor)
    print(ds[0]["pos_values"].shape)
    breakpoint()