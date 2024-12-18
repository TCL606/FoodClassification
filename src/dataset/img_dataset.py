from torch.utils.data import Dataset
from PIL import Image
import os
import torch
# import torchvision.transforms as transforms

# data_transforms = transforms.Compose([
#     transforms.RandomHorizontalFlip(p=0.5),  # 以50%的概率进行水平翻转
#     transforms.RandomRotation(degrees=(-30, 30)),  # 在-30到30度之间随机旋转
#     transforms.ColorJitter(                  
#         brightness=0.2,
#         contrast=0.2,     # 随机调整对比度
#         saturation=0.2,   # 随机调整饱和度
#         hue=0.1           # 随机调整色调
#     ),
# ])

class ImgDataset(Dataset):
    def __init__(self, data_path, data_root, processor, test=False, use_subfig=False):
        super().__init__()
        self.data_root = data_root
        self.processor = processor
        self.data = []
        self.test = test
        self.use_subfig = use_subfig
        if self.test:
            if "test_qtcom.txt" in data_path:
                with open(data_path, 'r') as fp:
                    for line in fp:
                        line = line.strip()
                        self.data.append(os.path.join(self.data_root, line))
            else:
                with open(data_path, 'r') as fp:
                    for line in fp:
                        line = line.strip()
                        line = line.split()
                        self.data.append(os.path.join(self.data_root, line[0]))
        else:
            with open(data_path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    line = line.split()
                    self.data.append((os.path.join(self.data_root, line[0]), int(line[1])))

            # import json
            # # /mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/val/siglip_lora_lr1e-4_bs16_8gpu_20epo_4k/test/results_final.json
            # with open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/test/siglip_lora_lr1e-4_bs16_8gpu_20epo_4k/test/results_final.json", "r") as fp:
            # # with open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/src/output/val/siglip_lora_lr1e-4_bs16_8gpu_20epo_4k/test/results_final.json", "r") as fp:
            #     data_f = json.load(fp)
            # for it in data_f:
            #     self.data.append((it[2], it[0]))

            # with open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/val_qtcom.txt", 'r') as fp:
            #     for line in fp:
            #         line = line.strip()
            #         line = line.split()
            #         self.data.append((os.path.join("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/val", line[0]), int(line[1])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.test:
            img_path = self.data[index]
            # img_path = os.path.join(self.data_root, img_path)
            img = Image.open(img_path)
            inputs = self.processor(img, return_tensors="pt")
            data_dict = dict()
        else:
            img_path, label = self.data[index]
            # img_path = os.path.join(self.data_root, img_path)
            img = Image.open(img_path)
            # img = data_transforms(img)
            inputs = self.processor(img, return_tensors="pt")
            data_dict = dict()
            data_dict["label"] = label

        if self.use_subfig:
            data_dict["sub_pixel_values"] = []
            width, height = img.size

            sub_width = width // 2
            sub_height = height // 2

            for i in range(2):
                for j in range(2):
                    left = j * sub_width
                    upper = i * sub_height
                    right = left + sub_width
                    lower = upper + sub_height
                    sub_img = img.crop((left, upper, right, lower))
                    inputs_i_j = self.processor(sub_img, return_tensors="pt")
                    data_dict["sub_pixel_values"].append(inputs_i_j["pixel_values"])

            data_dict["sub_pixel_values"] = torch.cat(data_dict["sub_pixel_values"], dim=0)

        data_dict["pixel_values"] = inputs["pixel_values"]
        data_dict["ids"] = img_path
        return data_dict

def collate_img(img_lst):
    pixel_values_lst = [img["pixel_values"] for img in img_lst]
    label_lst = [img["label"] for img in img_lst]
    ids_lst = [img["ids"] for img in img_lst]

    img_tensor = torch.cat(pixel_values_lst, dim=0)
    label_tensor = torch.tensor(label_lst, dtype=torch.int64)

    if "sub_pixel_values" in img_lst[0]:
        sub_pixel_values_lst = [img["sub_pixel_values"] for img in img_lst]
        sub_pixel_values_tensor = torch.stack(sub_pixel_values_lst)

        inputs = {
            "pixel_values": img_tensor,
            "labels": label_tensor,
            "ids": ids_lst,
            "sub_pixel_values": sub_pixel_values_tensor
        }
    else:
        inputs = {
            "pixel_values": img_tensor,
            "labels": label_tensor,
            "ids": ids_lst
        }

    return inputs

def collate_img_test(img_lst):
    pixel_values_lst = [img["pixel_values"] for img in img_lst]
    ids_lst = [img["ids"] for img in img_lst]

    img_tensor = torch.cat(pixel_values_lst, dim=0)

    inputs = {
        "pixel_values": img_tensor,
        "ids": ids_lst
    }

    return inputs
