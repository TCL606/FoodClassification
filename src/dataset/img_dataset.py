from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class ImgDataset(Dataset):
    def __init__(self, data_path, data_root, processor, test=False, use_subfig=False):
        super().__init__()
        self.data_root = data_root
        self.processor = processor
        self.data = []
        self.test = test
        self.use_subfig = use_subfig
        if self.test:
            with open(data_path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    self.data.append(line)
        else:
            with open(data_path, 'r') as fp:
                for line in fp:
                    line = line.strip()
                    line = line.split()
                    self.data.append((line[0], int(line[1])))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.test:
            img_path = self.data[index]
            img_path = os.path.join(self.data_root, img_path)
            img = Image.open(img_path)
            inputs = self.processor(img, return_tensors="pt")
            data_dict = dict()
        else:
            img_path, label = self.data[index]
            img_path = os.path.join(self.data_root, img_path)
            img = Image.open(img_path)
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
