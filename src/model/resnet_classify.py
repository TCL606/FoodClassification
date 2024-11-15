import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image

class ResnetClassify(nn.Module):
    def __init__(self, resnet_path):
        super().__init__()
        model = ResNetForImageClassification.from_pretrained(resnet_path)
        self.resnet = model.resnet
        self.classifier = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=model.config.hidden_sizes[-1], out_features=1000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000, bias=True)
        )

    def forward(self, ids, pixel_values, labels=None):
        feat = self.resnet(pixel_values)
        logits = self.classifier(feat.pooler_output)
        if labels is None:
            pred = torch.argmax(logits, dim=-1)
            top_k_values, top_k_indices = torch.topk(logits, 5, dim=-1)
            return {"pred": pred, "top5_pred": top_k_indices, "ids": ids, "logits": logits}
        else:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

if __name__ == "__main__":
    processor = AutoImageProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-50")
    image = Image.open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/Train_qtc/904/8510136.jpg")
    inputs = processor(image, return_tensors="pt")
    model = ResnetClassify("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-50")
    logits = model(inputs['pixel_values']).logits
    