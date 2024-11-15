import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image

class SiglipClassify(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = AutoModel.from_pretrained(path).vision_model
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=1000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000, bias=True)
        )

    def forward(self, ids, pixel_values, labels=None):
        feat = self.model(pixel_values)
        logits = self.classifier(feat.pooler_output)
        if labels is None:
            pred = torch.argmax(logits, dim=-1)
            top_k_values, top_k_indices = torch.topk(logits, 5, dim=-1)
            return {"pred": pred, "top5_pred": top_k_indices, "ids": ids, "logits": logits}
        else:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

if __name__ == "__main__":
    path = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/clip-vit-large-patch14"
    processor = AutoProcessor.from_pretrained(path)
    image = Image.open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/Train_qtc/904/8510136.jpg")
    inputs = processor(images=image, return_tensors="pt")
    model = SiglipClassify(path)

    model = model.cuda()
    inputs["pixel_values"] = inputs["pixel_values"].cuda()
    output = model(**inputs)
    breakpoint()