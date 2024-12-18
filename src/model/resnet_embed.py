import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image

class ResnetEmbed(nn.Module):
    def __init__(self, resnet_path):
        super().__init__()
        model = ResNetForImageClassification.from_pretrained(resnet_path)
        self.resnet = model.resnet
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=model.config.hidden_sizes[-1], out_features=1024, bias=True),
        )
        self.embeds = nn.Embedding(1000, 1024)

    def forward(self, pixel_values, pos_values, neg_values, ids, label=None):
        pixel_feat = self.head(self.resnet(pixel_values).pooler_output)
        pixel_feat = pixel_feat.unsqueeze(dim=1)

        pos_feat = self.embeds(pos_values)
        neg_feat = self.embeds(neg_values)

        all_feat = torch.cat([pos_feat.unsqueeze(dim=1), neg_feat.unsqueeze(dim=1)], dim=1)

        logits = F.cosine_similarity(pixel_feat, all_feat, dim=-1)
        pos_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, pos_labels)
        if label is None:
            return {"loss": loss, "logits": logits, "ids": ids}
        else:
            return {"loss": loss, "logits": logits, "labels": label, "ids": ids}

if __name__ == "__main__":
    processor = AutoImageProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-50")
    image = Image.open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/Train_qtc/904/8510136.jpg")
    inputs = processor(image, return_tensors="pt")
    model = ResnetClassify("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/resnet-50")
    output = model(inputs['pixel_values'], inputs['pixel_values'], inputs['pixel_values'])
    print(output)
    