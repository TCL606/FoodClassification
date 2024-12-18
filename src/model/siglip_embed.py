import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image
from peft import LoraConfig, get_peft_model

class SiglipEmbed(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = AutoModel.from_pretrained(path).vision_model
        self.lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "k_proj", "v_proj"],
            lora_dropout=0.05,
        )
        self.model = get_peft_model(self.model, self.lora_config)

        self.head = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=1024, bias=True),
            nn.GELU(),
            nn.Linear(in_features=1024, out_features=1024, bias=True)
        )
        self.embeds = nn.Embedding(1000, 1024)

    def forward(self, pixel_values, pos_values, neg_values, ids, label=None):
        pixel_feat = self.head(self.model(pixel_values).pooler_output).unsqueeze(1)

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
    