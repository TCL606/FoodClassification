import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image
from transformers import SiglipImageProcessor
import sys
sys.path.append("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-14-980-flash-attn2-navit")
from modeling_siglip import SiglipVisionModel

class NavitClassify(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = SiglipVisionModel.from_pretrained(path, _flash_attn_2_enabled=False).vision_model
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size, out_features=1000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000, bias=True)
        )
        # self.classifier.to(torch.float16)

    def forward(self, ids, pixel_values, labels=None):
        # pixel_values = pixel_values.to(torch.float16)
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
    path = "/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-14-980-flash-attn2-navit"
    processor = SiglipImageProcessor.from_pretrained("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/models/siglip-so400m-patch14-384")
    image = Image.open("/mnt/bn/tiktok-mm-4/aiic/users/tangchangli/test/meituan_hw/data/Train_qtc/904/8510136.jpg")
    breakpoint()
    inputs = processor(images=image, return_tensors="pt", size={"height": 980, "width": 980})
    model = NavitClassify(path)

    model = model.cuda()
    inputs["pixel_values"] = inputs["pixel_values"].cuda()
    inputs['ids'] = 1
    output = model(**inputs)
    breakpoint()
