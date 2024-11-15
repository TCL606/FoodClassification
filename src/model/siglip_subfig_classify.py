import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoProcessor, AutoModel
from PIL import Image

class SiglipSubfigClassify(nn.Module):
    def __init__(self, path):
        super().__init__()
        self.model = AutoModel.from_pretrained(path).vision_model
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.model.config.hidden_size * 5, out_features=1000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000, bias=True)
        )

    def forward(self, ids, pixel_values, sub_pixel_values, labels=None):
        feat = self.model(pixel_values)
        sub_pixel_values = sub_pixel_values.view(-1, sub_pixel_values.size(-3), sub_pixel_values.size(-2), sub_pixel_values.size(-1))
        feat_sub = self.model(sub_pixel_values)
        feat_sub = feat_sub.pooler_output
        feat_sub = feat_sub.view(pixel_values.size(0), -1)
        feat_all = torch.cat([feat.pooler_output, feat_sub], dim=-1)
        
        logits = self.classifier(feat_all)
        if labels is None:
            pred = torch.argmax(logits, dim=-1)
            top_k_values, top_k_indices = torch.topk(logits, 5, dim=-1)
            return {"pred": pred, "top5_pred": top_k_indices, "ids": ids, "logits": logits}
        else:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}

    