from transformers import Trainer
import torch.distributed as dist
from tqdm import tqdm
from typing import Dict

class ImgTrainer(Trainer):
    def __init__(self, train_embed=False, **kwargs):
        super().__init__(**kwargs)
        self.train_embed = train_embed

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only: bool,
        ignore_keys = None,
    ):
        if not self.train_embed:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        output = self.model(**inputs)
        loss, logits, labels = output["loss"], output["logits"], output["labels"]
        return loss, logits, labels

    def predict(self, test_dataset):
        test_dataloader = self.get_test_dataloader(test_dataset)
        if self.train_embed:
            self.prediction_loop_embed(test_dataloader)
        else:
            return self.prediction_loop(test_dataloader)

    def prediction_loop_embed(self, dataloader):
        model = self.model
        batch_size = dataloader.batch_size

        self.model.eval()
        results = []
        results = []
        if dist.get_rank() == 0:
            for inputs in tqdm(dataloader):
                label = inputs.pop("label")
                output = self.model(**inputs)
                pred = output["pred"].tolist()
                top5_pred = output["top5_pred"].tolist()
                results.append([pred, top5_pred, output["ids"]])
        else:
            for inputs in dataloader:
                output = self.model(**inputs)
                pred = output["pred"].tolist()
                top5_pred = output["top5_pred"].tolist()
                results.append([pred, top5_pred, output["ids"]])

        output_data = []
        for batch_item in results:
            for i in range(len(batch_item[0])):
                output_data.append([
                    batch_item[0][i],
                    batch_item[1][i],
                    batch_item[2][i],
                ])

        return output_data

    def prediction_loop(self, dataloader):
        model = self.model
        batch_size = dataloader.batch_size

        self.model.eval()
        results = []
        if dist.get_rank() == 0:
            for inputs in tqdm(dataloader):
                output = self.model(**inputs)
                pred = output["pred"].tolist()
                top5_pred = output["top5_pred"].tolist()
                results.append([pred, top5_pred, output["ids"]])
        else:
            for inputs in dataloader:
                output = self.model(**inputs)
                pred = output["pred"].tolist()
                top5_pred = output["top5_pred"].tolist()
                results.append([pred, top5_pred, output["ids"]])

        output_data = []
        for batch_item in results:
            for i in range(len(batch_item[0])):
                output_data.append([
                    batch_item[0][i],
                    batch_item[1][i],
                    batch_item[2][i],
                ])

        return output_data


