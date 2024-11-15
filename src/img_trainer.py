from transformers import Trainer
import torch.distributed as dist
from tqdm import tqdm

class ImgTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, test_dataset):
        test_dataloader = self.get_test_dataloader(test_dataset)
        return self.prediction_loop(test_dataloader)

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
