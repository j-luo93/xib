from typing import Sequence
from trainlib import Metrics
import torch

from xib.data_loader import MetricLearningDataLoader
from xib.model import MetricLearningModel


class Evaluator:

    def __init__(self, model, data_loader: MetricLearningDataLoader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self, dev_langs: Sequence[str]) -> Metrics:
        self.data_loader.select(dev_langs)

        metrics = Metrics()
        with torch.no_grad():
            self.model.eval()
            for batch in self.data_loader:
                output = self.model(batch)
                mse = (output - batch.dist) ** 2
                mse = Metric('mse', mse.sum(), len(batch))
                metrics += mse
        return metrics
