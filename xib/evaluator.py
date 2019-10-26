from typing import List, Sequence

import pandas as pd
import torch

from trainlib import Metric, Metrics
from xib.data_loader import MetricLearningBatch, MetricLearningDataLoader
from xib.model import MetricLearningModel


class Evaluator:

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader

    def evaluate(self, dev_langs: List[str]) -> Metrics:
        metrics = Metrics()
        fold_data_loader = self.data_loader.select(dev_langs, self.data_loader.all_langs)
        with torch.no_grad():
            self.model.eval()
            for batch in fold_data_loader:
                output = self.model(batch)
                mse = (output - batch.dist) ** 2
                mse = Metric('mse', mse.sum(), len(batch))
                metrics += mse
        return metrics

    def predict(self, dev_langs: List[str]) -> pd.DataFrame:
        fold_data_loader = self.data_loader.select(dev_langs, self.data_loader.all_langs)
        with torch.no_grad():
            model.eval()
            dfs = list()
            for batch in fold_data_loader:
                output = self.forward(batch)
                df = pd.DataFrame()
                df['lang1'] = batch.lang1
                df['lang2'] = batch.lang2
                df['predicted_dist'] = output.cpu().numpy()
                df['dist'] = batch.dist
                dfs.append(df)
            ret = pd.concat(dfs, axis=0)
        return ret
