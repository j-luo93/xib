from typing import List, Sequence

import pandas as pd
import torch

from dev_misc.arglib import init_g_attr
from dev_misc.trainlib import Metric, Metrics
from xib.data_loader import MetricLearningBatch, MetricLearningDataLoader
from xib.model.metric_learning_model import MetricLearningModel

from .runner import BaseLMRunner


class BaseEvaluator:

    def __init__(self, model, data_loader):
        self.model = model
        self.data_loader = data_loader


@init_g_attr()
class LMEvaluator(BaseEvaluator, BaseLMRunner):
    """An evaluator class for LMs. Note that this is done over the entire training corpus, instead of a separate split."""

    def __init__(self, model, data_loader, feat_groups: 'p'):
        super().__init__(model, data_loader)

    def evaluate(self) -> Metrics:
        with torch.no_grad():
            self.model.eval()
            all_metrics = Metrics()
            for batch in self.data_loader:
                scores = self.model.score(batch)
                metrics = self.analyze_scores(scores)
                all_metrics += metrics
        return all_metrics


class Evaluator(BaseEvaluator):

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
            self.model.eval()
            dfs = list()
            for batch in fold_data_loader:
                output = self.model.forward(batch)
                df = pd.DataFrame()
                df['lang1'] = batch.lang1
                df['lang2'] = batch.lang2
                df['predicted_dist'] = output.cpu().numpy()
                df['dist'] = batch.dist
                dfs.append(df)
            ret = pd.concat(dfs, axis=0)
        return ret
