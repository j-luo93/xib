import logging
import os
import random

from arglib import add_argument, init_g_attr
from trainlib import Metrics, set_random_seeds
from xib.data_loader import (ContinuousTextDataLoader, IpaDataLoader,
                             MetricLearningDataLoader)
from xib.evaluator import Evaluator
from xib.model import DecipherModel, MetricLearningModel, Model
from xib.trainer import DecipherTrainer, LMTrainer, MetricLearningTrainer

add_argument('task', default='lm', dtype=str, choices=['lm', 'decipher', 'metric'], msg='which task to run')


class Manager:

    data_loader_cls = IpaDataLoader
    trainer_cls = LMTrainer

    def __init__(self):
        self.model = self._get_model()
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        self.train_data_loader = self.data_loader_cls()
        self.trainer = self.trainer_cls(self.model, self.train_data_loader)

    def _get_model(self):
        return Model()

    def train(self):
        self.trainer.train()


class DecipherManager(Manager):

    data_loader_cls = ContinuousTextDataLoader
    trainer_cls = DecipherTrainer

    def _get_model(self):
        return DecipherModel(None)  # FIXME(j_luo) fill in pretrained lm_model.


@init_g_attr(default='property')
class MetricLearningManager(Manager):

    data_loader_cls = MetricLearningDataLoader
    trainer_cls = MetricLearningTrainer

    add_argument('k_fold', default=10, dtype=int, msg='number of folds for cross validation')

    def __init__(self, k_fold, random_seed):
        super().__init__()
        self.evaluator = Evaluator(self.model, self.train_data_loader)

    def _get_model(self):
        return MetricLearningModel()

    def train(self):
        all_langs = self.train_data_loader.all_langs
        num_langs = len(all_langs)
        idx = list(range(num_langs))
        random.shuffle(idx)

        num_langs_per_fold = num_langs // self.k_fold

        accum_metrics = Metrics()
        for fold in range(self.k_fold):
            # Get train-dev split.
            start_idx = fold * num_langs_per_fold
            end_idx = start_idx + num_langs_per_fold if fold < self.k_fold - 1 else num_langs
            dev_langs = [all_langs[idx[i]] for i in range(start_idx, end_idx)]
            train_langs = [all_langs[idx[i]] for i in range(num_langs) if i < start_idx or i >= end_idx]
            assert len(set(dev_langs + train_langs)) == num_langs

            set_random_seeds(self.random_seed)
            self.trainer.reset()
            best_mse = self.trainer.train(train_langs, self.evaluator, dev_langs)

            # Aggregate every fold.
            accum_metrics += best_mse

        logging.info(accum_metrics.get_table())
