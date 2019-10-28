import logging
import os
import random

from arglib import add_argument, init_g_attr
from trainlib import Metrics, set_random_seeds
from xib.data_loader import (ContinuousTextDataLoader, IpaDataLoader,
                             MetricLearningDataLoader, SparseIpaDataLoader)
from xib.evaluator import Evaluator
from xib.model.decipher_model import DecipherModel
from xib.model.lm_model import AdaptedLMModel, LMModel
from xib.model.metric_learning_model import MetricLearningBatch
from xib.trainer import (AdaptLMTrainer, DecipherTrainer, LMTrainer,
                         MetricLearningTrainer)

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
        return LMModel()

    def train(self):
        self.trainer.train()


class AdaptManager(Manager):

    data_loader_cls = SparseIpaDataLoader
    trainer_cls = AdaptLMTrainer

    def _get_model(self):
        return AdaptedLMModel()


class DecipherManager(Manager):

    data_loader_cls = ContinuousTextDataLoader
    trainer_cls = DecipherTrainer

    def _get_model(self):
        return DecipherModel()

# ------------------------------------------------------------- #
#                         Metric learner                        #
# ------------------------------------------------------------- #


@init_g_attr(default='property')
class MetricLearningManager(Manager):

    add_argument('k_fold', default=10, dtype=int, msg='number of folds for cross validation')

    def __init__(self, k_fold, random_seed, data_path, emb_groups, family_file_path):
        self.model = MetricLearningModel()
        self.data_loader = MetricLearningDataLoader()
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        self.trainer = MetricLearningTrainer(self.model, self.data_loader)
        self.evaluator = Evaluator(self.model, self.data_loader)

    def train(self):
        set_random_seeds(self.random_seed)
        all_langs = self.data_loader.all_langs
        num_langs = len(all_langs)
        idx = list(range(num_langs))
        random.shuffle(idx)

        num_langs_per_fold = (num_langs + self.k_fold - 1) // self.k_fold

        accum_metrics = Metrics()
        for fold in range(self.k_fold):
            # Get train-dev split.
            start_idx = fold * num_langs_per_fold
            end_idx = start_idx + num_langs_per_fold if fold < self.k_fold - 1 else num_langs
            dev_langs = [all_langs[idx[i]] for i in range(start_idx, end_idx)]
            logging.imp(f'dev_langs: {sorted(dev_langs)}')
            train_langs = [all_langs[idx[i]] for i in range(num_langs) if i < start_idx or i >= end_idx]
            assert len(set(dev_langs + train_langs)) == num_langs

            self.trainer.reset()
            best_mse = self.trainer.train(
                self.evaluator,
                train_langs,
                dev_langs,
                fold)

            # Aggregate every fold.
            accum_metrics += best_mse

        logging.info(accum_metrics.get_table())
