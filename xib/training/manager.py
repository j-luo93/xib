import logging
import os
import random

import torch
import torch.nn as nn

from dev_misc import g
from dev_misc.arglib import add_argument, init_g_attr
from dev_misc.trainlib import Metrics, has_gpus, set_random_seeds
from dev_misc.trainlib.trainer import freeze
from dev_misc.utils import deprecated
from xib.data_loader import (ContinuousTextDataLoader, DataLoaderRegistry,
                             DenseIpaDataLoader, IpaDataLoader)
from xib.model.decipher_model import DecipherModel
from xib.model.lm_model import LM, AdaptedLM
from xib.search.searcher import BruteForceSearcher
from xib.training.evaluator import DecipherEvaluator, LMEvaluator
from xib.training.task import DecipherTask, LMTask, MlmTask, TransferTask
from xib.training.trainer import DecipherTrainer, LMTrainer

add_argument('task', default='lm', dtype=str, choices=['lm', 'decipher'], msg='which task to run')


class LMManager:

    def __init__(self):
        self.model = LM()
        if has_gpus():
            self.model.cuda()

        task = LMTask()
        self.dl_reg = DataLoaderRegistry()
        self.dl_reg.register_data_loader(task, g.data_path)
        self.evaluator = LMEvaluator(self.model, self.dl_reg[task])
        self.trainer = LMTrainer(self.model, [task], [1.0], 'total_step',
                                 evaluator=self.evaluator,
                                 check_interval=g.check_interval,
                                 eval_interval=g.eval_interval)

    def run(self):
        self.trainer.train(self.dl_reg)


# class AdaptManager(Manager):

#     data_loader_cls = DenseIpaDataLoader
#     trainer_cls = AdaptLMTrainer

#     def _get_model(self):
#         return AdaptedLM()


class DecipherManager:

    add_argument('dev_data_path', dtype='path', msg='Path to dev data.')
    add_argument('saved_path', dtype='path')
    add_argument('local_model_path', dtype='path', msg='Path to a saved local model, skipping the local training phase.')
    add_argument('use_mlm_loss', dtype=bool, default=False, msg='Flag to use mlm loss.')
    add_argument('mlm_model_path', dtype='path', msg='Path to a saved mlm model.')

    def __init__(self):
        self.model = DecipherModel()
        if has_gpus():
            self.model.cuda()

        train_task = DecipherTask('train')
        dev_task = DecipherTask('dev')
        self.dl_reg = DataLoaderRegistry()

        self.dl_reg.register_data_loader(train_task, g.data_path)
        self.dl_reg.register_data_loader(dev_task, g.dev_data_path)
        self.evaluator = None
        # self.evaluator = DecipherEvaluator(self.model, self.dl_reg, [train_task, dev_task])
        self.trainer = DecipherTrainer(self.model, [train_task], [1.0], 'total_step',
                                       evaluator=self.evaluator,
                                       check_interval=g.check_interval,
                                       eval_interval=g.eval_interval)

    def run(self):
        self.trainer.train(self.dl_reg)
