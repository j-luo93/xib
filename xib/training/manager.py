import logging
import os
import random

import torch

from dev_misc import g
from dev_misc.arglib import add_argument, init_g_attr
from dev_misc.trainlib import Metrics, set_random_seeds
from xib.data_loader import (ContinuousTextDataLoader, DenseIpaDataLoader,
                             IpaDataLoader)
from xib.model.decipher_model import DecipherModel
from xib.model.lm_model import LM, AdaptedLM
from xib.training.evaluator import DecipherEvaluator, LMEvaluator
from xib.training.trainer import AdaptLMTrainer, DecipherTrainer, LMTrainer

add_argument('task', default='lm', dtype=str, choices=['lm', 'decipher', 'adapt'], msg='which task to run')


class Manager:

    data_loader_cls = IpaDataLoader
    trainer_cls = LMTrainer

    def __init__(self):
        self.model = self._get_model()
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        self.train_data_loader = self.data_loader_cls(g.data_path)
        self.evaluator = LMEvaluator(self.model, self.train_data_loader)
        self.trainer = self.trainer_cls(self.model, self.train_data_loader, self.evaluator)

    def _get_model(self):
        return LM()

    def train(self):
        self.trainer.train()


class AdaptManager(Manager):

    data_loader_cls = DenseIpaDataLoader
    trainer_cls = AdaptLMTrainer

    def _get_model(self):
        return AdaptedLM()


class DecipherManager:

    add_argument('dev_data_path', dtype='path', msg='Path to dev data.')
    add_argument('saved_path', dtype='path')

    data_loader_cls = ContinuousTextDataLoader
    trainer_cls = DecipherTrainer

    def _get_model(self):
        return DecipherModel()

    def __init__(self):
        self.model = self._get_model()
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        if g.saved_path:
            self.model.load_state_dict(torch.load(g.saved_path)['model'])
        self.train_data_loader = ContinuousTextDataLoader(g.data_path)
        dev_data_loader = ContinuousTextDataLoader(g.dev_data_path)
        self.evaluator = DecipherEvaluator(self.model, dev_data_loader)
        self.trainer = self.trainer_cls(self.model, self.train_data_loader, self.evaluator)
