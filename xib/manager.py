import os

from arglib import add_argument
from data_loader import IpaDataLoader, ContinuousTextDataLoader
from model import Model, DecipherModel
from trainer import LMTrainer, DecipherTrainer

add_argument('task', default='lm', dtype=str, choices=['lm', 'decipher'], msg='which task to run')


class Manager:

    model_cls = Model
    data_loader_cls = IpaDataLoader
    trainer_cls = LMTrainer

    def __init__(self):
        self.model = self.model_cls()
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        self.train_data_loader = self.data_loader_cls()
        self.trainer = self.trainer_cls(self.model, self.train_data_loader)

    def train(self):
        self.trainer.train()


class DecipherManager(Manager):

    model_cls = DecipherModel
    data_loader_cls = ContinuousTextDataLoader
    trainer_cls = DecipherTrainer
