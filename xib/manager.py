import os

from arglib import add_argument
from xib.data_loader import ContinuousTextDataLoader, IpaDataLoader
from xib.model import DecipherModel, Model
from xib.trainer import DecipherTrainer, LMTrainer

add_argument('task', default='lm', dtype=str, choices=['lm', 'decipher'], msg='which task to run')


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
