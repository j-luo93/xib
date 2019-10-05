from arglib import add_argument, init_g_attr

from .model import Model
from .trainer import Trainer


class Manager:

    def __init__(self):
        self.model = Model()
        self.train_data_loader = IpaDataLoader()
        self.trainer = Trainer(self.model, self.train_data_loader)

    def train(self):
        self.trainer.train()
