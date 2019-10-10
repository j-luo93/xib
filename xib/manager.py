from data_loader import IpaDataLoader
from model import Model
from trainer import Trainer
import os


class Manager:

    def __init__(self):
        self.model = Model()
        if os.environ.get('CUDA_VISIBLE_DEVICES', False):
            self.model.cuda()
        self.train_data_loader = IpaDataLoader()
        self.trainer = Trainer(self.model, self.train_data_loader)

    def train(self):
        self.trainer.train()
