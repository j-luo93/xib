import logging
import os
import random

import torch

from dev_misc import g
from dev_misc.arglib import add_argument, init_g_attr
from dev_misc.trainlib import Metrics, has_gpus, set_random_seeds
from dev_misc.trainlib.trainer import freeze
from xib.data_loader import (ContinuousTextDataLoader, DataLoaderRegistry,
                             DenseIpaDataLoader, IpaDataLoader)
from xib.model.decipher_model import DecipherModel, TransferModel
from xib.model.lm_model import LM, AdaptedLM
from xib.training.evaluator import DecipherEvaluator, LMEvaluator
from xib.training.task import DecipherTask, LMTask, TransferTask
from xib.training.trainer import DecipherTrainer, LMTrainer, TransferTrainer

add_argument('task', default='lm', dtype=str, choices=['lm', 'decipher', 'adapt', 'transfer'], msg='which task to run')


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

    # def _get_model(self):
    #     return DecipherModel()

    # def __init__(self):
    #     self.model = self._get_model()
    #     if os.environ.get('CUDA_VISIBLE_DEVICES', False):
    #         self.model.cuda()
    #     if g.saved_path:
    #         self.model.load_state_dict(torch.load(g.saved_path)['model'])
    #     self.train_data_loader = ContinuousTextDataLoader(g.data_path)
    #     dev_data_loader = ContinuousTextDataLoader(g.dev_data_path)
    #     self.evaluator = DecipherEvaluator(self.model, dev_data_loader)
    #     self.trainer = self.trainer_cls(self.model, self.train_data_loader, self.evaluator)

    def __init__(self):
        self.model = DecipherModel()
        if has_gpus():
            self.model.cuda()

        train_task = DecipherTask('train')
        dev_task = DecipherTask('dev')

        self.dl_reg = DataLoaderRegistry()
        self.dl_reg.register_data_loader(train_task, g.data_path)
        self.dl_reg.register_data_loader(dev_task, g.dev_data_path)
        self.evaluator = DecipherEvaluator(self.model, self.dl_reg, [train_task, dev_task])
        self.trainer = DecipherTrainer(self.model, [train_task], [1.0], 'total_step',
                                       evaluator=self.evaluator,
                                       check_interval=g.check_interval,
                                       eval_interval=g.eval_interval)

    def run(self):
        if g.local_model_path:
            self.trainer.load(g.local_model_path, load_lm_model=False,
                              load_seq_scorer=False, load_optimizer_state=False)
        else:
            logging.info('Running on local mode.')
            self.evaluator.mode = 'local'
            self.trainer.mode = 'local'
            self.trainer.train(self.dl_reg)

        logging.info('Running on global mode.')
        # HACK(j_luo)
        self.trainer.mode = 'global'
        self.evaluator.mode = 'global'
        self.trainer.tracker.reset_all()
        if not g.local_model_path:
            self.trainer.load(g.log_dir / 'saved.local.best')
        # freeze(self.model.self_attn_layers)
        # freeze(self.model.positional_embedding)
        # freeze(self.model.label_predictor)
        # freeze(self.model.emb_for_label)
        # DEBUG(j_luo)
        # self.model.seq_scorer[0].weight.data.copy_(torch.FloatTensor([[1, 0, 0]]))

        self.trainer.set_optimizer()
        self.trainer.train(self.dl_reg)


class TransferManager:

    add_argument('global_model_path', dtype='path', msg='Path to the saved global model.')

    def __init__(self):
        self.model = TransferModel()
        # freeze(self.model.self_attn_layers)
        # freeze(self.model.positional_embedding)
        # freeze(self.model.label_predictor)
        # freeze(self.model.emb_for_label)
        # freeze(self.model.adapter)
        # freeze(self.model.seq_scorer)
        if has_gpus():
            self.model.cuda()

        dev_task = TransferTask('dev')

        self.dl_reg = DataLoaderRegistry()
        self.dl_reg.register_data_loader(dev_task, g.dev_data_path)

        self.evaluator = DecipherEvaluator(self.model, self.dl_reg, [dev_task])
        self.evaluator.mode = 'global'  # HACK(j_luo)
        self.trainer = TransferTrainer(self.model, [dev_task], [1.0], 'total_step',
                                       evaluator=self.evaluator,
                                       check_interval=g.check_interval,
                                       eval_interval=g.eval_interval)
        self.trainer.mode = 'global'  # HACK(j_luo)

    def run(self):
        self.trainer.load(g.global_model_path)
        # DEBUG(j_luo)
        # self.model.seq_scorer[0].weight.data.copy_(torch.FloatTensor([[1.0, 0.2, 2.5]]))

        self.trainer.train(self.dl_reg)
