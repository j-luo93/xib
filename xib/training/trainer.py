import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.optim as optim

from dev_misc.arglib import add_argument, g, init_g_attr
from dev_misc.devlib import get_length_mask
from dev_misc.trainlib import (Metric, Metrics, Tracker, get_grad_norm,
                               get_trainable_params, log_this)
from dev_misc.trainlib.base_trainer import BaseTrainer as BaseTrainerDev
from xib.data_loader import (ContinuousTextDataLoader, ContinuousTextIpaBatch,
                             IpaDataLoader)
from xib.ipa import should_include
from xib.model.decipher_model import DecipherModel
from xib.model.lm_model import LM
from xib.training import evaluator
from xib.training.evaluator import DecipherEvaluator, LMEvaluator
from xib.training.optim import AdamInverseSqrtWithWarmup
from xib.training.runner import BaseDecipherRunner, BaseLMRunner


class BaseTrainer(BaseTrainerDev):

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('eval_interval', default=500, dtype=int, msg='save models after this many steps')
    # def __init__(self, model: Union[LM, DecipherModel], train_data_loader: Union[ContinuousTextDataLoader, IpaDataLoader]):
    #     super().__init__()
    #     self.model = model
    #     self.train_data_loader = train_data_loader
    #     self.tracker.add_trackable('step', total=g.num_steps)
    #     self.tracker.ready()
    #     self.optimizer = optim.Adam(get_trainable_params(self.model, named=False), g.learning_rate)

    #     self.init_params()

    #     # Prepare batch iterator.
    #     self.iterator = self._next_batch_iterator()

    # def init_params(self):
    #     self._init_params()

    # @log_this(log_level='IMP')
    # def _init_params(self, init_matrix=True, init_vector=False, init_higher_tensor=False):
    #     for name, p in get_trainable_params(self.model, named=True):
    #         if p.dim() == 2 and init_matrix:
    #             nn.init.xavier_uniform_(p)
    #         elif p.dim() == 1 and init_vector:
    #             nn.init.uniform_(p, 0.01)
    #         elif init_higher_tensor:
    #             nn.init.uniform_(p, 0.01)

    # def _next_batch_iterator(self):
    #     while True:
    #         yield from self.train_data_loader

    # # @property
    # # @abstractmethod
    # # def track(self):
    # #     pass

    # def check_metrics(self, accum_metrics: Metrics):
    #     if self.track % g.check_interval == 0:
    #         logging.info(accum_metrics.get_table(f'Step: {self.track}'))
    #         accum_metrics.clear()

    # def save(self):
    #     if self.track % g.save_interval == 0:
    #         out_path = g.log_dir / 'saved.latest'
    #         self._save(out_path)

    def _save(self, path: Path):
        to_save = {
            'model': self.model.state_dict(),
            'g': g.state_dict()
        }
        torch.save(to_save, path)
        logging.imp(f'Model saved to {path}.')


class LMTrainer(BaseTrainer, BaseLMRunner):

    add_argument('feat_groups', default='pcvdst', dtype=str,
                 msg='what to include during training: p(type), c(onstonant), v(vowel), d(iacritics), s(tress) and t(one).')

    # def __init__(self, model: LM, train_data_loader: IpaDataLoader, evaluator: LMEvaluator):
    # def __init__(self, model: LM, train_data_loader: IpaDataLoader, evaluator: LMEvaluator):
    #     BaseTrainer.__init__(self, model, train_data_loader)
    #     self.best_metrics: Metrics = None
    #     self.evaluator = evaluator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_optimizer(optim.Adam, lr=g.learning_rate)

    def add_trackables(self):
        self.tracker.add_trackable('total_step', total=g.num_steps)
        self.tracker.add_min_trackable('best_loss')

    def train_one_step(self, dl: IpaDataLoader) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        batch = dl.get_next_batch()
        scores = self.model.score(batch)
        metrics = self.analyze_scores(scores)
        metrics.loss.mean.backward()
        grad_norm = get_grad_norm(self.model)
        grad_norm = Metric('grad_norm', grad_norm * len(batch), len(batch))
        metrics += grad_norm
        self.optimizer.step()
        return metrics

    # def train(self, *args, **kwargs):
    #     accum_metrics = Metrics()
    #     while not self.tracker.is_finished('step'):
    #         metrics = self.train_loop(*args, **kwargs)
    #         accum_metrics += metrics
    #         self.tracker.update('step')

    #         self.check_metrics(accum_metrics)
    #         self.save()

    def save(self, eval_metrics: Metrics):
        # super().save()
        # if self.track % g.save_interval == 0:
        # metrics = self.evaluator.evaluate()
        # logging.info(f'New evaluation metrics is {getattr(metrics, name).mean:.3f}.')
        new_value = eval_metrics.loss.mean
        self._save(g.log_dir / 'saved.latest')
        if self.tracker.update('best_loss', value=new_value):
            out_path = g.log_dir / 'saved.best'
            logging.imp(f'Best model updated: new best is {self.tracker.best_loss:.3f}')
            self._save(out_path)


# class AdaptLMTrainer(LMTrainer):

#     def init_params(self):
#         pass


class DecipherTrainer(BaseTrainer, BaseDecipherRunner):

    add_argument('score_per_word', default=1.0, dtype=float, msg='score added for each word')
    add_argument('concentration', default=1e-2, dtype=float, msg='concentration hyperparameter')
    add_argument('supervised', dtype=bool, default=False, msg='supervised mode')
    add_argument('mode', default='local-supervised', dtype=str,
                 choices=['local-supervised', 'global-supervised'], msg='training mode')

    # def __init__(self, model: DecipherModel, train_data_loader: ContinuousTextDataLoader, evaluator: DecipherEvaluator):
    #     super().__init__(model, train_data_loader, g.num_steps, g.learning_rate,
    #                      g.check_interval, g.save_interval, g.log_dir, g.feat_groups)
    #     self.evaluator = evaluator

    def add_trackables(self):
        self.tracker.add_trackable('total_step', total=g.num_steps)
        self.tracker.add_min_trackable('best_loss')
        self.tracker.add_max_trackable('best_f1')

    def __init__(self, *args, mode: str = 'local', **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.set_optimizer()

    def set_optimizer(self):
        super().set_optimizer(AdamInverseSqrtWithWarmup,
                              lr=g.learning_rate, betas=(0.9, 0.98))

    # # DEBUG(j_luo)
    # def init_params(self):
    #     pass

    # def train(self, *args, **kwargs):
    #     accum_metrics = Metrics()
    #     while not self.tracker.is_finished('step'):
    #         metrics = self.train_loop(*args, **kwargs)
    #         accum_metrics += metrics
    #         self.tracker.update('step')

    #         self.check_metrics(accum_metrics)
    #         if self.tracker.step % g.save_interval == 0:
    #             if g.mode == 'local-supervised':
    #                 name = 'prf_local_f1'
    #             else:
    #                 name = 'prf_global_f1'
    #             eval_metrics = self.save(name=name)
    #             self.tracker.update('best_loss', value=eval_metrics.total_loss.mean)
    #             self.tracker.update('best_f1', value=getattr(eval_metrics, name).mean)
    #             logging.info(eval_metrics.get_table(title='dev'))

    def train_one_step(self, dl: ContinuousTextDataLoader) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        batch = dl.get_next_batch()
        metrics = self.get_metrics(batch)
        # modified_log_probs = ret['sample_log_probs'] * self.concentration + (~ret['is_unique']).float() * (-999.9)
        # sample_probs = modified_log_probs.log_softmax(dim='sample').exp()

        # if g.supervised:
        #     modified_seq_log_probs = ret['seq_scores'] + (~ret['is_unique']).float() * (-999.9)
        #     seq_log_probs = modified_seq_log_probs.log_softmax(dim='sample')
        #     target_log_probs = seq_log_probs.align_to('batch', 'sample', 'seq_feat')[:, 0]

        #     risk = (sample_probs * (1.0 - seq_log_probs.exp())).sum()
        #     seq_loss = -target_log_probs.sum()
        #     total_loss = risk + seq_loss
        #     risk = Metric('risk', risk, bs)
        #     seq_loss = Metric('seq_loss', seq_loss, bs)
        #     total_loss = Metric('total_loss', total_loss, bs)
        #     metrics = Metrics(seq_loss, risk, total_loss)
        #     loss = total_loss.mean
        # else:
        #     final_ret = ret['lm_score'] + ret['word_score'] * self.score_per_word
        #     score = (sample_probs * final_ret).sum()
        #     lm_score = Metric('lm_score', ret['lm_score'].sum(), bs)
        #     word_score = Metric('word_score', ret['word_score'].sum(), bs)
        #     score = Metric('score', score, bs)
        #     metrics = Metrics(score, lm_score, word_score)
        #     loss = -score.mean
        metrics.total_loss.mean.backward()
        self.optimizer.step()
        grad_norm = get_grad_norm(self.model)
        weight = (~batch.source_padding).sum()
        metrics += Metric('grad_norm', grad_norm * weight, weight)
        return metrics

    def load(self, path: Path):
        saved = torch.load(path)
        self.model.load_state_dict(saved['model'])
        logging.imp(f'Loading model from {path}.')

    def save(self, eval_metrics: Metrics):
        if self.mode == 'local':
            name = 'prf_local_f1'
        else:
            name = 'prf_global_f1'
        self._save(g.log_dir / 'saved.latest')
        self.tracker.update('best_loss', value=eval_metrics.total_loss.mean)
        if self.tracker.update('best_f1', value=eval_metrics[name].mean):
            out_path = g.log_dir / 'saved.best'
            logging.imp(f'Best model updated: new best is {self.tracker.best_f1:.3f}')
            self._save(out_path)
