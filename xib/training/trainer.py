import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from arglib import add_argument, g, init_g_attr
from devlib import get_trainable_params
from trainlib import Metric, Metrics, Tracker, Trainer, get_grad_norm, log_this
from xib.data_loader import MetricLearningDataLoader
from xib.ipa import should_include
from xib.training import evaluator

from .runner import BaseLMRunner


@init_g_attr(default='property')
class BaseTrainer(Trainer, metaclass=ABCMeta):

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('save_interval', default=500, dtype=int, msg='save models after this many steps')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps, learning_rate, check_interval, save_interval, log_dir, feat_groups):
        super().__init__()
        self.tracker.add_track('step', update_fn='add', finish_when=num_steps)
        self.optimizer = optim.Adam(get_trainable_params(self.model, named=False), learning_rate)

        self.init_params()

        # Prepare batch iterator.
        self.iterator = self._next_batch_iterator()

    def init_params(self):
        self._init_params()

    @log_this(log_level='IMP')
    def _init_params(self, init_matrix=True, init_vector=False, init_higher_tensor=False):
        for name, p in get_trainable_params(self.model, named=True):
            if p.dim() == 2 and init_matrix:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and init_vector:
                nn.init.uniform_(p, 0.01)
            elif init_higher_tensor:
                nn.init.uniform_(p, 0.01)

    def _next_batch_iterator(self):
        while True:
            yield from self.train_data_loader

    @property
    @abstractmethod
    def track(self):
        pass

    def check_metrics(self, accum_metrics: Metrics):
        if self.track % self.check_interval == 0:
            logging.info(accum_metrics.get_table(f'Step: {self.track}'))
            accum_metrics.clear()

    def save(self):
        if self.track % self.save_interval == 0:
            out_path = self.log_dir / 'saved.latest'
            self._save(out_path)

    def _save(self, path: Path):
        to_save = {
            'model': self.model.state_dict(),
            'g': g.state_dict()
        }
        torch.save(to_save, path)
        logging.imp(f'Model saved to {path}.')


@init_g_attr
class LMTrainer(BaseLMRunner, BaseTrainer):

    add_argument('feat_groups', default='pcvdst', dtype=str,
                 msg='what to include during training: p(type), c(onstonant), v(vowel), d(iacritics), s(tress) and t(one).')

    def __init__(self, model: 'a', train_data_loader: 'a', evaluator: 'a', num_steps, learning_rate, check_interval, save_interval, log_dir, feat_groups):
        BaseTrainer.__init__(self, model, train_data_loader)
        self.best_metrics: Metrics = None

    def train_loop(self) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        batch = next(self.iterator)
        scores = self.model.score(batch)
        metrics = self.analyze_scores(scores)
        metrics.loss.mean.backward()
        grad_norm = get_grad_norm(self.model)
        grad_norm = Metric('grad_norm', grad_norm * len(batch), len(batch))
        metrics += grad_norm
        self.optimizer.step()
        return metrics

    def save(self):
        super().save()
        if self.track % self.save_interval == 0:
            metrics = self.evaluator.evaluate()
            logging.info(f'New evaluation metrics is {metrics.loss.mean:.3f}.')
            if self.best_metrics is None or metrics.loss.mean < self.best_metrics.loss.mean:
                self.best_metrics = metrics
                out_path = self.log_dir / 'saved.best'
                logging.imp(f'Best model updated: new best is {self.best_metrics.loss.mean:.3f}.')
                self._save(out_path)


@init_g_attr
class AdaptLMTrainer(LMTrainer):

    def init_params(self):
        pass


@init_g_attr
class DecipherTrainer(LMTrainer):

    add_argument('score_per_word', default=1.0, dtype=float, msg='score added for each word')
    add_argument('concentration', default=1e-2, dtype=float, msg='concentration hyperparameter')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps, learning_rate, check_interval, save_interval, log_dir, feat_groups, score_per_word: 'p', concentration: 'p'):
        super().__init__(model, train_data_loader, num_steps, learning_rate, check_interval, save_interval, log_dir, feat_groups)

    def train_loop(self) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        batch = next(self.iterator)
        ret = self.model(batch)
        bs = batch.feat_matrix.size('batch')
        breakpoint()  # DEBUG(j_luo)
        modified_log_probs = ret['sample_log_probs'] * self.concentration + (~ret['is_unique']).float() * (-999.9)
        sample_probs = modified_log_probs.log_softmax(dim='sample').exp()
        final_ret = ret['lm_score'] + ret['word_score'] * self.score_per_word
        score = (sample_probs * final_ret).sum()
        lm_score = Metric('lm_score', ret['lm_score'].sum(), bs)
        word_score = Metric('word_score', ret['word_score'].sum(), bs)
        score = Metric('score', score, bs)
        metrics = Metrics(score, lm_score, word_score)
        loss = -score.mean
        loss.backward()
        self.optimizer.step()
        return metrics

# ------------------------------------------------------------- #
#                         Metric learner                        #
# ------------------------------------------------------------- #


@init_g_attr(default='property')
class MetricLearningTrainer(BaseTrainer):

    add_argument('num_epochs', default=5, dtype=int, msg='number of epochs')

    def __init__(self, model: 'a', data_loader: 'a', num_epochs, learning_rate, check_interval, save_interval, log_dir):
        Trainer.__init__(self)
        self.tracker.add_track('epoch', update_fn='add', finish_when=num_epochs)

    def train(self,
              evaluator: evaluator.Evaluator,
              train_langs: List[str],
              dev_langs: List[str],
              fold_idx: int) -> Metrics:
        # Reset parameters.
        self._init_params(init_matrix=True, init_vector=True, init_higher_tensor=True)
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        # Main boy.
        accum_metrics = Metrics()
        best_mse = None
        while not self.tracker.is_finished:
            # Get data first.
            metrics = self.train_loop(train_langs)
            accum_metrics += metrics
            self.tracker.update()

            self.check_metrics(accum_metrics)

            if self.track % self.save_interval == 0:
                self.save(dev_langs, f'{fold_idx}.latest')
                dev_metrics = evaluator.evaluate(dev_langs)
                logging.info(dev_metrics.get_table(title='dev'))
                if best_mse is None or dev_metrics.mse.mean < best_mse:
                    best_mse = dev_metrics.mse.mean
                    logging.imp(f'Updated best mse score: {best_mse:.3f}')
                    self.save(dev_langs, f'{fold_idx}.best')
        return Metric('best_mse', best_mse, 1)

    def save(self, dev_langs: List[str], suffix: str):
        out_path = self.log_dir / f'saved.{suffix}'
        to_save = {
            'model': self.model.state_dict(),
            'g': g.state_dict(),
            'dev_langs': dev_langs,
        }
        torch.save(to_save, out_path)
        logging.imp(f'Model saved to {out_path}.')

    def reset(self):
        # HACK(j_luo) Need to improve trakcer api.
        self.tracker._attrs['epoch'] = 0

    def train_loop(self, train_langs: List[str]) -> Metrics:
        fold_data_loader = self.data_loader.select(train_langs, train_langs)
        metrics = Metrics()
        for batch_i, batch in enumerate(fold_data_loader):
            self.model.train()
            self.optimizer.zero_grad()

            output = self.model(batch)
            mse = (output - batch.dist) ** 2
            mse = Metric('mse', mse.sum(), len(batch))
            metrics += mse

            mse.mean.backward()
            self.optimizer.step()
        return metrics

    @property
    def track(self):
        return self.tracker.epoch
