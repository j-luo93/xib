import logging

import torch
import torch.nn as nn
import torch.optim as optim

from arglib import add_argument, g, init_g_attr
from trainlib import Metric, Metrics, Tracker, Trainer
from xib.ipa import Category, should_include


@init_g_attr(default='property')
class BaseTrainer(Trainer):

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('save_interval', default=500, dtype=int, msg='save models after this many steps')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps, learning_rate, check_interval, save_interval, log_dir, mode):
        super().__init__()
        self.tracker.add_track('step', update_fn='add', finish_when=num_steps)
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate)

        for name, p in model.named_parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)

        # Prepare batch iterator.
        self.iterator = self._next_batch_iterator()

    def _next_batch_iterator(self):
        while True:
            yield from self.train_data_loader

    def check_metrics(self, accum_metrics: Metrics):
        if self.tracker.step % self.check_interval == 0:
            logging.info(accum_metrics.get_table(f'Step: {self.tracker.step}'))
            accum_metrics.clear()

    def save(self):
        if self.tracker.step % self.save_interval == 0:
            out_path = self.log_dir / 'saved.latest'
            to_save = {
                'model': self.model.state_dict(),
                'g': g.state_dict()
            }
            torch.save(to_save, out_path)
            logging.imp(f'Model saved to {out_path}.')


class LMTrainer(BaseTrainer):

    add_argument('mode', default='pcvdst', dtype=str,
                 msg='what to include during training: p(type), c(onstonant), v(vowel), d(iacritics), s(tress) and t(one).')

    def train_loop(self):
        self.model.train()
        self.optimizer.zero_grad()
        batch = next(self.iterator)
        scores = self.model.score(batch)
        metrics = self._analyze_scores(scores)
        metrics.loss.mean.backward()
        self.optimizer.step()
        return metrics

    def _analyze_scores(self, scores) -> Metrics:
        metrics = Metrics()
        total_loss = 0.0
        total_weight = 0.0
        for cat, (losses, weights) in scores.items():
            if should_include(self.mode, cat):
                loss = (losses * weights).sum()
                weight = weights.sum()
                total_loss += loss
                total_weight += weight
                loss = Metric(f'loss_{cat.name}', loss, weight)
                metrics += loss
        metrics += Metric('loss', total_loss, total_weight)
        return metrics


class DecipherTrainer(BaseTrainer):

    def train_loop(self):
        self.model.train()
        self.optimizer.zero_grad()
        batch = next(self.iterator)
        scores = self.model(batch)
        bs = batch.feat_matrix.size('batch')
        metrics = Metrics(*[
            Metric(name, score, bs) for name, score in scores.items()
        ])
        score = Metric('score', sum(metrics), bs)
        metrics += score
        loss = metrics.score.mean
        loss.backward()
        self.optimizer.step()
        return metrics
