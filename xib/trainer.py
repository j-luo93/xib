import logging

import torch
import torch.nn as nn
import torch.optim as optim
from arglib import add_argument, g, init_g_attr
from trainlib import Metric, Metrics, Tracker
from xib.ipa import Category


@init_g_attr(default='property')
class Trainer:

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('save_interval', default=500, dtype=int, msg='save models after this many steps')
    add_argument('mode', default='pcvdst', dtype=str,
                 msg='what to include during training: p(type), c(onstonant), v(vowel), d(iacritics), s(tress) and t(one).')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps, learning_rate, check_interval, save_interval, log_dir, mode):
        self.tracker = Tracker()
        self.tracker.add_track('step', update_fn='add', finish_when=num_steps)
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate)

        for name, p in model.named_parameters():
            if p.dim() == 2:
                nn.init.xavier_uniform_(p)

    def _next_batch_iterator(self):
        while True:
            yield from self.train_data_loader

    def train(self):
        iterator = self._next_batch_iterator()
        accum_metrics = Metrics()
        while not self.tracker.is_finished:
            batch = next(iterator)
            self.model.train()
            self.optimizer.zero_grad()
            scores = self.model.score(batch)
            metrics = self._analyze_scores(scores)
            accum_metrics += metrics
            metrics.loss.mean.backward()  # IDEA(j_luo) maybe clip gradient norm?
            self.optimizer.step()
            self.tracker.update()

            if self.tracker.step % self.check_interval == 0:
                logging.info(accum_metrics.get_table(f'Step: {self.tracker.step}'))
                accum_metrics.clear()
            if self.tracker.step % self.save_interval == 0:
                self._save()

    def _save(self):
        out_path = self.log_dir / 'saved.latest'
        to_save = {
            'model': self.model.state_dict(),
            'g': g.state_dict()
        }
        torch.save(to_save, out_path)
        logging.imp(f'Model saved to {out_path}.')

    def _analyze_scores(self, scores) -> Metrics:
        metrics = Metrics()
        total_loss = 0.0
        total_weight = 0.0
        for cat, (losses, weights) in scores.items():
            if self._should_include_this_loss(cat):
                loss = (losses * weights).sum()
                weight = weights.sum()
                total_loss += loss
                total_weight += weight
                loss = Metric(f'loss_{cat.name}', loss, weight)
                metrics += loss
        metrics += Metric('loss', total_loss, total_weight)
        return metrics

    def _should_include_this_loss(self, cat):
        name = cat.name
        if name == 'PTYPE' and 'p' in self.mode:
            return True
        if name.startswith('C_') and 'c' in self.mode:
            return True
        if name.startswith('V_') and 'v' in self.mode:
            return True
        if name.startswith('D_') and 'd' in self.mode:
            return True
        if name.startswith('S_') and 's' in self.mode:
            return True
        if name.startswith('T_') and 't' in self.mode:
            return True
        return False
