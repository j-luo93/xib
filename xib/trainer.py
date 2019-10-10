import logging

import torch
import torch.optim as optim

from arglib import add_argument, g, init_g_attr
from devlib import get_range
from trainlib import Metric, Metrics, Tracker
from xib.cfg import Category


@init_g_attr(default='property')
class Trainer:

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('save_interval', default=500, dtype=int, msg='save models after this many steps')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps, learning_rate, check_interval, save_interval, log_dir):
        self.tracker = Tracker()
        self.tracker.add_track('step', update_fn='add', finish_when=num_steps)
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate)

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
            distr = self.model(batch)
            metrics = self._analyze_output(distr, batch.target_feat, batch.target_weight)
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

    def _analyze_output(self, distr, target_feat, target_weight) -> Metrics:
        metrics = Metrics()
        total_loss = 0.0
        for i, value in enumerate(Category):
            target = target_feat[:, i]
            output = distr[value.name]
            log_probs = output.gather(1, target.view(-1, 1)).view(-1)

            loss = -(log_probs * target_weight).sum()
            total_loss += loss
            loss = Metric(f'loss_{value.name}', loss, target_weight.sum())
            metrics += loss
        metrics += Metric('loss', total_loss, target_weight.sum())
        return metrics
