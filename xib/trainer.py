import torch.optim as optim

from arglib import add_argument, init_g_attr
from devlib import get_range
from trainlib import Metric, Metrics, Tracker


@init_g_attr
class Trainer:

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('save_interval', default=500, dtype=int, msg='save models after this many steps')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps, learning_rate, check_interval, save_interval):
        self.tracker = Tracker()
        self.tracker.add_track('step', update_fn='add', finish_when=num_steps)
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate)

    def train(self):
        while not self.tracker.is_finished:
            for batch in self.train_data_loader:
                self.model.train()
                self.optimizer.zero_grad()
                distr = self.model(batch)
                metrics = self._analyze_output(distr, batch.target_feat, batch.target_weight)
                metrics.loss.mean.backward()  # IDEA(j_luo) maybe clip gradient norm?
                self.optimizer.step()
                self.tracker.update()

                if self.tracker.step % self.check_interval == 0:
                    print(metrics.get_table(f'Step: {self.tracker.step}'))
                    metrics.clear()
                if self.tracker.step % self.save_interval == 0:
                    torch.save(self.model.state_dict(), self.log_dir / 'saved.latest')

    def _analyze_output(self, distr, target_feat, target_weight) -> Metrics:
        log_probs = distr.gather(1, target_feat)
        loss = -(log_probs * target_weight.view(-1, 1)).sum()
        loss = Metric('loss', loss, target_weight.sum())  # .numel())  # mask.sum())
        return Metrics(loss)
