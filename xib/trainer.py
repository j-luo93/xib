import torch.optim as optim

from arglib import add_argument, init_g_attr
from devlib import get_range
from trainlib import Metric, Metrics, Tracker


@init_g_attr
class Trainer:

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')

    def __init__(self, model: 'a', train_data_loader: 'a', num_steps):
        self.tracker = Tracker()
        self.tracker.add_track('global_step', update_fn='add', finish_when=num_steps)
        self.optimizer = optim.Adam(self.model.parameters(), learning_rate)

    def train(self):
        while not self.tracker.is_finished:
            for batch in self.train_data_loader:
                self.model.train()
                self.optimizer.zero_grad()
                distr = self.model(batch)
                metrics = self._analyze_output(distr, batch.target_ipa)
                metrics.loss.mean.backwards()  # TODO(j_luo) maybe clip gradient norm?
                self.tracker.update()

    def _analyze_output(self, distr, target_ipa) -> Metrics:
        bs, ws = target_ipa.shape
        batch_i = get_range(bs, 2, 0)
        window_i = get_range(ws, 2, 1)
        # FIXME(j_luo) This is actually wrong because for each ipa sound, you have multiple features.
        log_probs = target_ipa[batch_i, window_i]
        loss = -log_probs.sum()
        loss = Metric('loss', loss, bs)
        return Metrics(loss)
