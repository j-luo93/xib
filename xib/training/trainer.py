import logging
from abc import ABCMeta
from pathlib import Path
from typing import ClassVar

import torch
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_

from dev_misc import get_tensor
from dev_misc.arglib import add_argument, g
from dev_misc.trainlib import (Metric, Metrics, freeze, get_grad_norm,
                               get_trainable_params)
from dev_misc.trainlib.base_trainer import BaseTrainer as BaseTrainerDev
from dev_misc.utils import deprecated, global_property, pbar
from xib.data_loader import ContinuousTextDataLoader, IpaDataLoader
from xib.model.decipher_model import DecipherModel
from xib.model.extract_model import ExtractModel
from xib.model.lm_model import LM, AdaptLM
from xib.training.analyzer import (AdaptLMAnalyzer, DecipherAnalyzer,
                                   ExtractAnalyzer, LMAnalyzer)
from xib.training.optim import AdamInverseSqrtWithWarmup


class BaseTrainer(BaseTrainerDev, metaclass=ABCMeta):  # pylint: disable=abstract-method

    add_argument('num_steps', default=10, dtype=int, msg='number of steps to train')
    add_argument('learning_rate', default=2e-3, dtype=float, msg='learning rate')
    add_argument('check_interval', default=2, dtype=int, msg='check metrics after this many steps')
    add_argument('eval_interval', default=500, dtype=int, msg='save models after this many steps')

    def save_to(self, path: Path):
        to_save = {
            'model': self.model.state_dict(),
            'g': g.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(to_save, path)
        logging.imp(f'Model saved to {path}.')


class LMTrainer(BaseTrainer):

    add_argument('feat_groups', default='pcvdst', dtype=str,
                 msg='what to include during training: p(type), c(onstonant), v(vowel), d(iacritics), s(tress) and t(one).')

    model: LM
    analyzer_cls: ClassVar = LMAnalyzer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # IDEA(j_luo) Preparing the trainer should be handled by the manager, not by __init__ call.
        logging.warning('Init model.')
        for p in get_trainable_params(self.model, named=False):
            if p.ndim == 2:
                torch.nn.init.xavier_uniform_(p)

        # # DEBUG(j_luo) identity init
        # logging.warning('identity init')
        # for p in self.model.adapter.adapters.values():
        #     lp = len(p)
        #     p.data[range(lp), range(lp)] += 20.0

        # freeze(self.model.adapter)

        self.set_optimizer(optim.Adam, lr=g.learning_rate)
        self.analyzer = self.analyzer_cls()

    def add_trackables(self):
        self.tracker.add_trackable('total_step', total=g.num_steps)
        self.tracker.add_min_trackable('best_loss')

    def train_one_step(self, dl: IpaDataLoader) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        batch = dl.get_next_batch()
        ret = self.model.score(batch)
        # for idx, segment in enumerate(batch.segments):
        #     if str(segment).startswith('e-s-t-a-n'):
        #         break
        # from xib.ipa import Name
        # name = Name('Ptype', 'camel')
        # print(torch.stack([ret.distr[name][0], ret.distr_noise[name][0]], new_name='tmp')[idx])
        # import time; time.sleep(1)
        metrics = self.analyzer.analyze(ret)
        metrics.loss.mean.backward()
        grad_norm = get_grad_norm(self.model)
        grad_norm = Metric('grad_norm', grad_norm * len(batch), len(batch))
        metrics += grad_norm
        self.optimizer.step()
        return metrics

    def save(self, eval_metrics: Metrics):
        new_value = eval_metrics.loss.mean
        self.save_to(g.log_dir / 'saved.latest')
        if self.tracker.update('best_loss', value=new_value):
            out_path = g.log_dir / 'saved.best'
            logging.imp(f'Best model updated: new best is {self.tracker.best_loss:.3f}')
            self.save_to(out_path)


class AdaptLMTrainer(LMTrainer):

    model: AdaptLM
    analyzer_cls: ClassVar = AdaptLMAnalyzer


class DecipherTrainer(BaseTrainer):

    add_argument('score_per_word', default=1.0, dtype=float, msg='score added for each word')
    add_argument('concentration', default=1e-2, dtype=float, msg='concentration hyperparameter')
    add_argument('supervised', dtype=bool, default=False, msg='supervised mode')
    # add_argument('mode', default='local-supervised', dtype=str,
    #              choices=['local-supervised', 'global-supervised'], msg='training mode')
    add_argument('mlm_coeff', dtype=float, default=0.05, msg='Flag to use mlm loss.')
    add_argument('warmup_updates', dtype=int, default=4000, msg='Number of warmup updates for Adam.')

    model: DecipherModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = DecipherAnalyzer()
        self.set_optimizer()

    def set_optimizer(self):
        super().set_optimizer(AdamInverseSqrtWithWarmup,
                              lr=g.learning_rate, betas=(0.9, 0.98),
                              warmup_updates=g.warmup_updates)

    def add_trackables(self):
        self.tracker.add_trackable('total_step', total=g.num_steps)
        # self.tracker.add_min_trackable('best_loss')
        self.tracker.add_max_trackable('best_f1')

    def train_one_step(self, dl: ContinuousTextDataLoader) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        batch = dl.get_next_batch()
        ret = self.model(batch)
        metrics = self.analyzer.analyze(ret, batch)
        metrics.total_loss.mean.backward()
        self.optimizer.step()
        grad_norm = clip_grad_norm_(self.model.parameters(), 5.0)
        weight = (~batch.source_padding).sum()
        metrics += Metric('grad_norm', grad_norm * weight, weight)
        return metrics.with_prefix_('decipher')

    def load(self, path: Path, load_lm_model: bool = False, load_optimizer_state: bool = False, load_phi_scorer: bool = False):
        saved = torch.load(path)
        smsd = saved['model']
        if not load_lm_model:
            smsd = {k: v for k, v in smsd.items() if not k.startswith('lm_model')}
        if not load_phi_scorer:
            smsd = {k: v for k, v in smsd.items() if not k.startswith('phi_scorer')}
        self.model.load_state_dict(smsd, strict=False)
        if load_optimizer_state:
            self.optimizer.load_state_dict(saved['optimizer'])
        logging.imp(f'Loading model from {path}.')

    def save(self, eval_metrics: Metrics):
        self.save_to(g.log_dir / 'saved.latest')
        # self.tracker.update('best_loss', value=eval_metrics.dev_total_loss.mean)
        if self.tracker.update('best_f1', value=eval_metrics.dev_prf_f1.value):
            out_path = g.log_dir / f'saved.best'
            logging.imp(f'Best model updated: new best is {self.tracker.best_f1:.3f}')
            self.save_to(out_path)


add_argument('accum_gradients', default=1, dtype=int, msg='Accumulate this many steps of gradients.')


class ExtractTrainer(BaseTrainer):

    model: ExtractModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.analyzer = ExtractAnalyzer()
        # for p in self.model.parameters():
        #     if p.ndim == 2:
        #         torch.nn.init.xavier_uniform_(p)

        # # # DEBUG(j_luo) identity init
        # logging.warning('identity init')
        # for p in self.model.adapter.adapters.values():
        #     lp = len(p)
        #     p.data[range(lp), range(lp)] += 200.0

        self.ins_del_cost = g.init_ins_del_cost

    # DEBUG(j_luo) dilute
    def dilute(self):
        logging.imp('Diluting the weights.')
        for p in self.model.adapter.adapters.values():
            lp = len(p)
            p.data *= 0.5

    @global_property
    def ins_del_cost(self):
        pass

    @ins_del_cost.setter
    def ins_del_cost(self, value):
        pass

    @global_property
    def threshold(self):
        pass

    @threshold.setter
    def threshold(self, value):
        pass

    def add_trackables(self):
        self.tracker.add_trackable('round', endless=True)
        self.tracker.add_trackable('total_step', total=g.num_steps)
        self.tracker.add_max_trackable('best_f1')
        self.tracker.add_max_trackable('best_score')
        self.tracker.add_trackable('early_stop', total=3)

    def reset(self):
        """Reset the tracker. But keep the best_f1 since it's related to evaluation."""
        self.tracker.reset('total_step', 'best_score', 'early_stop')

    def load(self, path: Path):
        saved = torch.load(path)
        smsd = saved['model']
        self.model.load_state_dict(smsd)
        logging.imp(f'Loading model from {path}.')

    def save(self, eval_metrics: Metrics):
        r = self.tracker.round
        self.save_to(g.log_dir / f'saved.{self.stage}.latest')
        # self.tracker.update('best_loss', value=eval_metrics.dev_total_loss.mean)
        if self.tracker.update('best_f1', value=eval_metrics.prf_exact_span_f1.value):
            out_path = g.log_dir / f'saved.{self.stage}.best'
            logging.imp(f'Best model updated: new best is {self.tracker.best_f1:.3f}')
            self.save_to(out_path)

        if not self.tracker.update('best_score', value=eval_metrics.score.value, threshold=0.01):
            self.tracker.update('early_stop')
            # DEBUG(j_luo)
            # self.lr_scheduler.step()

    def should_terminate(self):
        return self.tracker.is_finished('early_stop')

    def train_one_step(self, dl: ContinuousTextDataLoader) -> Metrics:
        self.model.train()
        self.optimizer.zero_grad()
        accum_metrics = Metrics()
        if self.tracker.total_step == 50:
            breakpoint()  # DEBUG(j_luo)
        # self.ins_del_cost *= 0.9
        # self.ins_del_cost = max(self.ins_del_cost, g.min_ins_del_cost)
        # import time; time.sleep(0.2)
        # print(self.ins_del_cost)
        for _ in pbar(range(g.accum_gradients), desc='accum_gradients'):
            batch = dl.get_next_batch()
            ret = self.model(batch)
            metrics = self.analyzer.analyze(ret, batch)

            # # # DEBUG(j_luo)
            # sparsity = 0.0
            # for cat, dfm in ret.adapted_dfm.items():
            #     sparsity += ((dfm ** 2) * ((1.0 - dfm) ** 2)).sum()
            # sparsity = Metric('sparsity', sparsity, batch.batch_size)
            # metrics += sparsity

            # loss = -metrics.score.mean + sparsity.mean * 0.01

            loss = -metrics.score.mean
            (loss / g.accum_gradients).backward()

            # (-metrics.score.mean / g.accum_gradients).backward()
            accum_metrics += metrics

            # # DEBUG(j_luo)
            # self.threshold *= 0.99
            # self.threshold = max(self.min_threshold, self.threshold)
            # if self.tracker.total_step % 10 == 0:
            #     print(self.threshold)

        grad_norm = clip_grad_norm_(self.model.parameters(), 5.0)
        self.optimizer.step()
        # DEBUG(j_luo)
        # self.lr_scheduler.step()
        accum_metrics += Metric('grad_norm', grad_norm * batch.batch_size, batch.batch_size)

        return accum_metrics
