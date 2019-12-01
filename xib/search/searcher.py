from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from itertools import product
from typing import Dict, List, Tuple

import torch

from dev_misc import FT, LT, BT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import BaseBatch, batch_class, get_length_mask
from dev_misc.devlib.named_tensor import NoName
from dev_misc.trainlib import Metric, Metrics, Tracker
from xib.data_loader import ContinuousTextDataLoader, ContinuousTextIpaBatch
from xib.ipa.process import B, I, O


class BaseSearcher(ABC):

    @abstractmethod
    def search(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]: ...


class BruteForceSearcher(BaseSearcher):

    def search(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]:
        max_length = lengths.max().item()
        samples = get_tensor(torch.LongTensor(list(product([B, I, O], repeat=max_length))))
        samples.rename_('sample', 'length')
        bs = label_log_probs.size('batch')
        samples = samples.align_to('batch', 'sample', 'length').expand(bs, -1, -1)
        sample_log_probs = label_log_probs.gather('label', samples)
        length_mask = get_length_mask(lengths, max_length).rename('batch', 'length')
        length_mask = length_mask.align_to(sample_log_probs)
        sample_log_probs = (sample_log_probs * length_mask.float()).sum(dim='length')
        return samples, sample_log_probs


@batch_class
class Beam(BaseBatch):
    batch_size: int
    hyps: List[LT] = field(init=False, default=None)
    hyp_log_probs: List[FT] = field(init=False, default=None)
    beam_ids: List[LT] = field(init=False, default=None)
    samples: LT = field(init=False, default=None)

    def __post_init__(self):
        if self.hyps is None:
            self.hyps = [get_zeros(self.batch_size, g.beam_size).long().rename('batch', 'beam')]
            log_probs = get_zeros(self.batch_size, g.beam_size).rename('batch', 'beam').fill_(-999.9)
            log_probs[:, 0] = 0.0
            self.hyp_log_probs = [log_probs]
            self.beam_ids = list()

    def extend(self, label_log_probs: FT):
        num_labels = label_log_probs.size('label')
        label_log_probs = label_log_probs.align_to('batch', 'beam', 'label')
        new_hyp_log_probs = self.hyp_log_probs[-1].align_to('batch', 'beam', 'label') + label_log_probs
        new_hyp_log_probs = new_hyp_log_probs.flatten(['beam', 'label'], 'beam_X_label')
        top_values, top_inds = torch.topk(new_hyp_log_probs, g.beam_size, 'beam_X_label')
        beam_ids = top_inds // num_labels
        label_ids = top_inds % num_labels
        self.beam_ids.append(beam_ids.rename(beam_X_label='beam'))
        self.hyps.append(label_ids.rename(beam_X_label='beam'))
        self.hyp_log_probs.append(top_values.rename(beam_X_label='beam'))

    def finish_search(self):
        last_beam_id = self.beam_ids[-1]
        samples = [self.hyps[-1]]
        for hyp, beam_id in zip(reversed(self.hyps[:-1]), reversed(self.beam_ids[:-1])):
            samples.append(hyp.gather('beam', last_beam_id))
            last_beam_id = beam_id.gather('beam', last_beam_id)
        self.samples = torch.stack(samples[::-1], new_name='length')


class BeamSearcher(BaseSearcher):

    add_argument('beam_size', default=200, dtype=int, msg='Size of beam.')

    def search(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]:
        max_length = lengths.max().item()
        bs = label_log_probs.size('batch')
        label_log_probs = label_log_probs.align_to('length', 'batch', 'label')
        beam = Beam(bs)
        for step in range(max_length):
            __label_log_probs = label_log_probs[step]
            __lengths = lengths[step]
            within_length = step < __lengths
            beam.extend(__label_log_probs * within_length.float())
        beam.finish_search()
        samples = beam.samples.rename(beam='sample')
        sample_log_probs = beam.hyp_log_probs[-1].rename(beam='sample')
        return samples, sample_log_probs
