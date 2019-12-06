from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from itertools import product
from typing import Dict, List, Optional, Tuple

import torch

from dev_misc import BT, FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import BaseBatch, batch_class, get_length_mask
from dev_misc.devlib.named_tensor import NoName, get_named_range
from dev_misc.trainlib import Metric, Metrics, Tracker
from xib.data_loader import ContinuousIpaBatch
from xib.ipa.process import B, I, O


class BaseSearcher(ABC):

    def search(self, lengths: LT, label_log_probs: FT, gold_tag_seqs: Optional[LT] = None) -> Tuple[LT, FT]:
        samples, sample_log_probs = self.search_by_probs(lengths, label_log_probs)
        if gold_tag_seqs is not None:
            gold_tag_seqs = gold_tag_seqs.align_as(samples)

            max_length = lengths.max().item()
            with NoName(lengths):
                length_mask = get_length_mask(lengths, max_length).rename('batch', 'length')
            gold_log_probs = label_log_probs.gather('label', gold_tag_seqs)
            gold_log_probs = (gold_log_probs * length_mask.align_as(gold_log_probs)).sum('length')

            samples = torch.cat([gold_tag_seqs, samples], dim='sample')
            sample_log_probs = torch.cat([gold_log_probs, sample_log_probs], dim='sample')
        return samples, sample_log_probs

    @abstractmethod
    def search_by_probs(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]: ...


class BruteForceSearcher(BaseSearcher):

    def search_by_probs(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]:
        max_length = lengths.max().item()
        samples = get_tensor(torch.LongTensor(list(product([B, I, O], repeat=max_length))))
        samples.rename_('sample', 'length')
        bs = label_log_probs.size('batch')
        samples = samples.align_to('batch', 'sample', 'length').expand(bs, -1, -1)
        sample_log_probs = label_log_probs.gather('label', samples)
        with NoName(lengths):
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
    sample_log_probs: FT = field(init=False, default=None)

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

    def finish_search(self, lengths: LT):
        last_beam_id = get_zeros(lengths.size('batch'), g.beam_size).long().rename('batch', 'beam')
        start_beam_id = get_named_range(g.beam_size, 'beam').align_as(last_beam_id)
        samples = list()
        for i, (hyp, beam_id) in enumerate(zip(reversed(self.hyps), reversed(self.beam_ids))):
            step = len(self.beam_ids) - i
            start_backtrack = (step == lengths).align_as(beam_id)
            # new_last_beam_id = beam_id.gather('beam', last_beam_id)
            this_beam_id = torch.where(start_backtrack, start_beam_id, last_beam_id)
            samples.append(hyp.gather('beam', this_beam_id))
            last_beam_id = beam_id.gather('beam', this_beam_id)
        self.samples = torch.stack(samples[::-1], new_name='length')

        hyp_log_probs = torch.stack(self.hyp_log_probs, new_name='length')
        self.sample_log_probs = hyp_log_probs.gather('length', lengths.align_as(hyp_log_probs)).squeeze('length')


class BeamSearcher(BaseSearcher):

    add_argument('beam_size', default=200, dtype=int, msg='Size of beam.')

    def search_by_probs(self, lengths: LT, label_log_probs: FT) -> Tuple[LT, FT]:
        max_length = lengths.max().item()
        bs = label_log_probs.size('batch')
        label_log_probs = label_log_probs.align_to('length', 'batch', 'label')
        beam = Beam(bs)
        for step in range(max_length):
            __label_log_probs = label_log_probs[step]
            # __lengths = lengths[step]
            within_length = (step < lengths).align_as(__label_log_probs)  # __lengths
            beam.extend(__label_log_probs * within_length.float())
        beam.finish_search(lengths)
        samples = beam.samples.rename(beam='sample')
        sample_log_probs = beam.sample_log_probs.rename(beam='sample')
        return samples, sample_log_probs
