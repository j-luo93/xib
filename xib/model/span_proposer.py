import logging
import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_zeros
from dev_misc.devlib import BaseBatch, batch_class, get_range
from dev_misc.devlib.named_tensor import NoName, get_named_range
from dev_misc.utils import WithholdKeys, pbar
from xib.aligned_corpus.corpus import AlignedSentence
from xib.aligned_corpus.data_loader import AlignedDataLoader


@batch_class
class ViableSpans(BaseBatch):
    viable: BT
    batch_indices: LT
    starts: LT
    ends: LT
    lengths: LT
    unit_id_seqs: LT
    start_candidates: LT
    end_candidates: LT
    len_candidates: LT


CandidateTuple = Tuple[LT, LT]


class BaseSpanProposer(nn.Module, ABC):

    @abstractmethod
    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple: ...

    def forward(self, lost_unit_id_seqs: LT, lost_lengths: LT, sentences: Sequence[AlignedSentence]) -> ViableSpans:
        max_length = lost_lengths.max().item()
        batch_size = lost_lengths.size('batch')

        start_candidates, len_candidates = self._propose_candidates(
            lost_unit_id_seqs, lost_lengths, sentences)
        # This is inclusive.
        end_candidates = start_candidates + len_candidates - 1

        # Only keep the viable/valid spans around.
        viable = (end_candidates < lost_lengths.align_as(end_candidates))
        start_candidates = start_candidates.expand_as(viable)
        end_candidates = end_candidates.expand_as(viable)
        len_candidates = len_candidates.expand_as(viable)
        batch_indices = get_named_range(batch_size, 'batch').expand_as(viable)
        with NoName(start_candidates, end_candidates, len_candidates,
                    batch_indices, viable):
            viable_starts = start_candidates[viable].rename('viable')
            viable_ends = end_candidates[viable].rename('viable')
            viable_lengths = len_candidates[viable].rename('viable')
            viable_batch_indices = batch_indices[viable].rename('viable')

        # Get the word positions to get the corresponding representations.
        viable_starts_2d = viable_starts.align_to('viable', 'len_w')
        word_pos_offsets = get_named_range(g.max_word_length, 'len_w').align_as(viable_starts_2d)
        word_pos = viable_starts_2d + word_pos_offsets
        word_pos = word_pos.clamp(max=max_length - 1)

        # Get the corresponding representations.
        viable_batch_indices_2d = viable_batch_indices.expand_as(word_pos)
        with NoName(viable_batch_indices_2d, word_pos, lost_unit_id_seqs):
            viable_unit_id_seqs = lost_unit_id_seqs[viable_batch_indices_2d, word_pos]
            viable_unit_id_seqs.rename_('viable', 'len_w')

        return ViableSpans(viable,
                           viable_batch_indices,
                           viable_starts,
                           viable_ends,
                           viable_lengths,
                           viable_unit_id_seqs,
                           start_candidates,
                           end_candidates,
                           len_candidates)


class OracleWordSpanProposer(BaseSpanProposer):

    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple:
        # Start from the first position.
        start_candidates = torch.zeros_like(lost_lengths).align_to('batch', 'len_s', 'len_e')
        # Use full word length.
        len_candidates = lost_lengths.align_to('batch', 'len_s', 'len_e')
        len_candidates.clamp_(min=g.min_word_length, max=g.max_word_length)
        return start_candidates, len_candidates


class OracleStemSpanProposer(BaseSpanProposer):

    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple:
        assert g.use_stem
        starts = list()
        lens = list()
        for i, sentence in enumerate(sentences):
            uss = sentence.to_unsegmented(is_lost_ipa=g.input_format == 'ipa', is_known_ipa=True, annotated=True)
            if uss.segments:
                starts.append(uss.segments[0].single_segments[0].start)
                lens.append(uss.segments[0].single_segments[0].end - starts[-1] + 1)
            else:
                starts.append(0)
                lens.append(lost_lengths[i])
        start_candidates = torch.zeros_like(lost_lengths)
        start_candidates[:] = torch.LongTensor(starts)
        start_candidates = start_candidates.align_to('batch', 'len_s', 'len_e')
        len_candidates = torch.zeros_like(lost_lengths)
        len_candidates[:] = torch.LongTensor(lens)
        len_candidates = len_candidates.align_to('batch', 'len_s', 'len_e')
        len_candidates.clamp_(min=g.min_word_length, max=g.max_word_length)
        return start_candidates, len_candidates


class AllSpanProposer(BaseSpanProposer):

    add_argument('momentum', dtype=float, default=0.5)

    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple:
        max_length = lost_lengths.max().item()
        # Propose all span start/end positions.
        start_candidates = get_named_range(max_length, 'len_s').align_to('batch', 'len_s', 'len_e')
        # Range from `min_word_length` to `max_word_length`.
        len_candidates = get_named_range(g.max_word_length + 1 - g.min_word_length, 'len_e') + g.min_word_length
        len_candidates = len_candidates.align_to('batch', 'len_s', 'len_e')
        return start_candidates, len_candidates


class SpanBias(nn.Module):

    add_argument('non_span_bias', dtype=float, default=0.5)
    add_argument('bias_mode', dtype=str, default='none', choices=['none', 'fixed', 'learned'])

    def __init__(self):
        super().__init__()
        self.span_lengths = list(range(g.min_word_length, g.max_word_length + 1))
        if g.bias_mode == 'learned':
            self.span_prior = nn.ParameterDict({
                str(l): nn.Parameter(torch.tensor(0).float())  # pylint: disable=not-callable
                for l in self.span_lengths + [0]
            })

    def forward(self, is_span: bool):
        if g.bias_mode == 'none':
            if is_span:
                return {l: 0.0 for l in self.span_lengths}
            return 0.0
        elif g.bias_mode == 'fixed':
            if is_span:
                return {l: math.log((1.0 - g.non_span_bias) / len(self.span_lengths)) for l in self.span_lengths}
            return math.log(g.non_span_bias)
        else:
            with NoName(*self.span_prior.values()):
                prior = torch.stack(list(self.span_prior.values()), new_name='stacked').log_softmax(dim='stacked')
                if is_span:
                    return {l: prior[l - g.min_word_length] for l in self.span_lengths}
                return prior[-1]
