from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, g, get_zeros
from dev_misc.devlib import BaseBatch, batch_class, get_range
from dev_misc.devlib.named_tensor import NoName, get_named_range
from dev_misc.utils import WithholdKeys
from xib.aligned_corpus.corpus import AlignedSentence


@batch_class
class ViableSpans(BaseBatch):
    viable: BT
    batch_indices: LT
    starts: LT
    ends: LT
    lengths: LT
    unit_id_seqs: LT
    p_weights: FT
    start_candidates: LT
    end_candidates: LT
    len_candidates: LT


CandidateTuple = Tuple[LT, LT, Optional[FT]]


class BaseSpanProposer(nn.Module, ABC):

    @abstractmethod
    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple: ...

    def forward(self, lost_unit_id_seqs: LT, lost_lengths: LT, sentences: Sequence[AlignedSentence]) -> ViableSpans:
        max_length = lost_lengths.max().item()
        batch_size = lost_lengths.size('batch')

        start_candidates, len_candidates, p_weights = self._propose_candidates(
            lost_unit_id_seqs, lost_lengths, sentences)
        # This is inclusive.
        end_candidates = start_candidates + len_candidates - 1

        # Only keep the viable/valid spans around.
        viable = (end_candidates < lost_lengths.align_as(end_candidates))
        start_candidates = start_candidates.expand_as(viable)
        end_candidates = end_candidates.expand_as(viable)
        len_candidates = len_candidates.expand_as(viable)
        p_weights = torch.zeros_like(viable).fill_(1.0) if p_weights is None else p_weights
        batch_indices = get_named_range(batch_size, 'batch').expand_as(viable)
        with NoName(start_candidates, end_candidates, len_candidates,
                    batch_indices, viable, p_weights):
            viable_starts = start_candidates[viable].rename('viable')
            viable_ends = end_candidates[viable].rename('viable')
            viable_lengths = len_candidates[viable].rename('viable')
            viable_batch_indices = batch_indices[viable].rename('viable')
            viable_p_weights = p_weights[viable].rename('viable')

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
                           viable_p_weights,
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
        return start_candidates, len_candidates, None


class OracleStemSpanProposer(BaseSpanProposer):

    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple:
        assert g.use_stem
        starts = list()
        lens = list()
        for i, sentence in enumerate(sentences):
            uss = sentence.to_unsegmented(is_lost_ipa=g.input_format == 'ipa', is_known_ipa=True, annotated=True)
            if uss.segments:
                starts.append(uss.segments[0].start)
                lens.append(uss.segments[0].end - starts[-1] + 1)
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
        return start_candidates, len_candidates, None


class AllSpanProposer(BaseSpanProposer):

    def __init__(self):
        super().__init__()
        self.p_weights = dict()

    def cuda(self):
        super().cuda()
        for k, v in self.p_weights.items():
            self.p_weights[k] = v.cuda()

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd['p_weights'] = self.p_weights
        return sd

    def load_state_dict(self, state_dict, *args, **kwargs):
        # TODO(j_luo) Phase out this kwarg.
        kwargs['strict'] = False
        with WithholdKeys(state_dict, 'p_weights'):
            super().load_state_dict(state_dict, *args, **kwargs)
        for sk, sv in state_dict['p_weights'].items():
            if sk in self.p_weights:
                self.p_weights[sk].copy_(sv)
            else:
                self.p_weights[sk] = sv

    def _get_p_weights(self, sentences: Sequence[AlignedSentence]) -> FT:
        p_weights = list()
        for sentence in sentences:
            s_key = str(sentence)
            if s_key not in self.p_weights:
                uniform_weight = get_zeros(sentence.length, g.max_word_length + 1 - g.min_word_length)
                uniform_weight = uniform_weight.fill_(1.0)
                self.p_weights[s_key] = uniform_weight
            p_weights.append(self.p_weights[s_key])
        p_weights = torch.nn.utils.rnn.pad_sequence(p_weights, batch_first=True)
        return p_weights

    def _propose_candidates(self, lost_unit_id_seqs: LT, lost_lengths: LT,
                            sentences: Sequence[AlignedSentence]) -> CandidateTuple:
        max_length = lost_lengths.max().item()
        # Propose all span start/end positions.
        start_candidates = get_named_range(max_length, 'len_s').align_to('batch', 'len_s', 'len_e')
        # Range from `min_word_length` to `max_word_length`.
        len_candidates = get_named_range(g.max_word_length + 1 - g.min_word_length, 'len_e') + g.min_word_length
        len_candidates = len_candidates.align_to('batch', 'len_s', 'len_e')
        p_weights = self._get_p_weights(sentences)
        p_weights = torch.nn.utils.rnn.pad_sequence(p_weights, batch_first=True)
        p_weights.rename_('batch', 'len_s', 'len_e')
        return start_candidates, len_candidates, p_weights
