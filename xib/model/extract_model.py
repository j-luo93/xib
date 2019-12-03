from typing import Optional

import numpy as np
import torch
import torch.nn as nn
# from pytorch_memlab import profile

from dev_misc import BT, FT, LT, add_argument, g, get_zeros
from dev_misc.devlib import BaseBatch, batch_class, get_length_mask, get_range
from dev_misc.devlib.named_tensor import (NameHelper, NoName, Rename,
                                          get_named_range)
from xib.data_loader import ContinuousTextIpaBatch, convert_to_dense
from xib.ipa.process import Segment
from xib.model.modules import AdaptLayer, FeatEmbedding

from .modules import DenseFeatEmbedding

# DEBUG(j_luo)
if 'profile' not in locals():
    profile = lambda x: x


@batch_class
class Matches(BaseBatch):
    end: LT  # Inclusive
    score: FT
    value: FT
    matched: BT


@batch_class
class Extracted(BaseBatch):
    batch_size: int
    matches: Optional[Matches] = None
    # last_end: Optional[LT] = None  # The end positions (inclusive) of the last extracted words.
    # score: Optional[FT] = None

    # def __post_init__(self):
    #     if self.score is None:
    #         # NOTE(j_luo) Mind the -1.
    #         # self.last_end = get_zeros(self.batch_size, g.max_extracted_candidates).long().rename('batch', 'cand') - 1
    #         self.score = get_zeros(self.batch_size, g.max_extracted_candidates).rename('batch', 'cand')


@batch_class
class ExtractModelReturn(BaseBatch):
    score: FT
    start: LT
    end: LT
    matched: BT


class ExtractModel(nn.Module):

    add_argument('max_num_words', default=3, dtype=int, msg='Max number of extracted words.')
    add_argument('max_word_length', default=10, dtype=int, msg='Max length of extracted words.')
    add_argument('max_extracted_candidates', default=200, dtype=int, msg='Max number of extracted candidates.')
    add_argument('threshold', default=0.05, dtype=float,
                 msg='Value of threshold to determine whether two words are matched.')
    add_argument('use_adapt', default=False, dtype=bool, msg='Flag to use adapter layer.')

    def __init__(self):
        super().__init__()
        emb_cls = DenseFeatEmbedding if g.dense_input else FeatEmbedding
        self.embedding = emb_cls('feat_emb', 'chosen_feat_group', 'char_emb')

        def _has_proper_length(segment):
            l = len(segment)
            return g.min_word_length <= l <= g.max_word_length

        with open(g.vocab_path, 'r', encoding='utf8') as fin:
            vocab = set(line.strip() for line in fin)
            segments = [Segment(w) for w in vocab]
            segments = [segment for segment in segments if _has_proper_length(segment)]
            lengths = torch.LongTensor(list(map(len, segments)))
            feat_matrix = [segment.feat_matrix for segment in segments]
            feat_matrix = torch.nn.utils.rnn.pad_sequence(feat_matrix, batch_first=True)
            max_len = lengths.max().item()
            source_padding = ~get_length_mask(lengths, max_len)
            self.register_buffer('vocab_feat_matrix', feat_matrix)
            self.register_buffer('vocab_source_padding', source_padding)
            self.register_buffer('vocab_length', lengths)
            self.vocab_feat_matrix.rename_('vocab', 'length', 'feat_group')
            self.vocab_source_padding.rename_('vocab', 'length')
            self.vocab_length.rename_('vocab')

            with Rename(self.vocab_feat_matrix, vocab='batch'):
                vocab_dense_feat_matrix = convert_to_dense(self.vocab_feat_matrix)
            self.vocab_dense_feat_matrix = {k: v.rename(batch='vocab') for k, v in vocab_dense_feat_matrix.items()}

            # Get the entire set of units from vocab.
            units = set()
            for segment in segments:
                units.update(segment.segment_list)
            self.id2unit = sorted(units)
            self.unit2id = {u: i for i, u in enumerate(self.id2unit)}
            # Now indexify the vocab. Gather feature matrices for units as well.
            indexed_segments = np.zeros([len(segments), max_len], dtype='int64')
            unit_feat_matrix = dict()
            for i, segment in enumerate(segments):
                indexed_segments[i, range(len(segment))] = [self.unit2id[u] for u in segment.segment_list]
                for j, u in enumerate(segment.segment_list):
                    if u not in unit_feat_matrix:
                        unit_feat_matrix[u] = segment.feat_matrix[j]
            unit_feat_matrix = [unit_feat_matrix[u] for u in self.id2unit]
            unit_feat_matrix = torch.nn.utils.rnn.pad_sequence(unit_feat_matrix, batch_first=True)
            self.register_buffer('unit_feat_matrix', unit_feat_matrix.unsqueeze(dim=1))
            self.register_buffer('indexed_segments', torch.from_numpy(indexed_segments))
            # Use dummy length to avoid the trouble later on.
            self.unit_feat_matrix.rename_('unit', 'length', 'feat_group')
            self.indexed_segments.rename_('vocab', 'length')
            with Rename(self.unit_feat_matrix, unit='batch'):
                unit_dense_feat_matrix = convert_to_dense(self.unit_feat_matrix)
            self.unit_dense_feat_matrix = {
                k: v.rename(batch='unit')
                for k, v in unit_dense_feat_matrix.items()
            }

        if g.use_adapt:
            assert g.dense_input
            self.adapter = AdaptLayer()

    @profile
    def forward(self, batch: ContinuousTextIpaBatch) -> ExtractModelReturn:
        if g.dense_input:
            # with Rename(self.vocab_source_padding, *self.vocab_dense_feat_matrix.values(), vocab='batch'):
            with Rename(*self.unit_dense_feat_matrix.values(), unit='batch'):
                word_repr = self.embedding(batch.dense_feat_matrix, batch.source_padding)
                dfm = self.adapter(self.unit_dense_feat_matrix)
                unit_repr = self.embedding(dfm)
        else:
            with Rename(self.unit_feat_matrix, unit='batch'):
                word_repr = self.embedding(batch.feat_matrix, batch.source_padding)
                unit_repr = self.embedding(self.unit_feat_matrix)
        unit_repr = unit_repr.squeeze('length')
        unit_repr.rename_(batch='unit')
        extracted = Extracted(batch.batch_size)
        # for i in range(g.max_num_words):
        new_extracted = self._extract_one_round(batch, extracted, word_repr, unit_repr)
        best_scores, best_inds = new_extracted.matches.score.flatten(['len_s', 'len_e'], 'cand').max(dim='cand')
        len_s = new_extracted.matches.score.size('len_s')
        len_e = new_extracted.matches.score.size('len_e')
        best_starts = best_inds // len_e
        # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
        best_ends = best_inds % len_e + best_starts + g.min_word_length - 1
        matched = new_extracted.matches.matched.flatten(['len_s', 'len_e'], 'cand')
        with NoName(matched):
            matched = matched.any(dim=-1)

        ret = ExtractModelReturn(best_scores, best_starts, best_ends, matched)

        return ret

    @profile
    def _extract_one_round(self, batch: ContinuousTextIpaBatch, extracted: Extracted, word_repr: FT, unit_repr: FT) -> Extracted:
        start_candidates = get_named_range(batch.max_length, 'len_s').align_to('batch', 'len_s', 'len_e')
        # Range from `min_word_length` to `max_word_length`.
        len_candidates = get_named_range(g.max_word_length + 1 - g.min_word_length, 'len_e') + g.min_word_length
        len_candidates = len_candidates.align_to('batch', 'len_s', 'len_e')
        # This is inclusive.
        end_candidates = start_candidates + len_candidates - 1

        # in_bound = (end_candidates < batch.lengths.align_as(end_candidates))
        # monotonic = (end_candidates > extracted.last_end.align_as(end_candidates))
        # viable = in_bound & monotonic
        viable = (end_candidates < batch.lengths.align_as(end_candidates))
        start_candidates = start_candidates.expand_as(viable)
        len_candidates = len_candidates.expand_as(viable)
        # NOTE(j_luo) Use `viable` to get the lengths. `len_candidates` has dummy axes. # IDEA(j_luo) Any better way of handling this?
        len_s = viable.size('len_s')
        len_e = viable.size('len_e')
        # end_candidates = end_candidates.expand_as(viable)
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        with NoName(start_candidates, end_candidates, len_candidates, bi, viable):
            viable_starts = start_candidates[viable].rename('viable')
            # viable_ends = end_candidates[viable].rename('viable')
            viable_lens = len_candidates[viable].rename('viable')
            viable_bi = bi[viable].rename('viable')

        viable_starts = viable_starts.align_to('viable', 'len_w')
        word_pos_offsets = get_named_range(g.max_word_length, 'len_w').align_as(viable_starts)
        word_pos = viable_starts + word_pos_offsets
        word_pos = word_pos.clamp(max=batch.max_length - 1)

        nh = NameHelper()
        viable_bi = viable_bi.expand_as(word_pos)
        word_pos = nh.flatten(word_pos, ['viable', 'len_w'], 'viable_X_len_w')
        viable_bi = nh.flatten(viable_bi, ['viable', 'len_w'], 'viable_X_len_w')
        # word_pos = word_pos.flatten([, 'len_w'], 'len_s_X_len_e_X_len_w')
        word_repr = word_repr.align_to('batch', 'length', 'char_emb')
        with NoName(word_repr, viable_bi, word_pos):
            extracted_word_repr = word_repr[viable_bi, word_pos].rename('viable_X_len_w', 'char_emb')
        extracted_word_repr = nh.unflatten(extracted_word_repr, 'viable_X_len_w', ['viable', 'len_w'])
        # extracted_word_repr = extracted_word_repr.unflatten('len_s_X_len_e_X_len_w',
        #                                                     [('len_s', batch.max_length), ('len_e', len_e), ('len_w', g.max_word_length)])

        # start_candidates_4d = start_candidates.align_to(..., 'len_w')
        # word_pos_offsets = get_named_range(len_e, 'len_w').align_as(start_candidates_4d)
        # word_pos = start_candidates_4d + word_pos_offsets
        # word_pos = word_pos.clamp(max=batch.max_length - 1).expand(-1, -1, g.max_word_length, -1)
        # word_pos = word_pos.flatten(['len_s', 'len_e', 'len_w'], 'len_s_X_len_e_X_len_w')
        # extracted_word_repr = word_repr.gather('length', word_pos)
        # extracted_word_repr = extracted_word_repr.unflatten('len_s_X_len_e_X_len_w',
        #                                                     [('len_s', batch.max_length), ('len_e', len_e), ('len_w', g.max_word_length)])

        matches = self._get_matches(extracted_word_repr, unit_repr, viable_lens)

        # Revert to the old shape.
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        lsi = get_named_range(len_s, 'len_s').expand_as(viable)
        lei = get_named_range(len_e, 'len_e').expand_as(viable)
        with NoName(bi, lsi, lei, viable, matches.score, matches.matched, matches.value):
            v_bi = bi[viable]
            v_lsi = lsi[viable]
            v_lei = lei[viable]

            def _unshape(tensor):
                shape = (batch.batch_size, len_s, len_e)
                if tensor.ndim > 1:
                    shape += tensor.shape[1:]
                ret = get_zeros(*shape).to(tensor.dtype)
                ret[v_bi, v_lsi, v_lei] = tensor
                return ret

            matches.score = _unshape(matches.score).rename('batch', 'len_s', 'len_e')
            matches.matched = _unshape(matches.matched).rename('batch', 'len_s', 'len_e')
            matches.value = _unshape(matches.value).rename('batch', 'len_s', 'len_e', 'vocab')

        new_extracted = Extracted(batch.batch_size, matches)
        return new_extracted

    @profile
    def _get_matches(self, extracted_word_repr: FT, unit_repr: FT, viable_lens: LT) -> Matches:
        # extracted_word_repr = extracted_word_repr.flatten(['len_w', 'char_emb'], 'len_w_X_char_emb')
        # vocab_repr = vocab_repr.flatten(['length', 'char_emb'], 'len_w_X_char_emb')
        # breakpoint()  # DEBUG(j_luo)
        # bs = extracted_word_repr.size('batch')
        # len_s = extracted_word_repr.size('len_s')
        # len_e = extracted_word_repr.size('len_e')
        d_char = extracted_word_repr.size('char_emb')
        # extracted_word_repr = extracted_word_repr.flatten(['batch', 'len_s', 'len_e'], 'batch_X_len_s_X_len_e')
        ns = extracted_word_repr.size('viable')
        # ns = extracted_word_repr.size('batch_X_len_s_X_len_e')
        nt = len(self.vocab_feat_matrix)
        msl = extracted_word_repr.size('len_w')
        mtl = self.vocab_feat_matrix.size('length')

        # not_viable = ~viable.flatten(['batch', 'len_s', 'len_e'], 'batch_X_len_s_X_len_e')
        # NOTE(j_luo) You need one extra position to keep 0-length outputs, and another one to dump invalid indices during DP.
        f = get_zeros(ns, nt, 2 + msl, 2 + mtl).fill_(99.9)
        for i in range(msl + 1):
            f[:, :, i, 0] = i
        for j in range(mtl + 1):
            f[:, :, 0, j] = j
        # with NoName(not_viable):
        #     f[not_viable] = 99.9

        def _get_cosine_matrix(x, y):
            # x = x.permute(1, 0, 2)
            # y = y.permute(1, 2, 0)
            dot = x @ y.t()
            with NoName(x, y):
                norms_x = x.norm(dim=-1, keepdim=True) + 1e-8
                norms_y = y.norm(dim=-1, keepdim=True) + 1e-8
            cos = dot / norms_x / norms_y.t()
            return (1.0 - cos) / 2

        nh = NameHelper()
        _extracted_word_repr = nh.flatten(extracted_word_repr, ['viable', 'len_w'], 'viable_X_len_w')
        cos = _get_cosine_matrix(_extracted_word_repr, unit_repr)
        # Name: ? x len_w x unit
        cos = nh.unflatten(cos, 'viable_X_len_w', ['viable', 'len_w'])

        # with NoName(self.indexed_segments, cos):
        #     all_ls = get_range(msl, 1, 0) + 1
        #     for s in range(2, msl + mtl + 1):
        #         all_lt = s - all_ls
        #         # Ignore ls and lt that are too far apart.
        #         close_to_diag = (all_ls - all_lt).abs() <= 2
        #         ls = all_ls[close_to_diag]
        #         lt = all_lt[close_to_diag]

        #         # NOTE(j_luo) Use one set of `lt` to write to invalid indices (the extra position at the end).
        #         invalid = (lt < 1) | (lt > mtl)
        #         lt[invalid] = mtl + 1
        #         # NOTE(j_luo) Use another set of `lt` to get invalid indices (set to 1 so that lt_get - 1 == 0).
        #         lt_get = lt.clone()
        #         lt_get[invalid] = 1

        #         # _extracted = extracted_word_repr[:, ls - 1].view(-1, msl, d_char)
        #         # _vocab = vocab_repr[:, lt_repr - 1].view(-1, msl, d_char)
        #         # diff = _get_cosine_matrix(_extracted, _vocab)
        #         # diff = diff.permute(1, 2, 0)
        #         vocab_inds = self.indexed_segments[:, lt_get - 1]
        #         diff = cos[:, ls - 1, vocab_inds]

        #         ins_s = f[:, :, ls - 1, lt] + 1
        #         del_s = f[:, :, ls, lt - 1] + 1
        #         sub_s = f[:, :, ls - 1, lt - 1] + diff
        #         all_s = torch.stack([ins_s, del_s, sub_s], dim=-1)
        #         f[:, :, ls, lt], _ = all_s.min(dim=-1)

        with NoName(self.indexed_segments, cos):
            for ls in range(1, msl + 1):
                min_lt = max(ls - 2, 1)
                max_lt = min(ls + 2, mtl + 1)
                for lt in range(min_lt, max_lt):
                    # # NOTE(j_luo) Off by 1 for these representation.
                    # _extracted = extracted_word_repr[:, ls - 1]
                    # _vocab = vocab_repr[:, lt - 1]

                    # diff = _get_cosine_matrix(_extracted, _vocab).rename(None)
                    vocab_inds = self.indexed_segments[:, lt - 1]
                    diff = cos[:, ls - 1, vocab_inds]

                    ins_s = f[:, :, ls - 1, lt] + 1
                    del_s = f[:, :, ls, lt - 1] + 1
                    sub_s = f[:, :, ls - 1, lt - 1] + diff
                    all_s = torch.stack([ins_s, del_s, sub_s], dim=-1)
                    # f[:, :, i, j] = torch.min(ins_s, torch.min(del_s, sub_s))
                    f[:, :, ls, lt], _ = all_s.min(dim=-1)

        # scores, _ = f.view(ns, -1).min(dim=-1)
        f.rename_('viable', 'vocab', 'len_w_src', 'len_w_tgt')
        # f = f.unflatten('batch_X_len_s_X_len_e', [('batch', bs), ('len_s', len_s), ('len_e', len_e)])
        with NoName(f, viable_lens, self.vocab_length):
            idx_src = viable_lens.unsqueeze(dim=-1)
            idx_tgt = self.vocab_length
            viable_i = get_range(ns, 2, 0)
            vocab_i = get_range(len(self.vocab_length), 2, 1)

            value = f[viable_i, vocab_i, idx_src, idx_tgt]
            value.rename_('viable', 'vocab')

            # idx_src = viabel.expand(bs, len_s, len_e).unsqueeze(dim=-1)

            # bi = get_range(bs, 4, 0)
            # si = get_range(len_s, 4, 1)
            # ei = get_range(len_e, 4, 2)
            # vi = get_range(len(self.vocab_length), 4, 3)
            # idx_src = len_candidates.expand(bs, len_s, len_e).unsqueeze(dim=-1)
            # idx_tgt = self.vocab_length
            # value = f[bi, si, ei, vi, idx_src, idx_tgt]
            # value.rename_('batch', 'len_s', 'len_e', 'vocab')

        best_value, matched_vocab = value.min(dim='vocab')
        # matched_vocab = matched_vocab.flatten(['batch', 'len_s', 'len_e'], 'batch_X_len_s_X_len_e')
        lengths = self.vocab_length.gather('vocab', matched_vocab)
        # lengths = lengths.unflatten('batch_X_len_s_X_len_e', [('batch', bs), ('len_s', len_s), ('len_e', len_e)])
        matched = best_value < g.threshold
        # # DEBUG(j_luo)
        # try:
        #     self._thresh -= 0.005
        # except:
        # self._thresh = g.threshold
        # # self._thresh = max(self._thresh, 0.2)
        # print(self._thresh)
        self._thresh = g.threshold

        score = lengths * (1.0 - best_value / self._thresh).clamp(min=0.0)
        matches = Matches(None, score, value, matched)
        return matches

        # dot = extracted_word_repr @ vocab_repr.align_to('len_w_X_char_emb', 'vocab')
        # with NoName(extracted_word_repr, vocab_repr):
        #     norms1 = extracted_word_repr.norm(dim=-1, keepdim=True)
        #     norms2 = vocab_repr.norm(dim=-1, keepdim=True)
        # cos = dot / norms1 / norms2.t()
