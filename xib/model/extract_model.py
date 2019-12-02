from typing import Optional

import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_zeros
from dev_misc.devlib import BaseBatch, batch_class, get_length_mask, get_range
from dev_misc.devlib.named_tensor import NoName, get_named_range
from xib.data_loader import ContinuousTextIpaBatch
from xib.ipa.process import Segment
from xib.model.modules import FeatEmbedding


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

    def __init__(self):
        super().__init__()
        self.embedding = FeatEmbedding('feat_emb', 'chosen_feat_group', 'char_emb')

        def _has_proper_length(segment):
            l = len(segment)
            return l <= g.max_word_length and l >= g.min_word_length

        with open(g.vocab_path, 'r', encoding='utf8') as fin:
            vocab = set(line.strip() for line in fin)
            segments = [Segment(w) for w in vocab]
            segments = [segment for segment in segments if _has_proper_length(segment)]
            lengths = torch.LongTensor(list(map(len, segments)))
            feat_matrix = [segment.feat_matrix for segment in segments]
            feat_matrix = torch.nn.utils.rnn.pad_sequence(feat_matrix, batch_first=True)
            source_padding = ~get_length_mask(lengths, lengths.max().item())
            self.register_buffer('vocab_feat_matrix', feat_matrix)
            self.register_buffer('vocab_source_padding', source_padding)
            self.register_buffer('vocab_length', lengths)
            self.vocab_feat_matrix.rename_('vocab', 'length', 'feat_group')
            self.vocab_source_padding.rename_('vocab', 'length')
            self.vocab_length.rename_('vocab')

    def forward(self, batch: ContinuousTextIpaBatch) -> ExtractModelReturn:
        word_repr = self.embedding(batch.feat_matrix, batch.source_padding)
        vocab_repr = self.embedding(self.vocab_feat_matrix.rename(vocab='batch'),
                                    self.vocab_source_padding.rename(vocab='batch'))
        vocab_repr.rename_(batch='vocab')
        extracted = Extracted(batch.batch_size)
        # for i in range(g.max_num_words):
        new_extracted = self._extract_one_round(batch, extracted, word_repr, vocab_repr)
        best_scores, best_inds = new_extracted.matches.score.flatten(['len_s', 'len_e'], 'cand').max(dim='cand')
        len_s = new_extracted.matches.score.size('len_s')
        len_e = new_extracted.matches.score.size('len_e')
        best_starts = best_inds // len_e
        best_ends = best_inds % len_e + best_starts
        matched = new_extracted.matches.matched.flatten(['len_s', 'len_e'], 'cand')
        with NoName(matched):
            matched = matched.any(dim=-1)

        ret = ExtractModelReturn(best_scores, best_starts, best_ends, matched)

        return ret

    def _extract_one_round(self, batch: ContinuousTextIpaBatch, extracted: Extracted, word_repr: FT, vocab_repr: FT) -> Extracted:
        start_candidates = get_named_range(batch.max_length, 'len_s').align_to('batch', 'len_s', 'len_e')
        len_candidates = get_named_range(g.max_word_length, 'len_e').align_to('batch', 'len_s', 'len_e') + 1
        # This is inclusive.
        end_candidates = start_candidates + len_candidates - 1

        # in_bound = (end_candidates < batch.lengths.align_as(end_candidates))
        # monotonic = (end_candidates > extracted.last_end.align_as(end_candidates))
        # viable = in_bound & monotonic
        viable = (end_candidates < batch.lengths.align_as(end_candidates))

        word_repr = word_repr.align_to('batch', 'length', 'char_emb')
        start_candidates_4d = start_candidates.align_to(..., 'len_w')
        word_pos_offsets = get_named_range(g.max_word_length, 'len_w').align_as(start_candidates_4d)
        word_pos = start_candidates_4d + word_pos_offsets
        word_pos = word_pos.clamp(max=batch.max_length - 1).expand(-1, -1, g.max_word_length, -1)
        word_pos = word_pos.flatten(['len_s', 'len_e', 'len_w'], 'len_s_X_len_e_X_len_w')
        extracted_word_repr = word_repr.gather('length', word_pos)
        extracted_word_repr = extracted_word_repr.unflatten('len_s_X_len_e_X_len_w',
                                                            [('len_s', batch.max_length), ('len_e', g.max_word_length), ('len_w', g.max_word_length)])

        matches = self._get_matches(extracted_word_repr, vocab_repr, len_candidates, viable)

        new_extracted = Extracted(batch.batch_size, matches)
        return new_extracted

    def _get_matches(self, extracted_word_repr: FT, vocab_repr: FT, len_candidates: LT, viable: BT) -> Matches:
        # extracted_word_repr = extracted_word_repr.flatten(['len_w', 'char_emb'], 'len_w_X_char_emb')
        # vocab_repr = vocab_repr.flatten(['length', 'char_emb'], 'len_w_X_char_emb')
        vocab_repr = vocab_repr.rename(length='len_w')
        bs = extracted_word_repr.size('batch')
        len_s = extracted_word_repr.size('len_s')
        len_e = extracted_word_repr.size('len_e')
        extracted_word_repr = extracted_word_repr.flatten(['batch', 'len_s', 'len_e'], 'batch_X_len_s_X_len_e')

        ns = extracted_word_repr.size('batch_X_len_s_X_len_e')
        nt = vocab_repr.size('vocab')
        msl = extracted_word_repr.size('len_w')
        mtl = vocab_repr.size('len_w')

        not_viable = ~viable.flatten(['batch', 'len_s', 'len_e'], 'batch_X_len_s_X_len_e')
        f = get_zeros(ns, nt, 1 + msl, 1 + mtl).fill_(99.9)
        for i in range(msl + 1):
            f[:, :, i, 0] = i
        for j in range(mtl + 1):
            f[:, :, 0, j] = j
        with NoName(not_viable):
            f[not_viable] = 99.9

        def _get_cosine_matrix(x, y):
            dot = x @ y.t()
            with NoName(x, y):
                norms_x = x.norm(dim=-1, keepdim=True) + 1e-8
                norms_y = y.norm(dim=-1, keepdim=True) + 1e-8
            cos = dot / norms_x / norms_y.t()
            return (1.0 - cos) / 2

        for i in range(1, msl + 1):
            min_j = max(i - 2, 1)
            max_j = min(i + 2, mtl + 1)
            for j in range(min_j, max_j):
                # NOTE(j_luo) Off by 1 for these representation.
                _extracted = extracted_word_repr[:, i - 1]
                _vocab = vocab_repr[:, j - 1]

                diff = _get_cosine_matrix(_extracted, _vocab).rename(None)
                ins_s = f[:, :, i - 1, j] + 1
                del_s = f[:, :, i, j - 1] + 1
                sub_s = f[:, :, i - 1, j - 1] + diff
                all_s = torch.stack([ins_s, del_s, sub_s], dim=-1)
                f[:, :, i, j], _ = all_s.min(dim=-1)

        # scores, _ = f.view(ns, -1).min(dim=-1)
        f.rename_('batch_X_len_s_X_len_e', 'vocab', 'len_w_src', 'len_w_tgt')
        f = f.unflatten('batch_X_len_s_X_len_e', [('batch', bs), ('len_s', len_s), ('len_e', len_e)])
        with NoName(f, len_candidates, self.vocab_length):
            bi = get_range(bs, 4, 0)
            si = get_range(len_s, 4, 1)
            ei = get_range(len_e, 4, 2)
            vi = get_range(len(self.vocab_length), 4, 3)
            idx_src = len_candidates.expand(bs, len_s, len_e).unsqueeze(dim=-1)
            idx_tgt = self.vocab_length
            value = f[bi, si, ei, vi, idx_src, idx_tgt]
            value.rename_('batch', 'len_s', 'len_e', 'vocab')

        best_value, matched_vocab = value.min(dim='vocab')
        matched_vocab = matched_vocab.flatten(['batch', 'len_s', 'len_e'], 'batch_X_len_s_X_len_e')
        lengths = self.vocab_length.gather('vocab', matched_vocab)
        lengths = lengths.unflatten('batch_X_len_s_X_len_e', [('batch', bs), ('len_s', len_s), ('len_e', len_e)])
        matched = (best_value < 0.05)
        score = lengths * matched * (1.0 - best_value)
        matches = Matches(None, score, value, matched)
        return matches

        # dot = extracted_word_repr @ vocab_repr.align_to('len_w_X_char_emb', 'vocab')
        # with NoName(extracted_word_repr, vocab_repr):
        #     norms1 = extracted_word_repr.norm(dim=-1, keepdim=True)
        #     norms2 = vocab_repr.norm(dim=-1, keepdim=True)
        # cos = dot / norms1 / norms2.t()
