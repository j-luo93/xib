import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_zeros
from dev_misc.devlib import (BaseBatch, batch_class, get_array,
                             get_length_mask, get_range)
from dev_misc.devlib.named_tensor import (NameHelper, NoName, Rename,
                                          drop_names, get_named_range)
from dev_misc.utils import WithholdKeys
from xib.data_loader import ContinuousTextIpaBatch, convert_to_dense
from xib.ipa import should_include
from xib.ipa.process import Segment, SegmentWindow
from xib.model.modules import AdaptLayer, FeatEmbedding

from .modules import DenseFeatEmbedding


@batch_class
class Matches(BaseBatch):
    end: LT  # Inclusive
    score: FT
    value: FT
    matched: BT
    matched_vocab: LT


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
    matched_vocab: LT
    extracted: Extracted


class ExtractModel(nn.Module):

    add_argument('max_num_words', default=3, dtype=int, msg='Max number of extracted words.')
    add_argument('max_word_length', default=10, dtype=int, msg='Max length of extracted words.')
    add_argument('max_extracted_candidates', default=200, dtype=int, msg='Max number of extracted candidates.')
    add_argument('init_threshold', default=0.05, dtype=float,
                 msg='Initial value of threshold to determine whether two words are matched.')
    add_argument('use_adapt', default=False, dtype=bool, msg='Flag to use adapter layer.')
    add_argument('use_embedding', default=True, dtype=bool, msg='Flag to use embedding.')
    add_argument('use_hamming', default=False, dtype=bool, msg='Flag to use hamming distance instead of cosine.')
    add_argument('anneal_factor', default=0.999, dtype=float, msg='Mulplication value for annealing.')

    def __init__(self):
        super().__init__()

        if g.use_embedding:
            emb_cls = DenseFeatEmbedding if g.dense_input else FeatEmbedding
            self.embedding = emb_cls('feat_emb', 'chosen_feat_group', 'char_emb')
        elif not g.dense_input:
            raise ValueError(f'Use embedding for sparse inputs.')

        def _has_proper_length(segment):
            l = len(segment)
            return g.min_word_length <= l <= g.max_word_length

        with open(g.vocab_path, 'r', encoding='utf8') as fin:
            _vocab = set(line.strip() for line in fin)
            segments = [Segment(w) for w in _vocab]
            self.vocab = get_array([segment for segment in segments if _has_proper_length(segment)])
            lengths = torch.LongTensor(list(map(len, self.vocab)))
            feat_matrix = [segment.feat_matrix for segment in self.vocab]
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
            for segment in self.vocab:
                units.update(segment.segment_list)
            self.id2unit = sorted(units)
            self.unit2id = {u: i for i, u in enumerate(self.id2unit)}
            # Now indexify the vocab. Gather feature matrices for units as well.
            indexed_segments = np.zeros([len(self.vocab), max_len], dtype='int64')
            unit_feat_matrix = dict()
            for i, segment in enumerate(self.vocab):
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

    _special_state_keys = ['vocab', 'vocab_dense_feat_matrix', 'unit2id', 'id2unit', 'unit_dense_feat_matrix']

    def state_dict(self, **kwargs):
        state = super().state_dict(**kwargs)
        for key in self._special_state_keys:
            attr = drop_names(getattr(self, key))
            state[key] = attr
        return state

    def load_state_dict(self, state_dict: Dict, **kwargs):
        with WithholdKeys(state_dict, *self._special_state_keys):
            super().load_state_dict(state_dict, **kwargs)
        # HACK(j_luo)
        for key in self._special_state_keys:
            attr = getattr(self, key)
            setattr(self, key, state_dict[key])
            if torch.is_tensor(attr):
                names = attr.names
                getattr(self, key).rename_(*names)
            elif isinstance(attr, dict):
                for k, v in getattr(self, key).items():
                    if torch.is_tensor(v):
                        v.rename_(*attr[k].names)

    # ------------------------ Debug section ----------------------- #
    # DEBUG(j_luo)

    def get_vector(self, c: str, adapt: bool = False):
        c = self._str2sw(c)
        fm = c.feat_matrix.rename('length', 'feat_group').align_to('batch', ...)
        dfm = convert_to_dense(fm)
        if adapt:
            dfm = self.adapter(dfm)
        names = [name for name in dfm if should_include(g.feat_groups, name)]
        names = sorted(names, key=lambda name: name.value)
        with NoName(*dfm.values()):
            word_repr = torch.cat([dfm[name] for name in names], dim=-1)
        word_repr = word_repr.rename('batch', 'length', 'char_emb').squeeze(dim='batch').squeeze(dim='length')
        return word_repr

    def _str2sw(self, c: str):
        return SegmentWindow([Segment(c)])

    def get_char_sim(self, c1: str, c2: str, adapt: bool = False):
        v1 = self.get_vector(c1, adapt=adapt)
        v2 = self.get_vector(c2)
        return (v1 * v2).sum() / (v1 ** 2).sum().sqrt() / (v2 ** 2).sum().sqrt()

    def get_char_hamming(self, c1: str, c2: str, adapt: bool = False):
        v1 = self.get_vector(c1, adapt=adapt)
        v2 = self.get_vector(c2)
        return (v1 - v2).abs().sum()

    # DEBUG(j_luo)
    add_argument('debug', dtype=bool, default=False)

    def forward(self, batch: ContinuousTextIpaBatch) -> ExtractModelReturn:
        self.get_char_hamming('a', 'a')
        if g.debug:
            torch.set_printoptions(sci_mode=False, linewidth=200)
            self._thresh = 0.5
            self.eval()
            breakpoint()

        # Prepare representations.
        if g.dense_input:
            with Rename(*self.unit_dense_feat_matrix.values(), unit='batch'):
                dfm = self.adapter(batch.dense_feat_matrix)
            if g.use_embedding:
                word_repr = self.embedding(dfm, batch.source_padding)
                unit_repr = self.embedding(self.unit_dense_feat_matrix)
            else:
                names = sorted(dfm, key=lambda name: name.value)
                # IDEA(j_luo) NoName shouldn't use reveal_name. Just keep the name in the context manager.
                with NoName(*self.unit_dense_feat_matrix.values(), *dfm.values()):
                    word_repr = torch.cat([dfm[name] for name in names], dim=-1)
                    unit_repr = torch.cat([self.unit_dense_feat_matrix[name] for name in names], dim=-1)
                word_repr.rename_('batch', 'length', 'char_emb')
                unit_repr.rename_('batch', 'length', 'char_emb')
        else:
            with Rename(self.unit_feat_matrix, unit='batch'):
                word_repr = self.embedding(batch.feat_matrix, batch.source_padding)
                unit_repr = self.embedding(self.unit_feat_matrix)
        unit_repr = unit_repr.squeeze('length')
        unit_repr.rename_(batch='unit')

        # Main body: extract one span.
        extracted = Extracted(batch.batch_size)
        new_extracted = self._extract_one_span(batch, extracted, word_repr, unit_repr)

        # Get the best score and span.
        flat_scores = new_extracted.matches.score.flatten(['len_s', 'len_e'], 'cand')
        if self.training:
            best_scores, best_inds = flat_scores.max(dim='cand')
            softmax = (flat_scores / 0.1).softmax(dim='cand')
            best_scores = (flat_scores * softmax).sum(dim='cand')
        else:
            best_scores, best_inds = flat_scores.max(dim='cand')
        len_s = new_extracted.matches.score.size('len_s')
        len_e = new_extracted.matches.score.size('len_e')
        starts = best_inds // len_e
        # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
        ends = best_inds % len_e + starts + g.min_word_length - 1
        matched = new_extracted.matches.matched.flatten(['len_s', 'len_e'], 'cand')
        with NoName(matched):
            matched = matched.any(dim=-1)
        matched_vocab = new_extracted.matches.matched_vocab.flatten(['len_s', 'len_e'], 'cand')
        matched_vocab = matched_vocab.gather('cand', best_inds)

        ret = ExtractModelReturn(best_scores, starts, ends, matched, matched_vocab, new_extracted)

        return ret

    def _extract_one_span(self, batch: ContinuousTextIpaBatch, extracted: Extracted, word_repr: FT, unit_repr: FT) -> Extracted:
        # Propose all span start/end positions.
        start_candidates = get_named_range(batch.max_length, 'len_s').align_to('batch', 'len_s', 'len_e')
        # Range from `min_word_length` to `max_word_length`.
        len_candidates = get_named_range(g.max_word_length + 1 - g.min_word_length, 'len_e') + g.min_word_length
        len_candidates = len_candidates.align_to('batch', 'len_s', 'len_e')
        # This is inclusive.
        end_candidates = start_candidates + len_candidates - 1

        # Only keep the viable/valid spans around.
        viable = (end_candidates < batch.lengths.align_as(end_candidates))
        start_candidates = start_candidates.expand_as(viable)
        len_candidates = len_candidates.expand_as(viable)
        # NOTE(j_luo) Use `viable` to get the lengths. `len_candidates` has dummy axes. # IDEA(j_luo) Any better way of handling this?
        len_s = viable.size('len_s')
        len_e = viable.size('len_e')
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        with NoName(start_candidates, end_candidates, len_candidates, bi, viable):
            viable_starts = start_candidates[viable].rename('viable')
            viable_lens = len_candidates[viable].rename('viable')
            viable_bi = bi[viable].rename('viable')

        # Get the word positions to get the corresponding representations.
        viable_starts = viable_starts.align_to('viable', 'len_w')
        word_pos_offsets = get_named_range(g.max_word_length, 'len_w').align_as(viable_starts)
        word_pos = viable_starts + word_pos_offsets
        word_pos = word_pos.clamp(max=batch.max_length - 1)

        # Get the corresponding representations.
        nh = NameHelper()
        viable_bi = viable_bi.expand_as(word_pos)
        word_pos = nh.flatten(word_pos, ['viable', 'len_w'], 'viable_X_len_w')
        viable_bi = nh.flatten(viable_bi, ['viable', 'len_w'], 'viable_X_len_w')
        word_repr = word_repr.align_to('batch', 'length', 'char_emb')
        with NoName(word_repr, viable_bi, word_pos):
            extracted_word_repr = word_repr[viable_bi, word_pos].rename('viable_X_len_w', 'char_emb')
        extracted_word_repr = nh.unflatten(extracted_word_repr, 'viable_X_len_w', ['viable', 'len_w'])

        # Main body: Run DP to find the best matches.
        matches = self._get_matches(extracted_word_repr, unit_repr, viable_lens)

        # Revert to the old shape (so that invalid spans are included).
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        lsi = get_named_range(len_s, 'len_s').expand_as(viable)
        lei = get_named_range(len_e, 'len_e').expand_as(viable)
        with NoName(bi, lsi, lei, viable, matches.score, matches.matched, matches.value, matches.matched_vocab):
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

            # if self.training:
            #     matches.score = _unshape(matches.score).rename('batch', 'len_s', 'len_e', 'vocab')
            #     matches.matched = _unshape(matches.matched).rename('batch', 'len_s', 'len_e', 'vocab')
            # else:
            matches.score = _unshape(matches.score).rename('batch', 'len_s', 'len_e')
            matches.matched = _unshape(matches.matched).rename('batch', 'len_s', 'len_e')
            matches.matched_vocab = _unshape(matches.matched_vocab).rename('batch', 'len_s', 'len_e')
            matches.value = _unshape(matches.value).rename('batch', 'len_s', 'len_e', 'vocab')

        new_extracted = Extracted(batch.batch_size, matches)
        return new_extracted

    def _get_matches(self, extracted_word_repr: FT, unit_repr: FT, viable_lens: LT) -> Matches:
        d_char = extracted_word_repr.size('char_emb')
        ns = extracted_word_repr.size('viable')
        nt = len(self.vocab_feat_matrix)
        msl = extracted_word_repr.size('len_w')
        mtl = self.vocab_feat_matrix.size('length')

        # # NOTE(j_luo) Use dictionary save every state.
        fs = dict()
        for i in range(msl + 1):
            fs[(i, 0)] = get_zeros(ns, nt).fill_(i)
        for j in range(mtl + 1):
            fs[(0, j)] = get_zeros(ns, nt).fill_(j)

        # Initialize f scores.
        # # NOTE(j_luo) You need one extra position to keep 0 - length outputs, and another one to dump invalid indices during DP.
        # f = get_zeros(1 + msl, 1 + mtl, ns, nt).fill_(99.9)
        # for i in range(msl + 1):
        #     f[i, 0] = i
        # for j in range(mtl + 1):
        #     f[0, j] = j
        # f = get_zeros(ns, nt, 1 + msl, 1 + mtl).fill_(99.9)
        # for i in range(msl + 1):
        #     f[:, :, i, 0] = i
        # for j in range(mtl + 1):
        #     f[:, :, 0, j] = j

        # Compute cosine distance all at once: for each viable span, compare it against all units.
        def _get_cosine_matrix(x, y):
            dot = x @ y.t()
            with NoName(x, y):
                norms_x = x.norm(dim=-1, keepdim=True) + 1e-8
                norms_y = y.norm(dim=-1, keepdim=True) + 1e-8
            cos = dot / norms_x / norms_y.t()
            return (1.0 - cos) / 2

        # IDEA(j_luo) Add temp argument for break points.
        def _get_hamming_matrix(x, y):
            with NoName(x, y):
                hamming = (x.unsqueeze(dim=1) - y).abs().sum(dim=-1)
            return hamming.rename_(x.names[0], y.names[0]) / 4

        nh = NameHelper()
        _extracted_word_repr = nh.flatten(extracted_word_repr, ['viable', 'len_w'], 'viable_X_len_w')
        dist_func = _get_hamming_matrix if g.use_hamming else _get_cosine_matrix
        costs = dist_func(_extracted_word_repr, unit_repr)
        # # Rescale the cosine distance.
        # _max, _ = cos.max(dim='unit', keepdim=True)
        # _min, _ = cos.min(dim='unit', keepdim=True)
        # cos = (cos - _min) / (_max - _min + 1e-8)

        # Name: viable x len_w x unit
        costs = nh.unflatten(costs, 'viable_X_len_w', ['viable', 'len_w'])

        # ------------------------ Main body: DP ----------------------- #

        # Transition.
        with NoName(self.indexed_segments, costs):
            for ls in range(1, msl + 1):
                min_lt = max(ls - 2, 1)
                max_lt = min(ls + 2, mtl + 1)
                for lt in range(min_lt, max_lt):
                    transitions = list()
                    # DEBUG(j_luo) Ignore insertions/deletions for now.
                    if (ls - 1, lt) in fs:
                        transitions.append(fs[(ls - 1, lt)] + 100)
                    if (ls, lt - 1) in fs:
                        transitions.append(fs[(ls, lt - 1)] + 100)
                    if (ls - 1, lt - 1) in fs:
                        vocab_inds = self.indexed_segments[:, lt - 1]
                        sub_cost = costs[:, ls - 1, vocab_inds]
                        transitions.append(fs[(ls - 1, lt - 1)] + sub_cost)
                    if transitions:
                        all_s = torch.stack(transitions, dim=-1)
                        new_s, _ = all_s.min(dim=-1)
                        fs[(ls, lt)] = new_s

                    # vocab_inds = self.indexed_segments[:, lt - 1]
                    # sub_cost = costs[:, ls - 1, vocab_inds]

                    # ins_s = f[ls - 1, lt] + 1
                    # del_s = f[ls, lt - 1] + 1
                    # sub_s = f[ls - 1, lt - 1] + sub_cost
                    # all_s = torch.stack([ins_s, del_s, sub_s], dim=-1)
                    # f[ls, lt], _ = all_s.min(dim=-1)

                    # ins_s = f[:, :, ls - 1, lt] + 1
                    # del_s = f[:, :, ls, lt - 1] + 1
                    # sub_s = f[:, :, ls - 1, lt - 1] + sub_cost
                    # all_s = torch.stack([ins_s, del_s, sub_s], dim=-1)
                    # f[:, :, ls, lt], _ = all_s.min(dim=-1)

        f_lst = list()
        for i in range(msl + 1):
            for j in range(mtl + 1):
                if (i, j) not in fs:
                    fs[(i, j)] = get_zeros(ns, nt).fill_(99.9)
                f_lst.append(fs[(i, j)])
        f = torch.stack(f_lst, dim=0).view(msl + 1, mtl + 1, -1, len(self.vocab))
        f.rename_('len_w_src', 'len_w_tgt', 'viable', 'vocab')
        # ls_idx, lt_idx = zip(*fs.keys())

        # Get the values wanted.
        # DEBUG(j_luo)
        RELATIVE = False

        # f.rename_('viable', 'vocab', 'len_w_src', 'len_w_tgt')
        if g.debug:
            breakpoint()  # DEBUG(j_luo)
        with NoName(f, viable_lens, self.vocab_length):
            idx_src = viable_lens.unsqueeze(dim=-1)
            idx_tgt = self.vocab_length
            viable_i = get_range(ns, 2, 0)
            vocab_i = get_range(len(self.vocab_length), 2, 1)

            value = f[idx_src, idx_tgt, viable_i, vocab_i]
            # value = f[viable_i, vocab_i, idx_src, idx_tgt]
            value.rename_('viable', 'vocab')

            # Normalize the values by length.
            if RELATIVE:
                min_len = torch.min(idx_src, self.vocab_length)
                value = value / min_len

        # Get the best spans.
        if self.training:
            # DEBUG(j_luo)
            try:
                self._thresh *= g.anneal_factor
            except AttributeError:
                self._thresh = g.init_threshold
            MIN_TH = 0.2 if RELATIVE else 0.001
            self._thresh = max(self._thresh, MIN_TH)
            logging.debug(self._thresh)

        best_value, matched_vocab = value.min(dim='vocab')
        matched = best_value < self._thresh

        if self.training:
            # DEBUG(j_luo)
            # TEMPERATURE = 0.05 if RELATIVE else 0.1
            TEMPERATURE = 1.0
            softmin = nn.functional.softmin(value / TEMPERATURE, dim='vocab')
            best_value = (value * softmin).sum(dim='vocab')

        lengths = self.vocab_length.gather('vocab', matched_vocab)
        score = lengths * (1.0 - best_value / self._thresh).clamp(min=0.0)
        matches = Matches(None, score, value, matched, matched_vocab)
        return matches
