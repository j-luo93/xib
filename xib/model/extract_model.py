import logging
import math
from dataclasses import fields
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_zeros
from dev_misc.devlib import (BaseBatch, batch_class, get_array,
                             get_length_mask, get_range)
from dev_misc.devlib.named_tensor import (NameHelper, NoName, Rename,
                                          drop_names, get_named_range)
from dev_misc.utils import WithholdKeys, global_property
from xib.data_loader import (ContinuousIpaBatch, UnbrokenTextBatch,
                             convert_to_dense)
from xib.ipa import Category, get_enum_by_cat, should_include
from xib.ipa.process import Segment, SegmentWindow
from xib.model.modules import AdaptLayer, FeatEmbedding

from .modules import DenseFeatEmbedding

ExtractBatch = Union[ContinuousIpaBatch, UnbrokenTextBatch]


@batch_class
class Matches(BaseBatch):
    ed_dist: FT


@batch_class
class MatchesLv0(Matches):
    matched_ed_dist: FT
    matched_vocab: LT
    matched_length: LT
    matched_thresh: FT
    matched_score: FT


@batch_class
class MatchesLv1(MatchesLv0):
    pass


@batch_class
class MatchesLv2(Matches):
    thresh: FT
    matched_thresh: FT
    matched_vocab: LT
    matched_length: LT
    matched_score: FT


@batch_class
class MatchesLv3(Matches):
    thresh: FT
    score: FT
    matched_score: FT
    matched_vocab: LT


@batch_class
class MatchesLv4(Matches):
    thresh: FT
    score: FT


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
    start: LT
    end: LT
    best_matched_score: FT
    best_matched_vocab: LT
    extracted: Extracted
    adapted_dfm: Dict[Category, FT]


def _soft_threshold(x, thresh):
    return (nn.functional.celu(1 - 2 * x / thresh) + 1) / 2


def _linear_threshold(x, thresh):
    return 1 - x / thresh


def _exp_linear_threshold(x, thresh):
    pt = - thresh * math.log(thresh)
    return torch.where(x > pt, -x / thresh, (-x / thresh).exp())


def _exp_threshold(x, thresh):
    return (-x / thresh).exp()


def _soft_max(x, dim):
    w = (x / g.temperature).log_softmax(dim=dim).exp()
    value = (x * w).sum(dim)
    _, index = x.max(dim=dim)
    return value, index


def _soft_min(x, dim):
    w = (-x / g.temperature).log_softmax(dim=dim).exp()
    value = (x * w).sum(dim)
    _, index = x.min(dim=dim)
    return value, index


def _restore_shape(tensor, bi, lsi, lei, viable, value: Optional[float] = None):
    bs = bi.size('batch')
    len_s = lsi.size('len_s')
    len_e = lei.size('len_e')

    shape = (bs, len_s, len_e)
    names = ('batch', 'len_s', 'len_e')
    if tensor.ndim > 1:
        shape += tensor.shape[1:]
        names += tensor.names[1:]

    with NoName(bi, lsi, lei, viable, tensor):
        v_bi = bi[viable]
        v_lsi = lsi[viable]
        v_lei = lei[viable]
        ret = get_zeros(*shape).to(tensor.dtype)
        if value is not None:
            ret.fill_(value)
        ret[v_bi, v_lsi, v_lei] = tensor

    ret.rename_(*names)
    return ret


class G2PLayer(nn.Module):

    add_argument('g2p_window_size', default=3, dtype=int, msg='Window size for g2p layer.')

    def __init__(self, unit_vocab_size: int, dataset):  # FIXME(j_luo) remove dataset
        super().__init__()

        # # DEBUG(j_luo)
        # logging.warn('Hacking it right now.')
        # units = torch.cat([Segment(u).feat_matrix for u in dataset.id2unit], dim=0)

        # from xib.ipa import Index, get_enum_by_cat, Category
        # xx = list()
        # for unit in units:
        #     x = list()
        #     for u, cat in zip(unit[:7], Category):
        #         if cat.name[0] in ['P', 'C', 'V']:
        #             e = get_enum_by_cat(cat)
        #             f_idx = Index.get_feature(int(u)).value.f_idx
        #             x.extend([0] * (f_idx) + [1] + [0] * (len(e) - f_idx - 1))
        #     assert len(x) == 60
        #     xx.append(x)
        # units = torch.LongTensor(xx)

        self.unit_embedding = nn.Embedding(unit_vocab_size, 60)  # Vg.dim)
        # self.unit_embedding.weight.data.copy_(units)

        # self.unit_embedding = nn.Embedding(unit_vocab_size, g.dim)
        self.conv = nn.Conv1d(g.dim, g.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)

        module_dict = dict()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                e = get_enum_by_cat(cat)
                nf = len(e)
                module_dict[cat.name] = nn.Linear(g.dim, nf)
        self.p_predictors = nn.ModuleDict(module_dict)

    def forward(self, unit_id_seqs: LT) -> Dict[Category, FT]:
        unit_embeddings = self.unit_embedding(unit_id_seqs).refine_names(..., 'unit_emb')
        return unit_embeddings

        # # DEBUG(j_luo)
        # logging.warn('HACKING')
        # output = unit_embeddings
        # ret = dict()
        # offset = 0
        # for i, cat in enumerate(Category):
        #     if should_include(g.feat_groups, cat):
        #         l = len(get_enum_by_cat(cat))
        #         ret[cat] = output[:, :, offset: offset + l]
        #         offset += l

        unit_embeddings = unit_embeddings.align_to('batch', 'unit_emb', 'length')
        with NoName(unit_embeddings):
            output = self.conv(unit_embeddings).rename('batch', 'unit_conv_repr', 'length')
        output = output.align_to(..., 'unit_conv_repr')

        ret = dict()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                predictor = self.p_predictors[cat.name]
                label = predictor(output)  # .log_softmax(dim=-1).exp()  # DEBUG(j_luo)
                ret[cat] = label
        return ret


class ExtractModel(nn.Module):

    add_argument('max_num_words', default=3, dtype=int, msg='Max number of extracted words.')
    add_argument('max_word_length', default=10, dtype=int, msg='Max length of extracted words.')
    add_argument('max_extracted_candidates', default=200, dtype=int, msg='Max number of extracted candidates.')
    add_argument('init_threshold', default=0.05, dtype=float,
                 msg='Initial value of threshold to determine whether two words are matched.')
    add_argument('thresh_func', default='soft', dtype=str,
                 choices=['soft', 'linear', 'exp', 'exp_linear'], msg='Threshold function to use.')
    add_argument('use_adapt', default=False, dtype=bool, msg='Flag to use adapter layer.')
    add_argument('use_embedding', default=True, dtype=bool, msg='Flag to use embedding.')
    add_argument('dist_func', default='hamming', dtype=str,
                 choices=['cos', 'hamming', 'sos'], msg='Type of distance function to use')
    add_argument('relaxation_level', default=1, dtype=int, choices=[0, 1, 2, 3, 4], msg='Level of relaxation.')
    add_argument('temperature', default=0.1, dtype=float, msg='Temperature.')
    add_argument('ins_del_cost', default=3.5, dtype=float, msg='Unit cost for insertions and deletions.')
    add_argument('debug', dtype=bool, default=False, msg='Flag to enter debug mode.')  # DEBUG(j_luo) debug mode

    def __init__(self, unit_vocab_size: Optional[int] = None, dataset=None):
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

        if g.input_format == 'text':
            self.g2p = G2PLayer(unit_vocab_size, dataset)

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

    @global_property
    def threshold(self):
        pass

    # ------------------------ Useful methods for debugging ----------------------- #

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

    def forward(self, batch: ExtractBatch) -> ExtractModelReturn:
        """
        There are four ways of relaxing the original hard objective.
        1. max_w |w| thresh(min_v d(v, w))
        2. max_w |w| max_v thresh(d(v, w))
        3. max_w max_v |w| thresh(d(v, w))
        4. max_{w, v} thresh(d(v, w))

        Terminologies:
        ed_dist: d(v, w)
        thresh: after thresholding
        matched_: the prefix after selecting v
        score: after multiplication with |w|
        best_: the prefix after selecting w
        """
        self.get_char_hamming('a', 'a')
        if g.debug:
            torch.set_printoptions(sci_mode=False, linewidth=200)
            self.eval()
            breakpoint()

        # Prepare representations.
        # If input_format is 'text', then we need to use g2p to induce ipa first.
        if g.input_format == 'text':
            dfm = self.g2p(batch.unit_id_seqs)
        else:
            dfm = batch.dense_feat_matrix

        if g.dense_input:
            # DEBUG(j_luo)
            # if g.input_format == 'ipa':
            #     with Rename(*self.unit_dense_feat_matrix.values(), unit='batch'):
            #         adapted_dfm = self.adapter(dfm)
            # else:
            #     adapted_dfm = dfm

            if g.use_embedding:
                word_repr = self.embedding(adapted_dfm, batch.source_padding)
                unit_repr = self.embedding(self.unit_dense_feat_matrix)
            else:
                # names = sorted(adapted_dfm, key=lambda name: name.value)
                names = sorted([cat for cat in Category if should_include(
                    g.feat_groups, cat)], key=lambda name: name.value)
                # IDEA(j_luo) NoName shouldn't use reveal_name. Just keep the name in the context manager.
                # with NoName(*self.unit_dense_feat_matrix.values(), *adapted_dfm.values()):
                with NoName(*self.unit_dense_feat_matrix.values()):
                    # word_repr = torch.cat([adapted_dfm[name] for name in names], dim=-1)
                    unit_repr = torch.cat([self.unit_dense_feat_matrix[name] for name in names], dim=-1)
                # DEBUG(j_luo)
                word_repr = dfm.rename('batch', 'length', 'char_emb')
                # word_repr.rename_('batch', 'length', 'char_emb')
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
        matches = new_extracted.matches
        len_s = matches.ed_dist.size('len_s')
        len_e = matches.ed_dist.size('len_e')
        vs = len(self.vocab)

        # Get the best score and span.
        # Only lv4 needs special treatment.
        if self.training and g.relaxation_level == 4:
            flat_score = matches.score.flatten(['len_s', 'len_e', 'vocab'], 'cand')
            best_matched_score, best_span_ind = _soft_max(flat_score, 'cand')
            start = best_span_ind // (len_e * vs)
            # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
            end = best_span_ind % (len_e * vs) // vs + start + g.min_word_length - 1
            best_matched_vocab = best_span_ind % vs
        else:
            flat_matched_score = matches.matched_score.flatten(['len_s', 'len_e'], 'cand')
            if self.training and g.relaxation_level in [1, 2, 3]:
                best_matched_score, best_span_ind = _soft_max(flat_matched_score, 'cand')
            else:
                best_matched_score, best_span_ind = flat_matched_score.max(dim='cand')
            start = best_span_ind // len_e
            # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
            end = best_span_ind % len_e + start + g.min_word_length - 1

            flat_matched_vocab = matches.matched_vocab.flatten(['len_s', 'len_e'], 'cand')
            best_matched_vocab = flat_matched_vocab.gather('cand', best_span_ind)

        ret = ExtractModelReturn(start, end, best_matched_score, best_matched_vocab, new_extracted, dfm)
        # ret = ExtractModelReturn(start, end, best_matched_score, best_matched_vocab, new_extracted, adapted_dfm)

        return ret

    def _extract_one_span(self, batch: ExtractBatch, extracted: Extracted, word_repr: FT, unit_repr: FT) -> Extracted:
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
        field_names = [f.name for f in fields(matches)]
        for fname in field_names:
            attr = getattr(matches, fname)
            value = -999.9 if 'score' in fname else None
            restored = _restore_shape(attr, bi, lsi, lei, viable, value=value)
            setattr(matches, fname, restored)

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

        def _get_sos_matrix(x, y):
            with NoName(x, y):
                sos = ((x.unsqueeze(dim=1) - y) ** 2).sum(dim=-1)
            return sos.rename_(x.names[0], y.names[0]) / 4

        nh = NameHelper()
        _extracted_word_repr = nh.flatten(extracted_word_repr, ['viable', 'len_w'], 'viable_X_len_w')
        if g.dist_func == 'cos':
            dist_func = _get_cosine_matrix
        elif g.dist_func == 'hamming':
            dist_func = _get_hamming_matrix
        else:
            dist_func = _get_sos_matrix
        costs = dist_func(_extracted_word_repr, unit_repr)

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
                    if (ls - 1, lt) in fs:
                        transitions.append(fs[(ls - 1, lt)] + g.ins_del_cost)
                    if (ls, lt - 1) in fs:
                        transitions.append(fs[(ls, lt - 1)] + g.ins_del_cost)
                    if (ls - 1, lt - 1) in fs:
                        vocab_inds = self.indexed_segments[:, lt - 1]
                        sub_cost = costs[:, ls - 1, vocab_inds]
                        transitions.append(fs[(ls - 1, lt - 1)] + sub_cost)
                    if transitions:
                        all_s = torch.stack(transitions, dim=-1)
                        new_s, _ = all_s.min(dim=-1)
                        fs[(ls, lt)] = new_s

        f_lst = list()
        for i in range(msl + 1):
            for j in range(mtl + 1):
                if (i, j) not in fs:
                    fs[(i, j)] = get_zeros(ns, nt).fill_(9999.9)
                f_lst.append(fs[(i, j)])
        f = torch.stack(f_lst, dim=0).view(msl + 1, mtl + 1, -1, len(self.vocab))
        f.rename_('len_w_src', 'len_w_tgt', 'viable', 'vocab')
        # ls_idx, lt_idx = zip(*fs.keys())

        # Get the values wanted.
        if g.debug:
            breakpoint()  # DEBUG(j_luo) debug mode
        with NoName(f, viable_lens, self.vocab_length):
            idx_src = viable_lens.unsqueeze(dim=-1)
            idx_tgt = self.vocab_length
            viable_i = get_range(ns, 2, 0)
            vocab_i = get_range(len(self.vocab_length), 2, 1)

            ed_dist = f[idx_src, idx_tgt, viable_i, vocab_i]
            ed_dist.rename_('viable', 'vocab')

        # Get the best spans.
        if g.thresh_func == 'soft':
            thresh_func = _soft_threshold
        elif g.thresh_func == 'linear':
            thresh_func = _linear_threshold
        elif g.thresh_func == 'exp':
            thresh_func = _exp_threshold
        else:
            thresh_func = _exp_linear_threshold
        if self.training and g.relaxation_level > 0:
            if g.relaxation_level == 1:
                matched_ed_dist, matched_vocab = _soft_min(ed_dist, 'vocab')
                matched_length = self.vocab_length.gather('vocab', matched_vocab)
                matched_thresh = thresh_func(matched_ed_dist, self.threshold)
                matched_score = matched_length * matched_thresh
                matches = MatchesLv1(ed_dist, matched_ed_dist, matched_vocab,
                                     matched_length, matched_thresh, matched_score)
            elif g.relaxation_level == 2:
                thresh = thresh_func(ed_dist, self.threshold)
                matched_thresh, matched_vocab = _soft_max(thresh, 'vocab')
                matched_length = self.vocab_length.gather('vocab', matched_vocab)
                matched_score = matched_length * matched_thresh
                matches = MatchesLv2(ed_dist, thresh, matched_thresh, matched_vocab, matched_length, matched_score)
            elif g.relaxation_level == 3:
                thresh = thresh_func(ed_dist, self.threshold)
                score = self.vocab_length * thresh
                matched_score, matched_vocab = _soft_max(score, 'vocab')
                matches = MatchesLv3(ed_dist, thresh, score, matched_score, matched_vocab)
            else:
                thresh = thresh_func(ed_dist, self.threshold)
                score = self.vocab_length * thresh
                matches = MatchesLv4(ed_dist, thresh, score)
        else:
            matched_ed_dist, matched_vocab = ed_dist.min(dim='vocab')
            matched_length = self.vocab_length.gather('vocab', matched_vocab)
            matched_thresh = thresh_func(matched_ed_dist, self.threshold)
            matched_score = matched_length * matched_thresh
            matches = MatchesLv0(ed_dist, matched_ed_dist, matched_vocab,
                                 matched_length, matched_thresh, matched_score)

        return matches