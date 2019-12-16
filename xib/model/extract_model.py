import logging
import math
from dataclasses import fields
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import (BaseBatch, batch_class, get_array,
                             get_length_mask, get_range)
from dev_misc.devlib.named_tensor import (NameHelper, NoName, Rename,
                                          drop_names, get_named_range)
from dev_misc.utils import WithholdKeys, global_property
from xib.data_loader import (ContinuousIpaBatch, UnbrokenTextBatch,
                             convert_to_dense)
from xib.ipa import Category, Index, get_enum_by_cat, should_include
from xib.ipa.process import Segment, Segmentation, SegmentWindow, Span
from xib.model.modules import AdaptLayer, FeatEmbedding

from .modules import DenseFeatEmbedding

ExtractBatch = Union[ContinuousIpaBatch, UnbrokenTextBatch]


@batch_class
class Matches(BaseBatch):
    ed_dist: FT
    f: FT  # All these dp scores.


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
    viable: Optional[BT] = None
    costs: Optional[FT] = None
    inverse_unit_costs: Optional[FT] = None
    len_candidates: Optional[LT] = None
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
    alignment: FT


def _soft_threshold(x, thresh):
    return (nn.functional.celu(1 - 2 * x / thresh) + 1) / 2


def _linear_threshold(x, thresh):
    return 1 - x / thresh


def _exp_linear_threshold(x, thresh):
    pt = - thresh * math.log(thresh)
    return torch.where(x > pt, -x / thresh, (-x / thresh).exp())


def _exp_threshold(x, thresh):
    return (-x / thresh).exp()


def _soft_max(x, dim, temperature):
    w = (x / temperature).log_softmax(dim=dim).exp()
    value = (x * w).sum(dim)
    _, index = x.max(dim=dim)
    return value, index


def _soft_min(x, dim, temperature):
    w = (-x / temperature).log_softmax(dim=dim).exp()
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


LU_SIZE = 33
KU_SIZE = 29
# KU_SIZE = 28
# LU_SIZE = 3
# KU_SIZE = 4


class G2PLayer(nn.Module):

    add_argument('g2p_window_size', default=3, dtype=int, msg='Window size for g2p layer.')
    # DEBUG(j_luo)
    add_argument('oracle', default=False, dtype=bool)

    def __init__(self, unit_vocab_size: int, dataset):  # FIXME(j_luo) remove dataset
        super().__init__()
        # DEBUG(j_luo)
        self.dataset = dataset

        # # DEBUG(j_luo)
        if g.oracle:
            logging.warn('Hacking it right now.')
            units = torch.cat([Segment(u).feat_matrix for u in dataset.id2unit], dim=0)
            xx = list()
            for unit in units:
                x = list()
                for u, cat in zip(unit[:7], Category):
                    if cat.name[0] in ['P', 'C', 'V']:
                        e = get_enum_by_cat(cat)
                        f_idx = Index.get_feature(int(u)).value.f_idx
                        x.extend([0] * (f_idx) + [1] + [0] * (len(e) - f_idx - 1))
                assert len(x) == 60
                xx.append(x)
            units = torch.LongTensor(xx)
            self.unit_embedding = nn.Embedding(unit_vocab_size, 60)  # Vg.dim)
            self.unit_embedding.weight.data.copy_(units * 5.0)
            # self.unit_embedding = nn.Embedding(unit_vocab_size, 60)

            # # DEBUG(j_luo)
            # noise = torch.randn_like(self.unit_embedding.weight) * 0.1
            # self.unit_embedding.weight.data.copy_(0.5 + noise)

            # DEBUG(j_luo)
            # self.unit_embedding = nn.Embedding(unit_vocab_size, g.dim)
            # self.conv = nn.Conv1d(g.dim, g.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)

            # DEBUG(j_luo)
        else:
            self.unit_embedding = nn.Embedding(unit_vocab_size, g.dim)
            nn.init.uniform_(self.unit_embedding.weight, 0.4, 0.6)
        self.conv = nn.Conv1d(g.dim, g.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)

        module_dict = dict()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                e = get_enum_by_cat(cat)
                nf = len(e)
                # DEBUG(j_luo)
                # module_dict[cat.name] = nn.Linear(g.dim, nf)
                module_dict[cat.name] = nn.Linear(g.dim, nf)
        self.p_predictors = nn.ModuleDict(module_dict)

        # DEBUG(j_luo)
        self.aligner = nn.Linear(g.dim, 60)
        # DEBUG(j_luo)
        logging.warning('unit aligner initialized.')
        # self.unit_aligner = nn.Embedding(24, 28)
        self.unit_aligner = nn.Embedding(LU_SIZE, KU_SIZE)
        logging.warning('unit aligner initialized uniformly.')
        torch.nn.init.uniform_(self.unit_aligner.weight, -0.1, 0.1)

    def forward(self, unit_id_seqs: LT) -> Tuple[Dict[Category, FT], FT]:
        unit_embeddings = self.unit_embedding(unit_id_seqs).refine_names(..., 'unit_emb')
        # DEBUG(j_luo)
        aligned_unit_emb = unit_embeddings
        # unit_embeddings = nn.functional.dropout(unit_embeddings, p=g.dropout)
        # aligned_unit_emb = self.aligner(unit_embeddings).rename(..., 'unit_emb')

        # # DEBUG(j_luo)
        # return nn.functional.dropout(unit_embeddings, p=g.dropout)

        # DEBUG(j_luo)
        # return unit_embeddings

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

        # DEBUG(j_luo)
        output = nn.functional.dropout(output, g.dropout)

        ret = dict()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                predictor = self.p_predictors[cat.name]
                label = predictor(output)  # .log_softmax(dim=-1).exp()  # DEBUG(j_luo)
                ret[cat] = label

        return ret, aligned_unit_emb


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
    add_argument('use_probs', default=False, dtype=bool, msg='Flag to use probabilities instead of distances.')
    add_argument('new_use_probs', default=False, dtype=bool)
    add_argument('use_global', default=False, dtype=bool)
    add_argument('use_g_embedding', default=False, dtype=bool)
    add_argument('use_residual', default=False, dtype=bool, msg='Flag to use residual.')
    add_argument('use_plain_embedding', default=False, dtype=bool, msg='Flag to use residual.')
    add_argument('use_direct_almt', default=False, dtype=bool, msg='Flag to use residual.')
    add_argument('use_full_prob', default=False, dtype=bool, msg='Flag to use residual.')
    add_argument('dist_func', default='hamming', dtype=str,
                 choices=['cos', 'hamming', 'sos'], msg='Type of distance function to use')
    add_argument('relaxation_level', default=1, dtype=int, choices=[0, 1, 2, 3, 4], msg='Level of relaxation.')
    add_argument('temperature', default=0.1, dtype=float, msg='Temperature.')
    add_argument('init_ins_del_cost', default=100, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('min_ins_del_cost', default=3.5, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('unextracted_prob', default=0.01, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('multiplier', default=1.0, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('debug', dtype=bool, default=False, msg='Flag to enter debug mode.')  # DEBUG(j_luo) debug mode
    add_argument('uniform_scheme', dtype=str, default='none',
                 choices=['none', 'prior', 'topk'], msg='How to use uniformly-weighted scores.')

    def __init__(self, unit_vocab_size: Optional[int] = None, dataset=None):
        super().__init__()

        if g.use_embedding:
            emb_cls = DenseFeatEmbedding if g.dense_input else FeatEmbedding
            self.embedding = emb_cls('feat_emb', 'chosen_feat_group', 'char_emb')
        # elif not g.dense_input:
        #     raise ValueError(f'Use embedding for sparse inputs.')

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
            if g.use_plain_embedding:
                self.plain_unit_embedding = nn.Embedding(LU_SIZE, g.dim)

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

    # IDEA(j_luo) The current api is worse than just declaring GlobalProperty(writeable=False) outright. And doesn't give proper type hints.
    @global_property
    def threshold(self):
        pass

    @global_property
    def ins_del_cost(self):
        pass

    @global_property
    def temperature(self):
        pass

    @global_property
    def uniform_prior(self):
        pass

    @global_property
    def topk_ratio(self):
        pass

    # ------------------------ Useful methods for debugging ----------------------- #

    def get_g2p_vector(self, c: str):
        idx = self.g2p.dataset.unit2id[c]
        idx = get_zeros(1).fill_(idx).long()
        adapted_dfm = self.g2p(idx.rename('batch'))
        names = sorted(adapted_dfm, key=lambda name: name.value)
        with NoName(*adapted_dfm.values()):
            unit_repr = torch.cat([adapted_dfm[name] for name in names], dim=-1)
        return adapted_dfm, unit_repr

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

    def get_one_hot_unit_repr(self):
        with NoName(*self.unit_dense_feat_matrix.values()):
            unit_repr = torch.cat([self.unit_dense_feat_matrix[name]
                                   for name in Category if should_include(g.feat_groups, name)], dim=-1)
        return unit_repr.rename_('batch', 'length', 'char_emb').squeeze('length')

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
        # DEBUG(j_luo)
        # Prepare representations.
        # If input_format is 'text', then we need to use g2p to induce ipa first.
        # dfm = self.g2p(batch.unit_id_seqs)

        # names = sorted([cat for cat in Category if should_include(g.feat_groups, cat)], key=lambda name: name.value)
        # # IDEA(j_luo) NoName shouldn't use reveal_name. Just keep the name in the context manager.
        # with NoName(*self.unit_dense_feat_matrix.values()):
        #     # word_repr = torch.cat([adapted_dfm[name] for name in names], dim=-1)
        #     unit_repr = torch.cat([self.unit_dense_feat_matrix[name] for name in names], dim=-1)
        # word_repr = dfm.rename('batch', 'length', 'char_emb')
        # unit_repr.rename_('batch', 'length', 'char_emb')

        # DEBUG(j_luo)
        # Prepare representations.
        # If input_format is 'text', then we need to use g2p to induce ipa first.
        if g.input_format == 'text':
            dfm, unit_emb = self.g2p(batch.unit_id_seqs)
        else:
            dfm = batch.dense_feat_matrix

        alignment = None

        if g.dense_input:
            if g.input_format == 'ipa':
                with Rename(*self.unit_dense_feat_matrix.values(), unit='batch'):
                    adapted_dfm = self.adapter(dfm)
            else:
                adapted_dfm = dfm
            if g.use_embedding:
                with Rename(*self.unit_dense_feat_matrix.values(), unit='batch'):
                    unit_repr = self.embedding(self.unit_dense_feat_matrix)
                # DEBUG(j_luo)
                if g.use_residual:
                    # # DEBUG(j_luo)
                    # word_repr = self.g2p.unit_aligner(get_range(33, 1, 0)).log_softmax(dim=0).exp()
                    word_repr = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0)).log_softmax(dim=0).exp()
                    # word_repr = self.g2p.unit_aligner(get_range(33, 1, 0)).log_softmax(dim=0).exp()
                    # word_repr = self.g2p.unit_aligner(get_range(24, 1, 0)).log_softmax(dim=0).exp() * 10.0
                    word_repr = word_repr @ unit_repr.squeeze(1)
                    word_repr = (word_repr / word_repr.sum(dim=-1, keepdim=True)) * 7.0 * 10.0
                    with NoName(batch.unit_id_seqs, word_repr):
                        word_repr = word_repr[batch.unit_id_seqs]
                    word_repr.rename_('batch', 'length', 'char_emb')
                else:
                    word_repr = self.embedding(adapted_dfm, batch.source_padding)
            else:
                names = sorted(adapted_dfm, key=lambda name: name.value)
                # IDEA(j_luo) NoName shouldn't use reveal_name. Just keep the name in the context manager.
                with NoName(*self.unit_dense_feat_matrix.values(), *adapted_dfm.values()):
                    word_repr = torch.cat([adapted_dfm[name] for name in names], dim=-1)
                    unit_repr = torch.cat([self.unit_dense_feat_matrix[name] for name in names], dim=-1)
                word_repr.rename_('batch', 'length', 'char_emb')
                # DEBUG(j_luo)
                if g.use_residual:
                    # # DEBUG(j_luo)
                    # word_repr = self.g2p.unit_aligner(get_range(33, 1, 0)).log_softmax(dim=0).exp()
                    # # word_repr = self.g2p.unit_aligner(get_range(24, 1, 0)).log_softmax(dim=0).exp() * 10.0
                    # word_repr = word_repr @ unit_repr.squeeze(1)
                    # word_repr = (word_repr / word_repr.sum(dim=-1, keepdim=True)) * 7.0 * 10.0
                    # with NoName(batch.unit_id_seqs, word_repr):
                    #     word_repr = word_repr[batch.unit_id_seqs]
                    # word_repr.rename_('batch', 'length', 'char_emb')

                    if g.use_global:
                        if g.use_g_embedding:
                            word_repr = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0)) @ unit_repr.squeeze(dim=1)
                            raw_logits = (word_repr @ unit_repr.squeeze(dim=1).t()).reshape(-1)  # / 5.0
                        else:
                            raw_logits = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0)).reshape(-1)
                        with NoName(raw_logits):
                            self.global_log_probs = raw_logits.log_softmax(
                                0).view(LU_SIZE, -1).rename('lost_unit', 'unit')

                    if g.use_direct_almt:
                        word_repr = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0))  # .log_softmax(dim=0).exp() * 10.0
                        # DEBUG(j_luo)
                        # alignment = word_repr.log_softmax(dim=-1).exp()
                        # word_repr = alignment @ unit_repr.squeeze(dim=1)
                        # word_repr = word_repr * g.multiplier
                        word_repr = word_repr @ unit_repr.squeeze(dim=1)
                        self.global_log_probs = (word_repr @ unit_repr.squeeze(dim=1).t()).log_softmax(dim=-1)
                        alignment = self.global_log_probs.exp()
                    else:
                        # # # DEBUG(j_luo)
                        word_repr = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0))  # .log_softmax(dim=0).exp() * 10.0
                        # word_repr = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0)).log_softmax(dim=0).exp()
                        # word_repr = self.g2p.unit_aligner(get_range(24, 1, 0)).log_softmax(dim=0).exp() * 10.0
                        word_repr = word_repr @ unit_repr.squeeze(1)
                        word_repr = 10 * word_repr
                    with NoName(batch.unit_id_seqs):
                        word_repr = word_repr[batch.unit_id_seqs]
                    word_repr.rename_('batch', 'length', 'char_emb')

                    # DEBUG(j_luo)
                    # word_repr = 0.0 * word_repr + unit_emb.rename(unit_emb='char_emb')

                unit_repr.rename_('batch', 'length', 'char_emb')
        else:
            if g.input_format == 'text':
                assert g.use_plain_embedding
                with NoName(batch.feat_matrix, batch.source_padding):
                    # IDEA(j_luo) A global switch to turn off the following names?
                    word_repr = self.plain_unit_embedding(batch.unit_id_seqs).rename(None)
                    word_repr[batch.source_padding] = 0.0
                    word_repr.rename_('batch', 'length', 'char_emb')
                    unit_repr = self.g2p.unit_embedding(get_range(KU_SIZE, 1, 0)).unsqueeze(dim=1)
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

        # # DEBUG(j_luo)
        # inverse_new_extracted = self._extract_one_span(batch, extracted, word_repr, unit_repr, inverse=True)
        # inverse_matches = inverse_new_extracted.matches

        # Get the best score and span.
        if g.use_probs or (self.training and g.relaxation_level == 4):  # Only lv4 needs special treatment.
            flat_score = matches.score.flatten(['len_s', 'len_e', 'vocab'], 'cand')
            # # NOTE(j_luo) Use min if g.use_probs is True.
            # _soft_func = _soft_min if g.use_probs else _soft_max
            best_matched_score, best_span_ind = _soft_max(flat_score, 'cand', self.temperature)
            start = best_span_ind // (len_e * vs)
            # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
            end = best_span_ind % (len_e * vs) // vs + start + g.min_word_length - 1
            best_matched_vocab = best_span_ind % vs

            # NOTE(j_luo) Some segments don't have any viable spans.
            any_viable = new_extracted.viable.any('len_s').any('len_e')
            # DEBUG(j_luo)
            # best_matched_score = best_matched_score * any_viable
            best_matched_score = best_matched_score + (~any_viable).float() * (-9999.9)
            # DEBUG(j_luo) This was actually the correct one.
            # best_matched_score = best_matched_score.logsumexp('batch').expand_as(best_matched_score)
            if g.new_use_probs:
                flat_viable = new_extracted.viable.expand_as(matches.score).flatten(['len_s', 'len_e', 'vocab'], 'cand')
                flat_viable_score = (~flat_viable) * (-9999.9) + flat_score
                if g.use_full_prob:
                    unextracted = batch.lengths.align_as(new_extracted.len_candidates) - new_extracted.len_candidates
                    unextracted = unextracted.expand_as(matches.score)
                    flat_unextracted_score = unextracted.flatten(
                        ['len_s', 'len_e', 'vocab'], 'cand') * math.log(g.unextracted_prob)
                    flat_viable_score = flat_viable_score + flat_unextracted_score
                    best_matched_score = flat_viable_score.logsumexp(dim='cand')
                    best_matched_score = best_matched_score * (any_viable).float()
                else:
                    best_matched_score = flat_viable_score.logsumexp(dim='cand')
                    best_matched_score = best_matched_score.logsumexp('batch').expand_as(best_matched_score)
                    best_matched_score = best_matched_score * any_viable
                ret = ExtractModelReturn(start, end, best_matched_score,
                                         best_matched_vocab, new_extracted, dfm, alignment)
            elif g.use_probs:
                flat_viable = new_extracted.viable.expand_as(matches.score).flatten(['len_s', 'len_e', 'vocab'], 'cand')

                # nh = NameHelper()
                # x = nh.flatten(matches.score, ['len_s', 'len_e'], 'len_s_X_len_e')
                # TOP = 100
                # _, inds = torch.topk(x.logsumexp('vocab'), TOP, 'len_s_X_len_e')
                # top_start = inds // len_e
                # top_end = inds % len_e + top_start + g.min_word_length - 1
                # top = torch.stack([top_start, top_end], new_name='index').cpu().numpy()

                # predictions = list()
                # for bi in range(batch.batch_size):
                #     segments = list()
                #     for k in range(TOP):
                #         s = top[bi, k, 0]
                #         e = top[bi, k, 1]
                #         span = Span('dummy', s, e)
                #         segment = Segmentation([span])
                #         segments.append(segment)
                #     predictions.append(segments)

                # ground_truths = [segment.to_segmentation() for segment in batch.segments]
                # recalled = list()
                # for preds, gt in zip(predictions, ground_truths):
                #     _recalled = np.zeros(TOP)
                #     for k, pred in enumerate(preds):
                #         p = pred.spans[0]
                #         for _g in gt:
                #             if p.is_same_span(_g):
                #                 _recalled[k] = 1
                #                 break
                #     _recalled = np.cumsum(_recalled)
                #     recalled.append(_recalled)
                # recalled = np.asarray(recalled).sum(axis=0)
                # try:
                #     self._cnt += 1
                #     self._recalled = self._recalled + recalled
                #     self._total += batch.batch_size
                # except AttributeError:
                #     self._cnt = 1
                #     self._recalled = np.zeros(TOP)
                #     self._total = batch.batch_size
                # if self._cnt % 100 == 0:
                #     print((self._recalled / self._total)[range(19, TOP, 20)])
                #     self._recalled = np.zeros(TOP)
                #     self._total = 0

                # flat_viable_score = flat_score + (-99999.9) * (~flat_viable).float()
                # viable_count = flat_viable.sum('cand').float()

                # combined_logits = (1.0 / viable_count + 1e-8).log().align_as(flat_score) + flat_score
                # best_matched_score = combined_logits.logsumexp(dim='cand').exp()

                best_matched_score = flat_score.logsumexp(dim='cand').exp()

                # # DEBUG(j_luo)
                # inverse_flat_score = inverse_matches.score.flatten(['len_s', 'len_e', 'vocab'], 'cand')
                # inverse_best_matched_score = inverse_flat_score.logsumexp(dim='cand').exp()

                # best_matched_score = 0.5 * best_matched_score + 0.5 * inverse_best_matched_score

                # best_matched_score = flat_score.logsumexp(dim='cand')
                # DEBUG(j_luo)
                # best_matched_score = flat_score.max(dim='cand')[0].exp()
                ret = ExtractModelReturn(start, end, best_matched_score, best_matched_vocab, new_extracted, dfm)

            # DEBUG(j_luo)
            elif g.uniform_scheme in ['prior', 'topk']:
                flat_viable = new_extracted.viable.expand_as(matches.score).flatten(['len_s', 'len_e', 'vocab'], 'cand')
                if g.uniform_scheme == 'prior':
                    flat_viable = flat_viable & (flat_score > -50)  # DEBUG(j_luo)
                    summed_score = (flat_viable * flat_score).sum(dim='cand')
                    summed_weight = flat_viable.sum(dim='cand')
                    uniform_matched_score = summed_score / summed_weight
                    best_matched_score = self.uniform_prior * uniform_matched_score + \
                        (1.0 - self.uniform_prior) * best_matched_score
                else:
                    # This is the upper boundary of k.
                    k = int(flat_viable.sum('cand').max() * self.topk_ratio)
                    k = max(1, k)
                    # k = 20  # DEBUG(j_luo)
                    # Get the top k candidates.
                    flat_viable_score = flat_score + (-99999.9) * (~flat_viable).float()
                    top_score, top_ind = torch.topk(flat_viable_score, k, 'cand')
                    # Each batch has a different k.
                    batch_k = torch.min(get_tensor([k]).long(), flat_viable.sum('cand'))
                    # Get the actual top indices for each batch
                    batch_k_mask = get_length_mask(batch_k, k)
                    summed_score = (batch_k_mask * top_score).sum('cand')
                    best_matched_score = summed_score / batch_k
        else:
            flat_matched_score = matches.matched_score.flatten(['len_s', 'len_e'], 'cand')
            if self.training and g.relaxation_level in [1, 2, 3]:
                best_matched_score, best_span_ind = _soft_max(flat_matched_score, 'cand', self.temperature)
                # DEBUG(j_luo)
                if g.uniform_scheme in ['prior', 'topk']:
                    flat_viable = new_extracted.viable.flatten(['len_s', 'len_e'], 'cand')
                    if g.uniform_scheme == 'prior':
                        flat_viable = flat_viable & (flat_matched_score > -50)  # DEBUG(j_luo)
                        summed_score = (flat_viable * flat_matched_score).sum(dim='cand')
                        summed_weight = flat_viable.sum(dim='cand')
                        uniform_matched_score = summed_score / summed_weight
                        best_matched_score = self.uniform_prior * uniform_matched_score + \
                            (1.0 - self.uniform_prior) * best_matched_score
                    else:
                        assert False, 'not updated'  # FIXME(j_luo)
                        k = int(flat_viable.size('cand') * self.topk_ratio)
                        k = max(1, k)
                        flat_viable_matched_score = flat_viable * flat_matched_score
                        top_score, top_ind = torch.topk(flat_viable_matched_score, k, 'cand')
                        summed_score = top_score.sum('cand')
                        uniform_matched_score = summed_score / k
            else:
                best_matched_score, best_span_ind = flat_matched_score.max(dim='cand')
            start = best_span_ind // len_e
            # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
            end = best_span_ind % len_e + start + g.min_word_length - 1

            any_viable = new_extracted.viable.any('len_s').any('len_e')
            # DEBUG(j_luo)
            # best_matched_score = best_matched_score * any_viable
            best_matched_score = best_matched_score + (~any_viable).float() * (-9999.9)

            flat_matched_vocab = matches.matched_vocab.flatten(['len_s', 'len_e'], 'cand')
            best_matched_vocab = flat_matched_vocab.gather('cand', best_span_ind)

        if g.debug:
            torch.set_printoptions(sci_mode=False, linewidth=200)
            from dev_misc.devlib.inspector import Inspector
            ins = Inspector()
            id_seq_emb = self.g2p.unit_embedding.weight
            unit_logits = id_seq_emb @ unit_repr.t()
            unit_logits.rename_('lost_unit', 'known_unit')
            unit_probs = unit_logits.log_softmax('known_unit').exp()
            ins.add_table(unit_logits, 'unit_logit')
            ins.add_table(unit_probs, 'unit_prob')

            ins.add_table(matches.ed_dist, 'ed_dist')
            ins.add_table(matches.f, 'f', auto_merge=False)
            ins.add_table(matches.score, 'score')
            ins.add_table(matches.thresh, 'thresh')
            ins.add_table(new_extracted.costs, 'cost')

            lost_units = self.g2p.dataset.id2unit
            ins.add_table(lost_units, 'lost_unit', is_index=True)
            known_units = self.id2unit
            ins.add_table(known_units, 'known_unit', is_index=True)
            vocab = [''.join(segment.segment_list) for segment in self.vocab]
            ins.add_table(vocab, 'vocab', is_index=True)
            segments = [''.join(segment.segment_list) for segment in batch.segments]
            ins.add_table(segments, 'batch', is_index=True)
            ins.add_table(new_extracted.viable, 'viable', is_mask_index=True)

            ins.take('viable')
            ins.take('batch_id', 2)
            ins.merge('f', how='left', left_index=True, right_on='viable_id')
            ins.take('len_s_id', 2)
            ins.take('len_e_id', 0)
            ins.narrow(['f', 'len_w_src_id', 'len_w_tgt_id', 'vocab_id'])
            ins.merge('vocab', how='right', right_index=True, left_on='vocab_id')
            ins.save_as('first')
            ins.take('vocab', 'tɾaeɾas')
            ins.pivot(index='len_w_src_id', columns='len_w_tgt_id', values='f')
            ins.run()

        ret = ExtractModelReturn(start, end, best_matched_score, best_matched_vocab, new_extracted, dfm, alignment)
        # ret = ExtractModelReturn(start, end, best_matched_score, best_matched_vocab, new_extracted, adapted_dfm)

        return ret

    @global_property
    def inverse_ratio(self):
        pass

    def _extract_one_span(self, batch: ExtractBatch, extracted: Extracted, word_repr: FT, unit_repr: FT, inverse: bool = False) -> Extracted:
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
        if g.input_format == 'text':
            with NoName(word_repr, viable_bi, word_pos, batch.unit_id_seqs):
                extracted_word_repr = word_repr[viable_bi, word_pos].rename('viable_X_len_w', 'char_emb')
                extracted_unit_ids = batch.unit_id_seqs[viable_bi, word_pos].rename('viable_X_len_w')
        else:
            with NoName(word_repr, viable_bi, word_pos):
                extracted_word_repr = word_repr[viable_bi, word_pos].rename('viable_X_len_w', 'char_emb')
            extracted_unit_ids = None
        extracted_word_repr = nh.unflatten(extracted_word_repr, 'viable_X_len_w', ['viable', 'len_w'])

        # DEBUG(j_luo)
        if g.input_format == 'text':
            unit_counts = get_zeros(extracted_unit_ids.max().item() + 1).rename('extracted_unit')
            with NoName(extracted_unit_ids, unit_counts):
                unit_counts.scatter_add_(0, extracted_unit_ids, torch.full_like(extracted_unit_ids, 1).float())
        else:
            unit_counts = None

        # Main body: Run DP to find the best matches.
        matches, costs, inverse_unit_costs = self._get_matches(
            extracted_word_repr, unit_repr, viable_lens, extracted_unit_ids, unit_counts, inverse=inverse)
        # Revert to the old shape (so that invalid spans are included).
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        lsi = get_named_range(len_s, 'len_s').expand_as(viable)
        lei = get_named_range(len_e, 'len_e').expand_as(viable)
        field_names = [f.name for f in fields(matches)]
        for fname in field_names:
            # Skip dp scores.
            if fname == 'f':
                continue

            attr = getattr(matches, fname)
            if attr is not None:
                value = None
                if 'score' in fname or 'thresh' in fname:
                    value = -99999.9
                elif 'ed_dist' in fname:
                    value = 99999.9
                    if g.use_probs:
                        value = -value
                # DEBUG(j_luo)
                # NOTE(j_luo) Reverse signs if g.use_probs is True
                # if g.use_probs:
                #     value = -value
                # value = -999.9 if 'score' in fname or 'ed_dist' in fname else None
                restored = _restore_shape(attr, bi, lsi, lei, viable, value=value)
                setattr(matches, fname, restored)

        new_extracted = Extracted(batch.batch_size, matches, viable, costs, inverse_unit_costs, len_candidates)
        return new_extracted

    def _get_matches(self, extracted_word_repr: FT, unit_repr: FT, viable_lens: LT, extracted_unit_ids: LT, unit_counts: FT, inverse: bool = False) -> Tuple[Matches, FT]:
        d_char = extracted_word_repr.size('char_emb')
        ns = extracted_word_repr.size('viable')
        nt = len(self.vocab_feat_matrix)
        msl = extracted_word_repr.size('len_w')
        mtl = self.vocab_feat_matrix.size('length')

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
        if g.use_probs or g.new_use_probs:
            dist_func = lambda x, y: x @ y.t()
            # DEBUG(j_luo) a specialized dist_function

            # def dist_func(x, y):
            #     weight = get_zeros(60).fill_(0.2)
            #     # weight[0:2] = 5.0
            #     weight[5:22] = 1.0
            #     weight[22:43] = 1.0
            #     x = x * weight
            #     return x @ y.t()
            # def dist_func(x, y):
            #     ptype = x[:, :2] * y[:, :2]
            #     c_voicing = x[:, 2: 5] * y[:, 2:5]
            #     c_place = x[:, 5: 22] * y[:, 5, :22]
            #     c_manner = x[:, 22:43] * y[:, 22:43]
            #     v_height = x[:, 43:51] * y[: 43: 51]
            #     v_backness = x[:, 51:57] * y[:, 51:57]
            #     v_roundness = x[: 57:] * y[:, 57:]
            #     ptype.log_softmax(dim=-1)
        elif g.dist_func == 'cos':
            dist_func = _get_cosine_matrix
        elif g.dist_func == 'hamming':
            dist_func = _get_hamming_matrix
        else:
            dist_func = _get_sos_matrix
        # DEBUG(j_luo)
        # from IPython import embed; embed()
        # DEBUG(j_luo)
        costs = dist_func(_extracted_word_repr, unit_repr)  # / 0.2

        # # DEBUG(j_luo) This is wrong
        # partial_counts = torch.zeros_like(unit_counts).fill_(-1e-4).align_to(..., 'unit').expand(-1, costs.size('unit'))
        # with NoName(partial_counts, extracted_unit_ids, costs):
        #     # NOTE(j_luo) There seems to be a bug with the in-place `scatter_add_`.
        #     _partial_counts = partial_counts.scatter_add(
        #         0, extracted_unit_ids.unsqueeze(dim=-1).expand_as(costs), costs)
        # partial_counts = _partial_counts.rename(*partial_counts.names)
        # wi2vi_logits = partial_counts / (1e-8 + unit_counts.align_as(partial_counts))
        # inverse_unit_costs = wi2vi_logits.log_softmax('extracted_unit')
        # with NoName(inverse_unit_costs, extracted_unit_ids):
        #     inverse_costs = inverse_unit_costs[extracted_unit_ids].rename('viable_X_len_w', 'unit')

        ins_del_cost = self.ins_del_cost
        if g.new_use_probs:
            costs = -costs.log_softmax(dim='unit')

        elif g.use_probs:
            costs = costs.log_softmax(dim='unit')
            ins_del_cost = -ins_del_cost

        if g.use_global or g.use_direct_almt:
            with NoName(self.global_log_probs, extracted_unit_ids):
                costs = -self.global_log_probs[extracted_unit_ids].rename('viable_X_len_w', 'unit')

        # DEBUG(j_luo)
        if g.input_format == 'text' and g.use_probs:
            partial_counts = torch.zeros_like(unit_counts).fill_(-1e-4).align_to(...,
                                                                                 'unit').expand(-1, costs.size('unit'))
            with NoName(partial_counts, extracted_unit_ids, costs):
                # NOTE(j_luo) There seems to be a bug with the in-place `scatter_add_`.
                _partial_counts = partial_counts.scatter_add(
                    0, extracted_unit_ids.unsqueeze(dim=-1).expand_as(costs), costs)
            partial_counts = _partial_counts.rename(*partial_counts.names)
            wi2vi_logits = partial_counts / (1e-8 + unit_counts.align_as(partial_counts))
            log_p_v_g_w = wi2vi_logits.log_softmax('unit')
            log_p_w = (unit_counts / (1e-8 + unit_counts.sum()) + 1e-8).log().align_as(log_p_v_g_w)
            log_p_v = (log_p_w + log_p_v_g_w).logsumexp('extracted_unit', keepdim=True)
            inverse_unit_costs = log_p_w_g_v = log_p_v_g_w + log_p_w - log_p_v + (-9999.9) * (wi2vi_logits < -9999)
            with NoName(inverse_unit_costs, extracted_unit_ids):
                inverse_costs = inverse_unit_costs[extracted_unit_ids].rename('viable_X_len_w', 'unit')
        else:
            inverse_unit_costs = None
        # if inverse:
        #     costs = inverse_costs

        # # # DEBUG(j_luo)
        # # costs = (1.0 - self.inverse_ratio) * costs + self.inverse_ratio * inverse_costs

        # # DEBUG(j_luo)
        # _tmp = 0.0
        # costs = (1.0 - _tmp) * costs + _tmp * inverse_costs

        # Name: viable x len_w x unit
        costs = nh.unflatten(costs, 'viable_X_len_w', ['viable', 'len_w'])

        # # NOTE(j_luo) Use dictionary save every state.
        fs = dict()
        for i in range(msl + 1):
            fs[(i, 0)] = get_zeros(ns, nt).fill_(i * ins_del_cost)
        for j in range(mtl + 1):
            fs[(0, j)] = get_zeros(ns, nt).fill_(j * ins_del_cost)
        # # DEBUG(j_luo)
        # fs[(0, 1)] = get_zeros(ns, nt)
        # fs[(1, 0)] = get_zeros(ns, nt)

        # ------------------------ Main body: DP ----------------------- #

        # Transition.
        with NoName(self.indexed_segments, costs):
            for ls in range(1, msl + 1):
                min_lt = max(ls - 2, 1)
                max_lt = min(ls + 2, mtl + 1)
                for lt in range(min_lt, max_lt):
                    transitions = list()
                    if (ls - 1, lt) in fs:
                        transitions.append(fs[(ls - 1, lt)] + ins_del_cost)
                    if (ls, lt - 1) in fs:
                        transitions.append(fs[(ls, lt - 1)] + ins_del_cost)
                    if (ls - 1, lt - 1) in fs:
                        vocab_inds = self.indexed_segments[:, lt - 1]
                        sub_cost = costs[:, ls - 1, vocab_inds]
                        transitions.append(fs[(ls - 1, lt - 1)] + sub_cost)
                    if transitions:
                        all_s = torch.stack(transitions, dim=-1)
                        if g.use_probs:
                            new_s, _ = all_s.max(dim=-1)
                        else:
                            new_s, _ = all_s.min(dim=-1)
                        fs[(ls, lt)] = new_s

        f_lst = list()
        value = -9999.9 if g.use_probs else 9999.9
        # value = 9999.9
        for i in range(msl + 1):
            for j in range(mtl + 1):
                if (i, j) not in fs:
                    fs[(i, j)] = get_zeros(ns, nt).fill_(value)
                f_lst.append(fs[(i, j)])
        f = torch.stack(f_lst, dim=0).view(msl + 1, mtl + 1, -1, len(self.vocab))
        f.rename_('len_w_src', 'len_w_tgt', 'viable', 'vocab')
        # ls_idx, lt_idx = zip(*fs.keys())

        # Get the values wanted.
        with NoName(f, viable_lens, self.vocab_length):
            idx_src = viable_lens.unsqueeze(dim=-1)
            idx_tgt = self.vocab_length
            viable_i = get_range(ns, 2, 0)
            vocab_i = get_range(len(self.vocab_length), 2, 1)

            ed_dist = f[idx_src, idx_tgt, viable_i, vocab_i]

            # # DEBUG(j_luo)
            # min_len = torch.min(idx_src, self.vocab_length)
            # ed_dist = ed_dist / min_len * 7

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
        # DEBUG(j_luo)
        if g.new_use_probs and self.training:
            thresh = None
            if g.use_full_prob:
                score = -ed_dist
            else:
                score = self.vocab_length.float().log() - ed_dist
            matches = MatchesLv4(ed_dist, f, thresh, score)
        elif g.use_probs:
            # DEBUG(j_luo)
            # thresh = (ed_dist / 5.0).exp()
            thresh = None
            # logging.warning('not weighted by length.')
            # score = self.vocab_length.float().log() + ed_dist
            # score = ed_dist

            score = self.vocab_length.float().log() + ed_dist
            # score = ed_dist
            matches = MatchesLv4(ed_dist, f, thresh, score)

        elif self.training:
            if g.relaxation_level == 1:
                matched_ed_dist, matched_vocab = _soft_min(ed_dist, 'vocab', self.temperature)
                matched_length = self.vocab_length.gather('vocab', matched_vocab)
                matched_thresh = thresh_func(matched_ed_dist, self.threshold)
                matched_score = matched_length * matched_thresh
                matches = MatchesLv1(ed_dist, f, matched_ed_dist, matched_vocab,
                                     matched_length, matched_thresh, matched_score)
            elif g.relaxation_level == 2:
                thresh = thresh_func(ed_dist, self.threshold)
                matched_thresh, matched_vocab = _soft_max(thresh, 'vocab', self.temperature)
                matched_length = self.vocab_length.gather('vocab', matched_vocab)
                matched_score = matched_length * matched_thresh
                matches = MatchesLv2(ed_dist, f, thresh, matched_thresh, matched_vocab, matched_length, matched_score)
            elif g.relaxation_level == 3:
                thresh = thresh_func(ed_dist, self.threshold)
                score = self.vocab_length * thresh
                matched_score, matched_vocab = _soft_max(score, 'vocab', self.temperature)
                matches = MatchesLv3(ed_dist, f, thresh, score, matched_score, matched_vocab)
            else:
                thresh = thresh_func(ed_dist, self.threshold)
                score = self.vocab_length * thresh
                matches = MatchesLv4(ed_dist, f, thresh, score)
        else:
            matched_ed_dist, matched_vocab = ed_dist.min(dim='vocab')
            matched_length = self.vocab_length.gather('vocab', matched_vocab)
            matched_thresh = thresh_func(matched_ed_dist, self.threshold)
            matched_score = matched_length * matched_thresh
            matches = MatchesLv0(ed_dist, f, matched_ed_dist, matched_vocab,
                                 matched_length, matched_thresh, matched_score)

        return matches, costs, inverse_unit_costs
