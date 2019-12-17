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
    nll: FT
    f: FT  # All these dp scores.


@batch_class
class Extracted(BaseBatch):
    batch_size: int
    matches: Optional[Matches] = None
    viable: Optional[BT] = None
    costs: Optional[FT] = None
    inverse_unit_costs: Optional[FT] = None
    len_candidates: Optional[LT] = None


@batch_class
class ExtractModelReturn(BaseBatch):
    start: LT
    end: LT
    best_matched_score: FT
    best_matched_vocab: LT
    extracted: Extracted
    adapted_dfm: Dict[Category, FT]
    alignment: FT


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
        self.unit_aligner = nn.Embedding(LU_SIZE, KU_SIZE)
        logging.warning('Unit aligner initialized to 0.')
        self.unit_aligner.weight.data.fill_(0.0)

    def forward(self, unit_id_seqs: LT) -> Tuple[Dict[Category, FT], FT]:
        unit_embeddings = self.unit_embedding(unit_id_seqs).refine_names(..., 'unit_emb')
        # DEBUG(j_luo)
        aligned_unit_emb = unit_embeddings
        # FIXME(j_luo) dropout

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
    add_argument('use_adapt', default=False, dtype=bool, msg='Flag to use adapter layer.')
    add_argument('use_g_embedding', default=False, dtype=bool)
    add_argument('init_ins_del_cost', default=100, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('min_ins_del_cost', default=3.5, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('unextracted_prob', default=0.01, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('multiplier', default=1.0, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('debug', dtype=bool, default=False, msg='Flag to enter debug mode.')
    add_argument('uniform_scheme', dtype=str, default='none',
                 choices=['none', 'prior', 'topk'], msg='How to use uniformly-weighted scores.')

    def __init__(self, unit_vocab_size: Optional[int] = None, dataset=None):
        super().__init__()

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

    # IDEA(j_luo) The current api is worse than just declaring GlobalProperty(writeable=False) outright. And doesn't give proper type hints.
    @global_property
    def threshold(self):
        pass

    @global_property
    def ins_del_cost(self):
        pass

    def forward(self, batch: ExtractBatch) -> ExtractModelReturn:
        """
        # FIXME(j_luo) Probably need to remove this.
        The generating story is:
            v
            |
            w
            |
            x -- ww -- theta

        Pr(x) = sum_w Pr(w) Pr(ww)
              = sum_w Pr(w) theta^|ww|
              = sum_{w, v} Pr(w | v) Pr(v) theta^|ww|

        Terminologies:
        thresh: after thresholding
        matched_: the prefix after selecting v
        score: after multiplication with |w|
        best_: the prefix after selecting w
        """
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

            names = sorted(adapted_dfm, key=lambda name: name.value)
            # IDEA(j_luo) NoName shouldn't use reveal_name. Just keep the name in the context manager.
            with NoName(*self.unit_dense_feat_matrix.values(), *adapted_dfm.values()):
                word_repr = torch.cat([adapted_dfm[name] for name in names], dim=-1)
                unit_repr = torch.cat([self.unit_dense_feat_matrix[name] for name in names], dim=-1)
            word_repr.rename_('batch', 'length', 'char_emb')

            word_repr = self.g2p.unit_aligner(get_range(LU_SIZE, 1, 0))  # .log_softmax(dim=0).exp() * 10.0
            word_repr = word_repr @ unit_repr.squeeze(dim=1)
            self.global_log_probs = (word_repr @ unit_repr.squeeze(dim=1).t()).log_softmax(dim=-1)
            alignment = self.global_log_probs.exp()

            with NoName(batch.unit_id_seqs):
                word_repr = word_repr[batch.unit_id_seqs]
            word_repr.rename_('batch', 'length', 'char_emb')

            # DEBUG(j_luo)
            # word_repr = 0.0 * word_repr + unit_emb.rename(unit_emb='char_emb')
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
        len_s = matches.nll.size('len_s')
        len_e = matches.nll.size('len_e')
        vs = len(self.vocab)

        # Get the best score and span.
        # NOTE(j_luo) Some segments don't have any viable spans.
        flat_score = matches.nll.flatten(['len_s', 'len_e', 'vocab'], 'cand')
        flat_viable = new_extracted.viable.expand_as(matches.nll).flatten(['len_s', 'len_e', 'vocab'], 'cand')
        flat_viable_score = (~flat_viable) * (-9999.9) + flat_score
        # Add probs for unextracted characters.
        unextracted = batch.lengths.align_as(new_extracted.len_candidates) - new_extracted.len_candidates
        unextracted = unextracted.expand_as(matches.nll)
        flat_unextracted = unextracted.flatten(['len_s', 'len_e', 'vocab'], 'cand')
        flat_unextracted_score = flat_unextracted * math.log(g.unextracted_prob)
        flat_total_score = flat_viable_score + flat_unextracted_score
        # Get the top candiates based on total scores.
        best_matched_score, best_span_ind = flat_total_score.max(dim='cand')
        start = best_span_ind // (len_e * vs)
        # NOTE(j_luo) Don't forget the length is off by g.min_word_length - 1.
        end = best_span_ind % (len_e * vs) // vs + start + g.min_word_length - 1
        best_matched_vocab = best_span_ind % vs

        if self.training:
            any_viable = new_extracted.viable.any('len_s').any('len_e')
            best_matched_score = flat_total_score.logsumexp(dim='cand')
            best_matched_score = best_matched_score * any_viable

        ret = ExtractModelReturn(start, end, best_matched_score,
                                 best_matched_vocab, new_extracted, dfm, alignment)

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
        # NOTE(j_luo) Use `viable` to get the lengths. `len_candidates` has dummy axes.
        # IDEA(j_luo) Any better way of handling this? Perhaps persistent names?
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

        # Main body: Run DP to find the best matches.
        matches, costs, inverse_unit_costs = self._get_matches(extracted_word_repr, viable_lens, extracted_unit_ids)
        # Revert to the old shape (so that invalid spans are included).
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        lsi = get_named_range(len_s, 'len_s').expand_as(viable)
        lei = get_named_range(len_e, 'len_e').expand_as(viable)
        vs = matches.nll.size('vocab')
        # IDEA(j_luo) NoName shouldn't make size() calls unavaiable. Otherwise size() calls have to be moved outside the context.
        with NoName(bi, lsi, lei, viable, matches.nll):
            v_bi = bi[viable]
            v_lsi = lsi[viable]
            v_lei = lei[viable]
            all_nll = get_zeros(batch.batch_size, len_s, len_e, vs)
            all_nll = all_nll.float().fill_(-9999.9)
            all_nll[v_bi, v_lsi, v_lei] = matches.nll
            matches.nll = all_nll.rename('batch', 'len_s', 'len_e', 'vocab')

        new_extracted = Extracted(batch.batch_size, matches, viable, costs, inverse_unit_costs, len_candidates)
        return new_extracted

    def _get_matches(self, extracted_word_repr: FT, viable_lens: LT, extracted_unit_ids: LT) -> Tuple[Matches, FT]:
        # FIXME(j_luo) Think about how to deal with repr and unit_ids. It is not needed right now to compute costs, but mayber later.
        ns = extracted_word_repr.size('viable')
        len_w = extracted_word_repr.size('len_w')
        nt = len(self.vocab_feat_matrix)
        msl = extracted_word_repr.size('len_w')
        mtl = self.vocab_feat_matrix.size('length')

        # Compute cosine distances all at once: for each viable span, compare it against all units.
        with NoName(self.global_log_probs, extracted_unit_ids):
            costs = -self.global_log_probs[extracted_unit_ids].rename('viable_X_len_w', 'unit')

        # Name: viable x len_w x unit
        costs = costs.unflatten('viable_X_len_w', [('viable', ns), ('len_w', len_w)])

        # NOTE(j_luo) Use dictionary to save every state.
        fs = dict()
        for i in range(msl + 1):
            fs[(i, 0)] = get_zeros(ns, nt).fill_(i * self.ins_del_cost)
        for j in range(mtl + 1):
            fs[(0, j)] = get_zeros(ns, nt).fill_(j * self.ins_del_cost)

        # ------------------------ Main body: DP ----------------------- #

        # Transition.
        with NoName(self.indexed_segments, costs):
            for ls in range(1, msl + 1):
                min_lt = max(ls - 2, 1)
                max_lt = min(ls + 2, mtl + 1)
                for lt in range(min_lt, max_lt):
                    transitions = list()
                    if (ls - 1, lt) in fs:
                        transitions.append(fs[(ls - 1, lt)] + self.ins_del_cost)
                    if (ls, lt - 1) in fs:
                        transitions.append(fs[(ls, lt - 1)] + self.ins_del_cost)
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

        # Get the values wanted.
        with NoName(f, viable_lens, self.vocab_length):
            idx_src = viable_lens.unsqueeze(dim=-1)
            idx_tgt = self.vocab_length
            viable_i = get_range(ns, 2, 0)
            vocab_i = get_range(len(self.vocab_length), 2, 1)
            log_probs = f[idx_src, idx_tgt, viable_i, vocab_i]
            log_probs.rename_('viable', 'vocab')

        # Get the best spans.
        nll = -log_probs
        matches = Matches(nll, f)
        return matches, costs, None
