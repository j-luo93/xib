import logging
import math
from dataclasses import fields
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import (BaseBatch, batch_class, get_array,
                             get_length_mask, get_range)
from dev_misc.devlib.named_tensor import (NameHelper, NoName, Rename,
                                          drop_names, get_named_range)
from dev_misc.utils import WithholdKeys, cached_property, global_property
from xib.aligned_corpus.data_loader import AlignedBatch
from xib.data_loader import (ContinuousIpaBatch, UnbrokenTextBatch,
                             convert_to_dense)
from xib.ipa import Category, Index, get_enum_by_cat, should_include
from xib.ipa.process import (Segment, Segmentation, SegmentWindow, SegmentX,
                             Span)
from xib.model.modules import AdaptLayer, FeatEmbedding

from .modules import DenseFeatEmbedding

ExtractBatch = Union[ContinuousIpaBatch, UnbrokenTextBatch]


@batch_class
class Matches(BaseBatch):
    ll: FT
    f: FT  # All these dp scores.


@batch_class
class Extracted(BaseBatch):
    batch_size: int
    matches: Optional[Matches] = None
    viable: Optional[BT] = None
    start_candidates: Optional[LT] = None
    end_candidates: Optional[LT] = None
    len_candidates: Optional[LT] = None


@batch_class
class ExtractModelReturn(BaseBatch):
    start: LT
    end: LT
    top_matched_ll: FT
    top_matched_vocab: LT
    unmatched_ll: FT
    marginal_ll: FT
    top_word_ll: FT
    best_span_ll: FT
    extracted: Extracted
    alignment: Optional[FT] = None


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

    def __init__(self, lu_size: int, ku_size: int):
        """`lu_size`: number of lost units, `ku_size`: number of known units."""
        super().__init__()

        self.unit_aligner = nn.Embedding(lu_size, ku_size)
        logging.imp('Unit aligner initialized to 0.')
        self.unit_aligner.weight.data.fill_(0.0)

        self.conv = nn.Conv1d(g.dim, g.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)
        self.dropout = nn.Dropout(g.dropout)

    def forward(self, lu_id_seqs: LT, ku_repr: FT) -> Tuple[FT, FT]:
        """Returns lu x ku representation and bs x l x ku representation."""
        lu_char_weight = self.unit_aligner.weight
        lu_char_repr = lu_char_weight @ ku_repr

        lu_char_repr = lu_char_repr.refine_names('lu_char_emb', 'char_emb')
        with NoName(lu_char_repr, lu_id_seqs):
            _lu_repr = lu_char_repr[lu_id_seqs].rename('batch', 'length', 'char_emb')
        _lu_repr = _lu_repr.align_to('batch', 'char_emb', ...)
        with NoName(_lu_repr):
            lu_ctx_repr = self.conv(_lu_repr).rename('batch', 'char_emb', 'length')
        lu_ctx_repr = lu_ctx_repr.align_to(..., 'char_emb')
        lu_ctx_repr = self.dropout(lu_ctx_repr)

        return lu_char_repr, lu_ctx_repr


class ExtractModel(nn.Module):

    add_argument('max_num_words', default=1, dtype=int, msg='Max number of extracted words.')
    add_argument('top_k_predictions', default=10, dtype=int, msg='Number of top predictions to keep.')
    add_argument('max_word_length', default=10, dtype=int, msg='Max length of extracted words.')
    add_argument('init_threshold', default=0.05, dtype=float,
                 msg='Initial value of threshold to determine whether two words are matched.')
    add_argument('use_adapt', default=False, dtype=bool, msg='Flag to use adapter layer.')
    add_argument('init_ins_del_cost', default=100, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('min_ins_del_cost', default=3.5, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('unextracted_prob', default=0.01, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('context_weight', default=0.0, dtype=float, msg='Weight for the context probabilities.')
    add_argument('debug', dtype=bool, default=False, msg='Flag to enter debug mode.')
    add_argument('use_empty_symbol', dtype=bool, default=False, msg='Flag to use empty symbol')
    add_argument('span_candidates', dtype=str,
                 choices=['all', 'oracle_full', 'oracle_stem'], default='all', msg='How to generate candidates for spans.')

    def __init__(self, lu_size: Optional[int] = None, ku_size: Optional[int] = None):
        super().__init__()

        self.adapter = AdaptLayer()

        if g.input_format == 'text':
            self.g2p = G2PLayer(lu_size, ku_size)

    # IDEA(j_luo) The current api is worse than just declaring GlobalProperty(writeable=False) outright. And doesn't give proper type hints.
    @global_property
    def threshold(self):
        pass

    @global_property
    def ins_del_cost(self):
        pass

    @cached_property
    def effective_categories(self) -> List[Category]:
        ret = list()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                ret.append(cat)
        return ret

    def forward(self, batch: Union[ExtractBatch, AlignedBatch]) -> ExtractModelReturn:
        """
        The generating story is:
            v
            |
            w
            |
            x -- ww -- theta

        Pr(x) = sum_w Pr(w) Pr(ww)
              = sum_w Pr(w) theta^|ww|
              = sum_{w, v} Pr(w | v) Pr(v) theta^|ww|

        """
        # Prepare representations.
        alignment = None
        char_log_probs = None
        vocab = batch.known_vocab
        if g.dense_input:
            # IDEA(j_luo) NoName shouldn't use reveal_name. Just keep the name in the context manager.
            with NoName(*vocab.unit_dense_feat_matrix.values()):
                unit_repr = torch.cat([vocab.unit_dense_feat_matrix[cat] for cat in self.effective_categories], dim=-1)
            unit_repr = unit_repr.rename('batch', 'length', 'char_emb').squeeze(dim='length')

            # Both input formats will undergo the same probability calculation, but representations are derived differently. For text, we use g2p; for ipa, we use adapter.
            if g.input_format == 'text':
                lu_char_repr, word_repr = self.g2p(batch.unit_id_seqs, unit_repr)
            else:
                try:
                    lu_char_adapted_dfm = self.adapter(batch.lu_dfm)  # HACK(j_luo) Note that this is a hack.
                except AttributeError:
                    lu_char_adapted_dfm = self.adapter(batch.all_lost_dense_feat_matrix)
                with NoName(*lu_char_adapted_dfm.values()):
                    lu_char_repr = torch.cat([lu_char_adapted_dfm[cat] for cat in self.effective_categories], dim=-1)
                lu_char_repr.squeeze_(dim=1)

                dfm = batch.dense_feat_matrix
                adapted_dfm = self.adapter(dfm)
                with NoName(*adapted_dfm.values()):
                    word_repr = torch.cat([adapted_dfm[cat] for cat in self.effective_categories], dim=-1)
                word_repr.rename_('batch', 'length', 'char_emb')
            char_log_probs = (lu_char_repr @ unit_repr.t()).log_softmax(dim=-1)
            alignment = char_log_probs.exp()
        else:
            with Rename(vocab.unit_feat_matrix, unit='batch'):
                word_repr = self.embedding(batch.feat_matrix, batch.source_padding)
                unit_repr = self.embedding(vocab.unit_feat_matrix)
            unit_repr = unit_repr.squeeze('length')
        unit_repr.rename_(batch='unit')

        # Main body: extract one span.
        extracted = Extracted(batch.batch_size)
        new_extracted = self._extract_one_span(batch, extracted, word_repr, unit_repr, char_log_probs)
        matches = new_extracted.matches
        len_e = matches.ll.size('len_e')
        vs = len(vocab)

        # Get the best score and span.
        lp_per_unmatched = math.log(g.unextracted_prob)
        # NOTE(j_luo) Some segments don't have any viable spans.
        nh = NameHelper()
        flat_ll = nh.flatten(matches.ll, ['len_s', 'len_e', 'vocab'], 'cand')
        flat_viable = nh.flatten(new_extracted.viable.expand_as(matches.ll), ['len_s', 'len_e', 'vocab'], 'cand')
        flat_viable_ll = (~flat_viable) * (-9999.9) + flat_ll
        # Add probs for unextracted characters.
        unextracted = batch.lengths.align_as(new_extracted.len_candidates) - new_extracted.len_candidates
        unextracted = torch.where(new_extracted.viable, unextracted, torch.zeros_like(unextracted))
        unextracted = unextracted.expand_as(matches.ll)
        flat_unextracted = nh.flatten(unextracted, ['len_s', 'len_e', 'vocab'], 'cand')
        flat_unextracted_ll = flat_unextracted * lp_per_unmatched
        flat_total_ll = flat_viable_ll + flat_unextracted_ll
        # Get the top candiates based on total scores.
        k = min(g.top_k_predictions, flat_total_ll.size('cand'))
        top_matched_ll, top_span_ind = torch.topk(flat_total_ll, k, dim='cand')
        top_word_ll = flat_viable_ll.gather('cand', top_span_ind)
        start_idx = top_span_ind // (len_e * vs)
        end_idx = top_span_ind % (len_e * vs) // vs
        batch_idx = torch.arange(batch.batch_size).unsqueeze(dim=-1)
        with NoName(start_idx, end_idx, new_extracted.start_candidates, new_extracted.end_candidates):
            # pylint: disable=unsubscriptable-object
            start = new_extracted.start_candidates[batch_idx, start_idx, end_idx]
            end = new_extracted.end_candidates[batch_idx, start_idx, end_idx]
        top_matched_vocab = top_span_ind % vs
        # Get unmatched scores -- no word is matched for the entire inscription.
        unmatched_ll = batch.lengths * lp_per_unmatched
        # Concatenate all.
        marginal_ll = torch.cat([flat_total_ll, unmatched_ll.align_to('batch', 'cand')], dim='cand')
        marginal_ll = marginal_ll.logsumexp(dim='cand')

        if g.debug:
            import pandas as pd
            pd.set_option('pprint_nest_depth', 0)
            from dev_misc.devlib.inspector import Inspector
            ins = Inspector()
            ins.add_table(nh.unflatten(flat_total_ll, 'cand', ['len_s', 'len_e', 'vocab']), 'll')
            ins.add_table(new_extracted.start_candidates, 'sc')
            ins.add_table(new_extracted.len_candidates, 'lc')
            ins.add_table(new_extracted.matches.f, 'f', auto_merge=False)
            ins.add_table(new_extracted.viable, 'viable', is_mask_index=True)
            ins.add_table(batch.known_vocab.vocab, 'vocab', is_index=True)

        ret = ExtractModelReturn(start, end, top_matched_ll, top_matched_vocab,
                                 unmatched_ll, marginal_ll, top_word_ll, new_extracted, alignment)

        return ret

    def _extract_one_span(self,
                          batch: Union[ExtractBatch, AlignedBatch],
                          extracted: Extracted,
                          word_repr: FT,
                          unit_repr: FT,
                          char_log_probs: Optional[FT] = None) -> Extracted:
        if g.span_candidates == 'all':
            # Propose all span start/end positions.
            start_candidates = get_named_range(batch.max_length, 'len_s').align_to('batch', 'len_s', 'len_e')
            # Range from `min_word_length` to `max_word_length`.
            len_candidates = get_named_range(g.max_word_length + 1 - g.min_word_length, 'len_e') + g.min_word_length
            len_candidates = len_candidates.align_to('batch', 'len_s', 'len_e')
        else:
            # Start from the first position.
            start_candidates = torch.zeros_like(batch.lengths).align_to('batch', 'len_s', 'len_e')
            # Use full word length.
            len_candidates = batch.lengths.align_to('batch', 'len_s', 'len_e')
            len_candidates.clamp_(min=g.min_word_length, max=g.max_word_length)
        # This is inclusive.
        end_candidates = start_candidates + len_candidates - 1

        # Only keep the viable/valid spans around.
        viable = (end_candidates < batch.lengths.align_as(end_candidates))
        start_candidates = start_candidates.expand_as(viable)
        end_candidates = end_candidates.expand_as(viable)
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
        matches = self._get_matches(batch, extracted_word_repr, unit_repr,
                                    viable_lens, extracted_unit_ids, char_log_probs)
        # Revert to the old shape (so that invalid spans are included).
        bi = get_named_range(batch.batch_size, 'batch').expand_as(viable)
        lsi = get_named_range(len_s, 'len_s').expand_as(viable)
        lei = get_named_range(len_e, 'len_e').expand_as(viable)
        vs = matches.ll.size('vocab')
        # IDEA(j_luo) NoName shouldn't make size() calls unavaiable. Otherwise size() calls have to be moved outside the context. Also the names should be preserved as well.
        with NoName(bi, lsi, lei, viable, matches.ll):
            v_bi = bi[viable]
            v_lsi = lsi[viable]
            v_lei = lei[viable]
            all_ll = get_zeros(batch.batch_size, len_s, len_e, vs)
            all_ll = all_ll.float().fill_(-9999.9)
            all_ll[v_bi, v_lsi, v_lei] = matches.ll
            matches.ll = all_ll.rename('batch', 'len_s', 'len_e', 'vocab')

        new_extracted = Extracted(batch.batch_size, matches, viable, start_candidates, end_candidates, len_candidates)
        return new_extracted

    def _get_matches(self,
                     batch: Union[ExtractBatch, AlignedBatch],
                     extracted_word_repr: FT,
                     unit_repr: FT,
                     viable_lens: LT,
                     extracted_unit_ids: LT,
                     char_log_probs: Optional[FT] = None) -> Matches:
        vocab = batch.known_vocab
        ns = extracted_word_repr.size('viable')
        len_w = extracted_word_repr.size('len_w')
        nt = len(vocab.vocab_feat_matrix)
        msl = extracted_word_repr.size('len_w')
        mtl = vocab.vocab_feat_matrix.size('length')

        # Compute cosine distances all at once: for each viable span, compare it against all units.
        if g.input_format == 'text':
            ctx_logits = extracted_word_repr @ unit_repr.t()
            ctx_log_probs = ctx_logits.log_softmax(dim='unit').flatten(['viable', 'len_w'], 'viable_X_len_w')
            with NoName(char_log_probs, extracted_unit_ids):
                global_log_probs = char_log_probs[extracted_unit_ids].rename('viable_X_len_w', 'unit')
            # lp1 = math.log(g.context_weight) + ctx_log_probs
            # lp2 = math.log(1.0 - g.context_weight) + global_log_probs
            # weighted_log_probs = torch.stack([lp1, lp2], new_name='mixture').logsumexp(dim='mixture')
            weighted_log_probs = g.context_weight * ctx_log_probs + (1.0 - g.context_weight) * global_log_probs
            costs = -weighted_log_probs
            costs = costs.unflatten('viable_X_len_w', [('viable', ns), ('len_w', len_w)])
        else:
            logits = extracted_word_repr @ unit_repr.t()
            costs = -logits.log_softmax(dim='unit')

        # NOTE(j_luo) Use dictionary to save every state.
        fs = dict()
        for i in range(msl + 1):
            fs[(i, 0)] = get_zeros(ns, nt).fill_(i * self.ins_del_cost)
        for j in range(mtl + 1):
            fs[(0, j)] = get_zeros(ns, nt).fill_(j * self.ins_del_cost)

        # ------------------------ Main body: DP ----------------------- #

        # Transition.
        with NoName(vocab.indexed_segments, costs):
            for ls in range(1, msl + 1):
                min_lt = max(ls - 2, 1)
                max_lt = min(ls + 2, mtl + 1)
                for lt in range(min_lt, max_lt):
                    transitions = list()
                    if (ls - 1, lt) in fs:
                        if g.use_empty_symbol:
                            del_cost = costs[..., ls - 1, 0].rename(None).unsqueeze(dim=1)
                            transitions.append(fs[(ls - 1, lt)] + del_cost)
                        else:
                            transitions.append(fs[(ls - 1, lt)] + self.ins_del_cost)
                    if (ls, lt - 1) in fs:
                        # transitions.append(fs[(ls, lt - 1)] + self.ins_del_cost)
                        # FIXME(j_luo) How to parameterize insertion costs.
                        transitions.append(fs[(ls, lt - 1)] + self.ins_del_cost)
                    if (ls - 1, lt - 1) in fs:
                        vocab_inds = vocab.indexed_segments[:, lt - 1]
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
        f = torch.stack(f_lst, dim=0).view(msl + 1, mtl + 1, -1, len(vocab))
        f.rename_('len_w_src', 'len_w_tgt', 'viable', 'vocab')

        # Get the values wanted.
        with NoName(f, viable_lens, vocab.vocab_length):
            idx_src = viable_lens.unsqueeze(dim=-1)
            idx_tgt = vocab.vocab_length
            viable_i = get_range(ns, 2, 0)
            vocab_i = get_range(len(vocab.vocab_length), 2, 1)
            nll = f[idx_src, idx_tgt, viable_i, vocab_i]
            nll.rename_('viable', 'vocab')

        # Get the best spans.
        matches = Matches(-nll, f)
        return matches
