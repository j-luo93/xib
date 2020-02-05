import logging
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from dev_misc import BT, FT, LT, add_argument, g, get_tensor, get_zeros
from dev_misc.devlib import (BaseBatch, batch_class, get_array,
                             get_length_mask, get_range)
from dev_misc.devlib.named_tensor import (NameHelper, NoName, Rename,
                                          drop_names, get_named_range)
from dev_misc.utils import WithholdKeys, cached_property, global_property
from xib.aligned_corpus.char_set import DELETE_ID
from xib.aligned_corpus.corpus import AlignedSentence
from xib.aligned_corpus.data_loader import AlignedBatch
from xib.data_loader import (ContinuousIpaBatch, UnbrokenTextBatch,
                             convert_to_dense)
from xib.ipa import Category, Index, get_enum_by_cat, should_include
from xib.ipa.process import (Segment, Segmentation, SegmentWindow, SegmentX,
                             Span)
from xib.model.modules import AdaptLayer, FeatEmbedding
from xib.model.span_proposer import (AllSpanProposer, OracleStemSpanProposer,
                                     OracleWordSpanProposer, ViableSpans)

from .modules import DenseFeatEmbedding

ExtractBatch = Union[ContinuousIpaBatch, UnbrokenTextBatch]


@batch_class
class Matches(BaseBatch):
    ll: FT
    f: FT  # All these dp scores.
    raw_ll: FT = None  # HACK(j_luo)


@batch_class
class Extracted(BaseBatch):
    batch_size: int
    matches: Matches
    viable_spans: ViableSpans


@batch_class
class Costs(BaseBatch):
    sub: FT
    ins: FT


@batch_class
class CtcBookkeeper(BaseBatch):
    best_vocab: LT
    best_prev_tags: Dict[int, FT] = field(init=False, default_factory=dict)


@batch_class
class CtcReturn(BaseBatch):
    final_nodes: FT
    final_score: FT
    bookkeeper: Optional[CtcBookkeeper] = None
    expected_num_spans: Optional[FT] = None


@batch_class
class ExtractModelReturn(BaseBatch):
    start: LT
    end: LT
    top_matched_ll: FT
    top_matched_vocab: LT
    unmatched_ll: FT
    marginal: FT
    top_word_ll: FT
    best_span_ll: FT
    extracted: Extracted
    alignment: Optional[FT] = None
    ctc_return: Optional[CtcReturn] = None


class ExtractModel(nn.Module):

    add_argument('g2p_window_size', default=3, dtype=int, msg='Window size for g2p layer.')
    add_argument('max_num_words', default=1, dtype=int, msg='Max number of extracted words.')
    add_argument('top_k_predictions', default=10, dtype=int, msg='Number of top predictions to keep.')
    add_argument('max_word_length', default=10, dtype=int, msg='Max length of extracted words.')
    add_argument('init_threshold', default=0.05, dtype=float,
                 msg='Initial value of threshold to determine whether two words are matched.')
    add_argument('use_adapt', default=False, dtype=bool, msg='Flag to use adapter layer.')
    add_argument('init_ins_del_cost', default=100, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('min_ins_del_cost', default=3.5, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('context_weight', default=0.0, dtype=float, msg='Weight for the context probabilities.')
    add_argument('context_agg_mode', default='log_interpolation', dtype=str,
                 choices=['log_interpolation', 'linear_interpolation', 'log_add', 'dilute'])
    add_argument('dilute_hyper', dtype=float, default=0.5)
    add_argument('debug', dtype=bool, default=False, msg='Flag to enter debug mode.')
    add_argument('truncate_unextracted', dtype=bool, default=False, msg='Flag to enter debug mode.')
    add_argument('use_empty_symbol', dtype=bool, default=False, msg='Flag to use empty symbol')
    add_argument('use_base_embedding', dtype=bool, default=False,
                 msg='Flag to use base embeddings to compute character embeddings.')
    add_argument('base_embedding_dim', dtype=int, default=250, msg='Dimensionality for base embeddings.')
    add_argument('span_candidates', dtype=str,
                 choices=['all', 'oracle_word', 'oracle_stem'], default='all', msg='How to generate candidates for spans.')
    add_argument('one2two', dtype=bool, default=False, msg='Use conv on both sides.')
    add_argument('ins_del_prior', dtype=float, default=0.1, msg='Prior value for insertion/deletion operations.')
    add_argument('cut_off', dtype=float, nargs='+')
    add_argument('include_unmatched', dtype=bool, default=True, msg='Flag to include unmatched scores in the loss.')
    add_argument('em_training', dtype=bool, default=False)
    add_argument('use_ctc', dtype=bool, default=False)
    add_argument('use_s_prior', dtype=bool, default=False)
    add_argument('best_ctc', dtype=bool, default=False)
    add_argument('one_span_hack', dtype=bool, default=False)
    add_argument('dense_embedding', dtype=bool, default=False)
    add_argument('use_posterior_reg', dtype=bool, default=False)
    add_argument('use_constrained_learning', dtype=bool, default=False)
    add_argument('non_span_bias', dtype=float, default=0.5)
    add_argument('expected_ratio', dtype=float, default=0.2)

    def __init__(self, lost_size: int, known_size: int, dl):  # FIXME(j_luo) type hints here
        super().__init__()
        if g.use_base_embedding:
            if g.dense_embedding:
                self.dim = g.dim * 7
            else:
                self.dim = g.base_embedding_dim
        else:
            self.dim = g.dim
        self.unit_aligner = nn.Embedding(lost_size, known_size)
        logging.imp('Unit aligner initialized to 0.')
        self.unit_aligner.weight.data.fill_(0.0)
        # logging.imp('Unit aligner initialized uniformly.')
        # nn.init.uniform_(self.unit_aligner.weight, -0.01, 0.01)
        self.conv = nn.Conv1d(self.dim, self.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)
        self.ins_conv = nn.Conv1d(self.dim, self.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)
        self.dropout = nn.Dropout(g.dropout)
        opt2sp_cls = {'all': AllSpanProposer,
                      'oracle_word': OracleWordSpanProposer,
                      'oracle_stem': OracleStemSpanProposer}
        sp_cls = opt2sp_cls[g.span_candidates]
        if g.span_candidates == 'all':
            self.span_proposer = sp_cls(dl)
        else:
            self.span_proposer = sp_cls()
        if g.use_base_embedding:
            if g.dense_embedding:
                self.base_embeddings = DenseFeatEmbedding('feat_emb', 'feat_group', 'char_emb', dim=g.dim)
            else:
                self.base_embeddings = nn.Embedding(60, g.base_embedding_dim)
                logging.imp('Base embeddigns initialized uniformly.')
                nn.init.uniform_(self.base_embeddings.weight, -0.05, 0.05)

    @global_property
    def ins_del_cost(self):
        pass

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd['span_proposer'] = self.span_proposer.state_dict(*args, **kwargs)
        return sd

    @cached_property
    def effective_categories(self) -> List[Category]:
        ret = list()
        for cat in Category:
            if should_include(g.feat_groups, cat):
                ret.append(cat)
        return ret

    def forward(self, batch: AlignedBatch) -> ExtractModelReturn:
        vocab = batch.known_vocab
        # Get relevant representations -- one for all the known units, and one for a contextualized representation.
        known_unit_emb, known_ctx_repr, known_ins_ctx_repr = self._get_repr(batch)

        # Get cost matrix.
        costs, alignment = self._get_costs(vocab.indexed_segments,
                                           known_unit_emb,
                                           known_ctx_repr,
                                           known_ins_ctx_repr)

        # Get span candidates:
        viable_spans = self.span_proposer(batch.unit_id_seqs, batch.lengths, batch.sentences)

        # Get matches.
        matches = self._get_matches(costs,
                                    viable_spans,
                                    vocab.vocab_length)

        # Get extracted object.
        extracted = self._get_extracted(matches, viable_spans, len(vocab))

        # Prepare return.
        model_ret = self._prepare_return(batch, extracted, alignment)

        return model_ret

    def _get_repr(self, batch: AlignedBatch) -> Tuple[FT, FT, FT]:
        vocab = batch.known_vocab
        # Get unit embeddings for each known unit.
        if g.dense_embedding and g.use_base_embedding:
            with Rename(*vocab.unit_dense_feat_matrix.values(), unit='batch'):
                known_unit_emb = self.base_embeddings(vocab.unit_dense_feat_matrix)
            known_unit_emb = self.dropout(known_unit_emb)
            with NoName(known_unit_emb):
                known_unit_emb = known_unit_emb / (1e-8 + known_unit_emb.norm(dim=-1, keepdim=True)) * math.sqrt(5)
        else:
            with NoName(*vocab.unit_dense_feat_matrix.values()):
                known_unit_emb = torch.cat([vocab.unit_dense_feat_matrix[cat]
                                            for cat in self.effective_categories], dim=-1)
                if g.use_base_embedding:
                    known_unit_emb = known_unit_emb @ self.base_embeddings.weight
                    known_unit_emb = self.dropout(known_unit_emb)
                    known_unit_emb = known_unit_emb / (1e-8 + known_unit_emb.norm(dim=-1, keepdim=True)) * math.sqrt(5)
        known_unit_emb = known_unit_emb.rename('known_unit', 'length', 'char_emb').squeeze(dim='length')

        # Get known embedding sequences for the entire vocabulary.
        max_len = vocab.indexed_segments.size('length')
        with NoName(vocab.indexed_segments, known_unit_emb, vocab.vocab_length):
            known_vocab_emb = known_unit_emb[vocab.indexed_segments]
            length_mask = get_length_mask(vocab.vocab_length, max_len).unsqueeze(dim=-1)
            known_vocab_emb.masked_fill_(~length_mask, 0.0)
            known_vocab_emb.rename_('vocab', 'length', 'char_emb')

        # Get known contextualized embeddings.
        inp_conv = known_vocab_emb.align_to('vocab', 'char_emb', 'length')
        with NoName(inp_conv):
            known_ctx_repr = self.conv(inp_conv).rename('vocab', 'char_emb', 'length')
            known_ins_ctx_repr = self.ins_conv(inp_conv).rename('vocab', 'char_emb', 'length')
        known_ctx_repr = known_ctx_repr.align_to('vocab', 'length', 'char_emb')
        known_ins_ctx_repr = known_ins_ctx_repr.align_to('vocab', 'length', 'char_emb')

        return known_unit_emb, known_ctx_repr, known_ins_ctx_repr

    def _get_costs(self, vocab_unit_id_seqs: LT, known_unit_emb: FT, known_ctx_repr: FT, known_ins_ctx_repr: FT) -> Tuple[Costs, FT]:
        # Get lost unit embeddings.
        lost_unit_weight = self.unit_aligner.weight
        lost_unit_emb = lost_unit_weight @ known_unit_emb
        # with NoName(lost_unit_emb):
        #     lost_unit_emb = (lost_unit_emb / (1e-8 + lost_unit_emb.norm(dim=-1, keepdim=True))) * math.sqrt(7)
        #     lost_unit_emb.rename_('lost_unit', 'char_emb')

        # Get global (non-contextualized) log probs.
        unit_logits = known_unit_emb @ lost_unit_emb.t()
        unit_log_probs = unit_logits.log_softmax(dim=-1)
        alignment = unit_log_probs.exp()
        with NoName(unit_log_probs, vocab_unit_id_seqs):
            global_log_probs = unit_log_probs[vocab_unit_id_seqs].rename('vocab', 'length', 'lost_unit')

        # Get contextualized log probs.
        sub_ctx_logits = known_ctx_repr @ lost_unit_emb.t()
        sub_ctx_log_probs = sub_ctx_logits.log_softmax(dim=-1).rename('vocab', 'length', 'lost_unit')

        def interpolate(ctx, plain):
            if g.context_agg_mode == 'log_interpolation':
                return g.context_weight * ctx + (1.0 - g.context_weight) * plain
            elif g.context_agg_mode == 'linear_interpolation':
                z_ctx = math.log(g.context_weight + 1e-8)
                z_plain = math.log(1.0 - g.context_weight + 1e-8)
                return torch.stack([z_ctx + ctx, z_plain + plain], new_name='stacked').logsumexp(dim='stacked')
            elif g.context_agg_mode == 'log_add':
                return ctx + plain
            else:
                return (ctx + plain) * g.dilute_hyper

        # Get interpolated log probs and costs.
        sub_weighted_log_probs = interpolate(sub_ctx_log_probs, global_log_probs)
        sub = -sub_weighted_log_probs

        # Get secondary costs for insertions.
        ins_ctx_logits = known_ins_ctx_repr @ lost_unit_emb.t()
        ins_ctx_log_probs = ins_ctx_logits.log_softmax(dim=-1).rename('vocab', 'length', 'lost_unit')
        ins_weighted_log_probs = interpolate(ins_ctx_log_probs, global_log_probs)
        ins = -ins_weighted_log_probs + self.ins_del_cost

        costs = Costs(sub, ins)

        return costs, alignment

    def _get_matches(self, costs: Costs, viable_spans: ViableSpans, vocab_lengths: LT) -> Matches:
        nk = costs.sub.size('vocab')
        nl = viable_spans.starts.size('viable')
        mkl = costs.sub.size('length')
        mll = viable_spans.lengths.max().item()
        # NOTE(j_luo) Use dictionary to save every state.
        fs = dict()
        if g.one2two:
            fs[(0, 0)] = get_zeros(nk, nl)
            with NoName(costs.sub):
                for i in range(1, mkl + 1):
                    del_cost = costs.sub[:, i - 1, DELETE_ID].unsqueeze(dim=-1)  # - math.log(g.ins_del_prior)
                    fs[(i, 0)] = fs[(i - 1, 0)] + del_cost
        else:
            for i in range(mkl + 1):
                fs[(i, 0)] = get_zeros(nk, nl).fill_(i * self.ins_del_cost)
            for j in range(mll + 1):
                fs[(0, j)] = get_zeros(nk, nl).fill_(j * self.ins_del_cost)

        # ------------------------ Main body: DP ----------------------- #

        # Transition.
        with NoName(costs.sub, costs.ins, viable_spans.unit_id_seqs):
            for kl in range(1, mkl + 1):
                min_ll = max(kl - 2, 1)
                max_ll = min(kl + 3, mll + 1)
                for ll in range(min_ll, max_ll):
                    self._update_fs(fs, costs, viable_spans, kl, ll)

        f_lst = list()
        for i in range(mkl + 1):
            for j in range(mll + 1):
                if (i, j) not in fs:
                    fs[(i, j)] = get_zeros(nk, nl).fill_(9999.9)
                f_lst.append(fs[(i, j)])
        f = torch.stack(f_lst, dim=0).view(mkl + 1, mll + 1, nk, nl)
        f.rename_('known_pos', 'lost_pos', 'vocab', 'viable')

        # Get the values wanted.
        with NoName(f, viable_spans.lengths, vocab_lengths):
            idx_known = vocab_lengths.unsqueeze(dim=-1)
            idx_lost = viable_spans.lengths
            vocab_i = get_range(nk, 2, 0)
            viable_i = get_range(nl, 2, 1)
            nll = f[idx_known, idx_lost, vocab_i, viable_i]
            nll.rename_('vocab', 'viable')

        # Get the best spans.
        matches = Matches(-nll, f)
        return matches

    def _get_extracted(self, matches: Matches, viable_spans: ViableSpans, vocab_size: int) -> Extracted:
        viable = viable_spans.viable
        batch_size = viable.size('batch')
        len_s = viable.size('len_s')
        len_e = viable.size('len_e')
        # Revert to the old shape (so that invalid spans are included).
        bi = get_named_range(batch_size, 'batch').expand_as(viable)
        lsi = get_named_range(len_s, 'len_s').expand_as(viable)
        lei = get_named_range(len_e, 'len_e').expand_as(viable)
        vs = matches.ll.size('vocab')
        with NoName(bi, lsi, lei, viable, matches.ll, viable_spans.p_weights):
            v_bi = bi[viable]
            v_lsi = lsi[viable]
            v_lei = lei[viable]
            all_ll = get_zeros(batch_size, len_s, len_e, vs)
            all_ll.fill_(-9999.9)
            all_ll[v_bi, v_lsi, v_lei] = matches.ll.t()
            all_ll.rename_('batch', 'len_s', 'len_e', 'vocab')
            # NOTE(j_luo) Remember to add the prior for vocab.
            matches.ll = all_ll - math.log(vocab_size)
            matches.raw_ll = all_ll
            # NOTE(j_luo) `p_weights` is now in log scale.
            all_p_weights = get_zeros(batch_size, len_s, len_e)
            if g.use_constrained_learning:
                all_p_weights[v_bi, v_lsi, v_lei] = viable_spans.p_weights.exp()
            else:
                all_p_weights.fill_(-9999.9)
                all_p_weights[v_bi, v_lsi, v_lei] = (viable_spans.p_weights + 1e-8).log()
            viable_spans.p_weights = all_p_weights.rename('batch', 'len_s', 'len_e')

        extracted = Extracted(batch_size,
                              matches,
                              viable_spans)
        return extracted

    def _update_fs(self, fs: Dict[Tuple[int, int], FT], costs: Costs, viable_spans: ViableSpans, kl: int, ll: int):
        transitions = list()
        if (kl - 1, ll) in fs:
            if g.one2two:
                del_cost = costs.sub[:, kl - 1, DELETE_ID].unsqueeze(dim=-1)  # - math.log(g.ins_del_prior)
            else:
                del_cost = self.ins_del_cost
            transitions.append(fs[(kl - 1, ll)] + del_cost)
        if not g.one2two and (kl, ll - 1) in fs:
            # if g.use_conv_both_sides:
            #     ins_cost = costs[:, kl - 1, INSERT_ID].unsqueeze(dim=-1) - math.log(g.ins_del_prior)
            # else:
            ins_cost = self.ins_del_cost
            transitions.append(fs[(kl, ll - 1)] + ins_cost)
        if g.one2two and (kl - 1, ll - 2) in fs:
            first_lost_ids = viable_spans.unit_id_seqs[:, ll - 2]
            sub_cost = costs.sub[:, kl - 1, first_lost_ids]  # - math.log(g.ins_del_prior)
            second_lost_ids = viable_spans.unit_id_seqs[:, ll - 1]
            ins_cost = costs.ins[:, kl - 1, second_lost_ids]  # - math.log(g.ins_del_prior)
            transitions.append(fs[(kl - 1, ll - 2)] + ins_cost + sub_cost)
        if (kl - 1, ll - 1) in fs:
            lost_ids = viable_spans.unit_id_seqs[:, ll - 1]
            sub_cost = costs.sub[:, kl - 1, lost_ids]
            if g.one2two:
                sub_cost = sub_cost  # - math.log(1.0 - g.ins_del_prior)
            transitions.append(fs[(kl - 1, ll - 1)] + sub_cost)
        if transitions:
            all_s = torch.stack(transitions, dim=-1)
            new_s, _ = all_s.min(dim=-1)
            fs[(kl, ll)] = new_s

    @property
    def lp_per_unmatched(self) -> float:
        return math.log(1.0 / self.unit_aligner.num_embeddings)

    def _prepare_return(self, batch: AlignedBatch, extracted: Extracted, alignment: FT) -> ExtractModelReturn:
        # Get the best score and span.
        matches = extracted.matches
        len_e = matches.ll.size('len_e')
        vs = len(batch.known_vocab)
        # NOTE(j_luo) Some segments don't have any viable spans.
        nh = NameHelper()
        viable_spans = extracted.viable_spans
        if g.em_training:
            flat_ll = nh.flatten(matches.ll,
                                 ['len_s', 'len_e', 'vocab'], 'cand')
        else:
            flat_ll = nh.flatten(matches.ll,  # + viable_spans.p_weights.align_as(matches.ll),
                                 ['len_s', 'len_e', 'vocab'], 'cand')
        flat_viable = nh.flatten(viable_spans.viable.expand_as(matches.ll), ['len_s', 'len_e', 'vocab'], 'cand')
        flat_viable_ll = (~flat_viable) * (-9999.9) + flat_ll
        # Add probs for unextracted characters.
        if g.truncate_unextracted:
            # Truncate unextracted segments.
            max_lengths_to_consider = batch.lengths.clamp(max=g.max_word_length)
            unextracted = max_lengths_to_consider.align_as(viable_spans.len_candidates) - viable_spans.len_candidates
        else:
            unextracted = batch.lengths.align_as(viable_spans.len_candidates) - viable_spans.len_candidates
        unextracted = torch.where(viable_spans.viable, unextracted, torch.zeros_like(unextracted))
        unextracted = unextracted.expand_as(matches.ll)
        flat_unextracted = nh.flatten(unextracted, ['len_s', 'len_e', 'vocab'], 'cand')
        flat_unextracted_ll = flat_unextracted * self.lp_per_unmatched
        flat_total_ll = flat_viable_ll + flat_unextracted_ll
        # Get the top candiates based on total scores.
        k = min(g.top_k_predictions, flat_total_ll.size('cand'))
        top_matched_ll, top_span_ind = torch.topk(flat_total_ll, k, dim='cand')
        top_word_ll = flat_viable_ll.gather('cand', top_span_ind)
        start_idx = top_span_ind // (len_e * vs)
        end_idx = top_span_ind % (len_e * vs) // vs
        batch_idx = torch.arange(batch.batch_size).unsqueeze(dim=-1)
        with NoName(start_idx, end_idx, viable_spans.start_candidates, viable_spans.end_candidates):
            # pylint: disable=unsubscriptable-object
            start = viable_spans.start_candidates[batch_idx, start_idx, end_idx]
            end = viable_spans.end_candidates[batch_idx, start_idx, end_idx]
        top_matched_vocab = top_span_ind % vs
        # Get unmatched scores -- no word is matched for the entire inscription.
        unmatched_ll = batch.lengths * self.lp_per_unmatched
        total_ll = nh.unflatten(flat_total_ll, 'cand', ['len_s', 'len_e', 'vocab'])
        total_span_ll = nh.flatten(total_ll, ['len_s', 'len_e'], 'span')
        best_span_ll, _ = total_span_ll.max(dim='span')
        best_span_ll = best_span_ll.logsumexp(dim='vocab')
        if g.include_unmatched:
            best_span_ll = torch.max(best_span_ll, unmatched_ll)
        # Get marginal.
        ctc_return = None
        if g.use_constrained_learning:
            viable_ll = nh.unflatten(flat_viable_ll, 'cand', ['len_s', 'len_e', 'vocab'])
            span_log_probs = viable_ll.logsumexp(dim='vocab')
            marginal = span_log_probs * viable_spans.p_weights
        elif g.use_ctc:
            viable_ll = nh.unflatten(flat_viable_ll, 'cand', ['len_s', 'len_e', 'vocab'])
            span_log_probs = viable_ll.logsumexp(dim='vocab')
            if g.update_p_weights and self.training and g.em_training:
                marginal = span_log_probs * viable_spans.p_weights.exp()
            else:
                ctc_return = self._run_ctc(batch.lengths, span_log_probs, matches.ll, matches.raw_ll)
                marginal = ctc_return.final_score
            # old_marginal = torch.cat([flat_total_ll, unmatched_ll.align_to('batch', 'cand')], dim='cand')
            # old_marginal = old_marginal.logsumexp(dim='cand')
            # print(marginal.sum().item(), old_marginal.sum().item())
        elif g.em_training:
            # if self.training:
            #    breakpoint()
            viable_ll = nh.unflatten(flat_viable_ll, 'cand', ['len_s', 'len_e', 'vocab'])
            viable_span_ll = viable_ll.logsumexp(dim='vocab')
            p_exp = viable_spans.p_weights.exp()
            p_exp = torch.where(p_exp < 1e-3, torch.zeros_like(p_exp), p_exp)
            weighted_ll = viable_span_ll * p_exp
            # viable_spans.viable.sum(dim=['len_s', 'len_e'])
            marginal = weighted_ll.sum(dim=['len_s', 'len_e']) / (1e-8 + p_exp.sum(dim=['len_s', 'len_e']))
            # if g.include_unmatched:
            #     marginal = torch.stack([marginal, unmatched_ll], new_name='stacked')
            #     marginal = marginal.logsumexp(dim='stacked')
        elif g.include_unmatched:
            marginal = torch.cat([flat_total_ll, unmatched_ll.align_to('batch', 'cand')], dim='cand')
            marginal = marginal.logsumexp(dim='cand')
        else:
            marginal = flat_total_ll.logsumexp(dim='cand')

        if g.use_s_prior:
            s_log_probs = -(1e-8 + viable_spans.viable.sum(dim=['len_s', 'len_e'])).log()
            marginal = marginal + s_log_probs

        model_ret = ExtractModelReturn(start,
                                       end,
                                       top_matched_ll,
                                       top_matched_vocab,
                                       unmatched_ll,
                                       marginal,
                                       top_word_ll,
                                       best_span_ll,
                                       extracted,
                                       alignment,
                                       ctc_return)
        return model_ret

    @global_property
    def cut_off(self):
        pass

    def _run_ctc(self, lengths: LT, span_log_probs: FT, vocab_log_probs: FT, raw_vocab_log_probs: FT) -> CtcReturn:
        r"""To speed up DP, everything is packed into tensors.

        tag \ case
        -------------------------------------------------------------
        O           (l-1, O)    (l-1, E_m)      ...     (l-1, E_M)
        E_m         (l-m, O)  (l-m, E_m)    ...     (l-m, E_M)
        E_m+1       (l-m-1, O)    (l-m-1, E_m)      ...     (l-m-1, E_M)
        ...
        E_M         (l-M, O)  (l-M, E_m)    ...     (l-M, E_M)
        """
        ctc = dict()
        max_length = lengths.max().item()
        batch_size = span_log_probs.size('batch')
        if self.training:
            log_probs = span_log_probs
            bookkeeper = None
        else:
            log_probs, best_vocab = vocab_log_probs.max(dim='vocab')
            bookkeeper = CtcBookkeeper(best_vocab)

            if g.cut_off is not None:
                warnings.warn('Only taking most confident ones.')
                len_range = get_named_range(log_probs.size('len_e'), 'len_e') + g.min_word_length
                avg = raw_vocab_log_probs.gather('vocab', best_vocab) / len_range.align_as(best_vocab)
                log_probs = torch.where(avg > self.cut_off, log_probs, torch.zeros_like(log_probs).fill_(-9999.9))

        def get_init(init_value: float = -9999.9):
            ret = get_zeros(batch_size, g.max_word_length - g.min_word_length + 2 + g.one_span_hack).fill_(init_value)
            ret.rename_('batch', 'tag')
            return ret

        start = get_init()
        start[:, 0] = 0.0
        ctc[0] = start

        if g.use_posterior_reg and self.training:
            pr = dict()
            pr_start = get_init()
            pr_start[:, 0] = -99.9
            pr[0] = pr_start

        # HACK(j_luo)
        if g.one_span_hack:
            E_mask = get_init()
            E_mask[:, 0] = 0.0
            O_mask = get_init()
            O_mask[:, 0] = 0.0
            O2_mask = get_init()
            O2_mask[:, 1:] = 0.0

        # # HACK(j_luo)
        # hack = True
        # hack_per_step = math.log(0.5)

        non_span_bias = math.log(g.non_span_bias + 1e-8)
        span_bias = math.log((1.0 - g.non_span_bias + 1e-8) /
                             (g.max_word_length - g.min_word_length + 1 + g.one_span_hack))
        #non_span_bias = span_bias = 0.0

        for l in range(1, max_length + 1):
            transitions = list()
            pr_trans = list()
            tmp = list()
            another_tmp = list()
            if g.one_span_hack:
                # Case 'O'.
                transitions.append(ctc[l - 1] + O_mask + self.lp_per_unmatched + non_span_bias)
                # Case 'O2'.
                transitions.append(ctc[l - 1] + O2_mask + self.lp_per_unmatched + non_span_bias)
            else:
                transitions.append(ctc[l - 1] + self.lp_per_unmatched + non_span_bias)
                if g.use_posterior_reg and self.training:
                    # pr_trans.append((pr[l - 1] + self.lp_per_unmatched + non_span_bias))
                    pr_trans.append((pr[l - 1] + self.lp_per_unmatched +
                                     non_span_bias).align_to('batch', 'tag', 'new_tag'))

            # Case 'E's.
            for word_len in range(g.min_word_length, g.max_word_length + 1):
                prev_l = l - word_len
                try:
                    prev_v = ctc[prev_l]
                    # FIXME(j_luo)  This is wrong if oracle spans are used.
                    start_idx = prev_l
                    end_idx = word_len - g.min_word_length
                    if g.one_span_hack:
                        transitions.append(
                            (prev_v + E_mask + log_probs[:, start_idx, end_idx].align_as(prev_v)) + span_bias)
                    else:
                        e_trans = (prev_v + log_probs[:, start_idx, end_idx].align_as(prev_v)) + span_bias
                        transitions.append(e_trans)
                        if g.use_posterior_reg and self.training:
                            tmp.extend([ctc[prev_l] + math.log(word_len), pr[prev_l]])
                            another_tmp.append(span_bias + log_probs[:, start_idx, end_idx].expand_as(prev_v))
                            # tmp = torch.stack([ctc[prev_l] + math.log(word_len), pr[prev_l]],
                            #                   new_name='stacked').logsumexp(dim='stacked')
                            # pr_trans.append(span_bias + log_probs[:, start_idx, end_idx].align_as(prev_v) + tmp)
                except (KeyError, IndexError):
                    transitions.append(get_init())
                    if g.use_posterior_reg and self.training:
                        tmp.extend([get_init(), get_init()])
                        another_tmp.append(get_init())
                        # pr_trans.append(get_init())

            if g.use_posterior_reg and self.training:
                tmp = torch.stack(tmp, new_name='stacked').unflatten(
                    'stacked', [('new_tag', g.max_word_length - g.min_word_length + 1), ('tmp', 2)])
                tmp = tmp.logsumexp(dim='tmp')
                pr_trans.append(torch.stack(another_tmp, new_name='new_tag') + tmp)
            # Sum up all alignments.
            if self.training:
                if g.best_ctc:
                    ctc[l] = torch.stack(transitions, new_name='new_tag').max(dim='tag')[0].rename(new_tag='tag')
                else:
                    ctc[l] = torch.stack(transitions, new_name='new_tag').logsumexp(dim='tag').rename(new_tag='tag')
                    if g.use_posterior_reg and self.training:
                        pr[l] = torch.cat(pr_trans, dim='new_tag').logsumexp(dim='tag').rename(new_tag='tag')
                        # pr[l] = torch.stack(pr_trans, new_name='new_tag').logsumexp(dim='tag').rename(new_tag='tag')
            else:
                best_value, best_prev_tag = torch.stack(transitions, new_name='new_tag').max(dim='tag')
                ctc[l] = best_value.rename(new_tag='tag')
                bookkeeper.best_prev_tags[l] = best_prev_tag.rename(new_tag='tag')

        ctc = torch.stack([ctc[l] for l in range(max_length + 1)], new_name='length')
        expected_num_spans = None
        with NoName(ctc, lengths):
            final_nodes = ctc[range(batch_size), :, lengths]
            final_nodes.rename_('batch', 'tag')

        if g.best_ctc or not self.training:
            final_score = final_nodes.max(dim='tag')[0]
        else:
            final_score = final_nodes.logsumexp(dim='tag')

        if g.use_posterior_reg and self.training:
            pr = torch.stack([pr[l] for l in range(max_length + 1)], new_name='length')
            with NoName(pr, lengths):
                pr_nodes = pr[range(batch_size), :, lengths]
                expected_num_spans = (pr_nodes.logsumexp(dim=-1) - final_score).exp()
        return CtcReturn(final_nodes, final_score, bookkeeper, expected_num_spans)
