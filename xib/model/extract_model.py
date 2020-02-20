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
from dev_misc.utils import (WithholdKeys, cached_property, concat_lists,
                            global_property)
from xib.aligned_corpus.char_set import DELETE_ID
from xib.aligned_corpus.corpus import AlignedSentence
from xib.aligned_corpus.data_loader import AlignedBatch, AlignedDataLoader
from xib.aligned_corpus.vocabulary import Vocabulary
from xib.data_loader import (ContinuousIpaBatch, UnbrokenTextBatch,
                             convert_to_dense)
from xib.ipa import Category, Index, get_enum_by_cat, should_include
from xib.ipa.process import (Segment, Segmentation, SegmentWindow, SegmentX,
                             Span)
from xib.model.log_tensor import LogTensor
from xib.model.modules import AdaptLayer, FeatEmbedding
from xib.model.span_proposer import (AllSpanProposer, OracleStemSpanProposer,
                                     OracleWordSpanProposer, SpanBias,
                                     ViableSpans)

from .modules import DenseFeatEmbedding

ExtractBatch = Union[ContinuousIpaBatch, UnbrokenTextBatch]


@batch_class
class Matches(BaseBatch):
    ll: FT
    f: FT  # All these dp scores.
    raw_ll: FT = None  # HACK(j_luo)


@batch_class
class EmbAndRepr(BaseBatch):
    known_unit_emb: FT
    known_ctx_repr: FT
    known_ins_ctx_repr: FT
    lost_unit_emb: FT


@batch_class
class Extracted(BaseBatch):
    batch_size: int
    matches: Matches
    viable_spans: ViableSpans


@batch_class
class Costs(BaseBatch):
    sub: FT
    ins: FT
    alignment: FT


@batch_class
class CtcBookkeeper(BaseBatch):
    best_vocab: LT
    best_prev_tags: Dict[int, FT] = field(init=False, default_factory=dict)


@batch_class
class CtcReturn(BaseBatch):
    final_nodes: FT
    final_score: FT
    expected_num_spans: FT
    expected_avg_log_probs: FT
    bookkeeper: Optional[CtcBookkeeper] = None


@batch_class
class ExtractModelReturn(BaseBatch):
    emb_repr: EmbAndRepr
    extracted: Extracted
    costs: Costs
    ctc_return: CtcReturn


class ExtractModel(nn.Module):

    add_argument('g2p_window_size', default=3, dtype=int, msg='Window size for g2p layer.')
    add_argument('top_k_predictions', default=10, dtype=int, msg='Number of top predictions to keep.')
    add_argument('max_word_length', default=10, dtype=int, msg='Max length of extracted words.')
    add_argument('init_ins_del_cost', default=100, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('min_ins_del_cost', default=3.5, dtype=float, msg='Initial unit cost for insertions and deletions.')
    add_argument('context_weight', default=0.0, dtype=float, msg='Weight for the context probabilities.')
    add_argument('context_agg_mode', default='log_interpolation', dtype=str,
                 choices=['log_interpolation', 'linear_interpolation', 'log_add', 'dilute'])
    add_argument('dilute_hyper', dtype=float, default=0.5)
    add_argument('debug', dtype=bool, default=False, msg='Flag to enter debug mode.')
    add_argument('use_base_embedding', dtype=bool, default=False,
                 msg='Flag to use base embeddings to compute character embeddings.')
    add_argument('base_embedding_dim', dtype=int, default=250, msg='Dimensionality for base embeddings.')
    add_argument('span_candidates', dtype=str,
                 choices=['all', 'oracle_word', 'oracle_stem'], default='all', msg='How to generate candidates for spans.')
    add_argument('one2two', dtype=bool, default=False, msg='Use conv on both sides.')
    add_argument('cut_off', dtype=float, nargs='+')
    add_argument('dense_embedding', dtype=bool, default=False)
    add_argument('use_entropy_reg', dtype=bool, default=False)
    add_argument('expected_ratio', dtype=float, default=0.2)
    add_argument('init_expected_ratio', dtype=float, default=1.0)
    add_argument('baseline', dtype=float)
    add_argument('positive_reward_only', dtype=bool, default=False)
    add_argument('use_log_tensor', dtype=bool, default=False)
    add_argument('inference_mode', dtype=str, default='mixed', choices=['new', 'old', 'mixed'])
    add_argument('thresh_func', dtype=str, default='linear', choices=['linear', 'trunc'])

    def __init__(self, lost_size: int, known_size: int, vocab: Vocabulary):
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
        self.conv = nn.Conv1d(self.dim, self.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)
        self.ins_conv = nn.Conv1d(self.dim, self.dim, g.g2p_window_size, padding=g.g2p_window_size // 2)
        self.dropout = nn.Dropout(g.dropout)
        opt2sp_cls = {'all': AllSpanProposer,
                      'oracle_word': OracleWordSpanProposer,
                      'oracle_stem': OracleStemSpanProposer}
        sp_cls = opt2sp_cls[g.span_candidates]
        self.span_proposer = sp_cls()
        if g.use_base_embedding:
            if g.dense_embedding:
                self.base_embeddings = DenseFeatEmbedding('feat_emb', 'feat_group', 'char_emb', dim=g.dim)
            else:
                self.base_embeddings = nn.Embedding(60, g.base_embedding_dim)
                logging.imp('Base embeddings initialized uniformly.')
                nn.init.uniform_(self.base_embeddings.weight, -0.05, 0.05)

        self.span_bias = SpanBias()
        self.vocab = vocab

        if g.use_log_tensor:
            assert g.inference_mode == 'old'

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

    def forward(self, batch: AlignedBatch) -> ExtractModelReturn:
        vocab = batch.known_vocab
        # Get relevant representations -- one for all the known units, and one for a contextualized representation.
        emb_repr = self._get_repr(batch)

        # Get cost matrix.
        costs = self._get_costs(vocab.indexed_segments, emb_repr)

        # Get span candidates:
        viable_spans = self.span_proposer(batch.unit_id_seqs, batch.lengths, batch.sentences)

        # Get matches.
        matches = self._get_matches(costs,
                                    viable_spans,
                                    vocab.vocab_length)

        # Get extracted object.
        extracted = self._get_extracted(matches, viable_spans, len(vocab))

        # Prepare return.
        model_ret = self._prepare_return(batch, extracted, costs, emb_repr)

        return model_ret

    def get_known_unit_emb(self, vocab: Vocabulary) -> FT:
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
                    known_unit_emb = known_unit_emb / \
                        (1e-8 + known_unit_emb.norm(dim=-1, keepdim=True)) * math.sqrt(5)
        known_unit_emb = known_unit_emb.rename('known_unit', 'length', 'char_emb').squeeze(dim='length')
        return known_unit_emb

    def _get_repr(self, batch: AlignedBatch) -> EmbAndRepr:
        vocab = batch.known_vocab
        known_unit_emb = self.get_known_unit_emb(vocab)

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

        lost_unit_emb = self.get_lost_unit_emb(known_unit_emb)

        emb_repr = EmbAndRepr(known_unit_emb, known_ctx_repr, known_ins_ctx_repr, lost_unit_emb)
        return emb_repr

    def get_lost_unit_emb(self, known_unit_emb: FT) -> FT:
        lost_unit_weight = self.unit_aligner.weight
        lost_unit_emb = lost_unit_weight @ known_unit_emb
        return lost_unit_emb

    def _get_alignment_log_probs(self, known_unit_emb: FT, lost_unit_emb: FT, reverse: bool = False) -> FT:
        unit_logits = known_unit_emb @ lost_unit_emb.t()
        dim = 0 if reverse else -1
        unit_log_probs = unit_logits.log_softmax(dim=dim)
        return unit_log_probs

    def get_alignment(self, reverse: bool = False) -> FT:
        known_unit_emb = self.get_known_unit_emb(self.vocab)
        lost_unit_emb = self.get_lost_unit_emb(known_unit_emb)
        unit_log_probs = self._get_alignment_log_probs(known_unit_emb, lost_unit_emb, reverse=reverse)
        return unit_log_probs.exp()

    def _get_costs(self, vocab_unit_id_seqs: LT, emb_repr: EmbAndRepr) -> Costs:
        # Get global (non-contextualized) log probs.
        unit_log_probs = self._get_alignment_log_probs(emb_repr.known_unit_emb, emb_repr.lost_unit_emb)
        with NoName(unit_log_probs, vocab_unit_id_seqs):
            global_log_probs = unit_log_probs[vocab_unit_id_seqs].rename('vocab', 'length', 'lost_unit')
        # Compute alignment -- set reverse to True if entropy regularization is used.
        if g.use_entropy_reg:
            rev_unit_log_probs = self._get_alignment_log_probs(emb_repr.known_unit_emb,
                                                               emb_repr.lost_unit_emb,
                                                               reverse=True)
            alignment = rev_unit_log_probs.exp()
        else:
            alignment = unit_log_probs.exp()

        # Get contextualized log probs.
        sub_ctx_logits = emb_repr.known_ctx_repr @ emb_repr.lost_unit_emb.t()
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
        ins_ctx_logits = emb_repr.known_ins_ctx_repr @ emb_repr.lost_unit_emb.t()
        ins_ctx_log_probs = ins_ctx_logits.log_softmax(dim=-1).rename('vocab', 'length', 'lost_unit')
        ins_weighted_log_probs = interpolate(ins_ctx_log_probs, global_log_probs)
        ins = -ins_weighted_log_probs + self.ins_del_cost

        costs = Costs(sub, ins, alignment)

        return costs

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
                    del_cost = costs.sub[:, i - 1, DELETE_ID].unsqueeze(dim=-1)
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
        with NoName(bi, lsi, lei, viable, matches.ll):
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

        extracted = Extracted(batch_size,
                              matches,
                              viable_spans)
        return extracted

    def _update_fs(self, fs: Dict[Tuple[int, int], FT], costs: Costs, viable_spans: ViableSpans, kl: int, ll: int):
        transitions = list()
        if (kl - 1, ll) in fs:
            if g.one2two:
                del_cost = costs.sub[:, kl - 1, DELETE_ID].unsqueeze(dim=-1)
            else:
                del_cost = self.ins_del_cost
            transitions.append(fs[(kl - 1, ll)] + del_cost)
        if not g.one2two and (kl, ll - 1) in fs:
            ins_cost = self.ins_del_cost
            transitions.append(fs[(kl, ll - 1)] + ins_cost)
        if g.one2two and (kl - 1, ll - 2) in fs:
            first_lost_ids = viable_spans.unit_id_seqs[:, ll - 2]
            sub_cost = costs.sub[:, kl - 1, first_lost_ids]
            second_lost_ids = viable_spans.unit_id_seqs[:, ll - 1]
            ins_cost = costs.ins[:, kl - 1, second_lost_ids]
            transitions.append(fs[(kl - 1, ll - 2)] + ins_cost + sub_cost)
        if (kl - 1, ll - 1) in fs:
            lost_ids = viable_spans.unit_id_seqs[:, ll - 1]
            sub_cost = costs.sub[:, kl - 1, lost_ids]
            if g.one2two:
                sub_cost = sub_cost
            transitions.append(fs[(kl - 1, ll - 1)] + sub_cost)
        if transitions:
            all_s = torch.stack(transitions, dim=-1)
            new_s, _ = all_s.min(dim=-1)
            fs[(kl, ll)] = new_s

    @property
    def lp_per_unmatched(self) -> float:
        return math.log(self.p_unmatched)

    @property
    def p_unmatched(self) -> float:
        return 1.0 / self.unit_aligner.num_embeddings

    @property
    def baseline(self) -> float:
        if g.baseline is None:
            return self.lp_per_unmatched
        else:
            return math.log(g.baseline)

    def _prepare_return(self, batch: AlignedBatch, extracted: Extracted, costs: Costs, emb_repr: EmbAndRepr) -> ExtractModelReturn:
        # Get the best score and span.
        # NOTE(j_luo) Some segments don't have any viable spans.
        matches = extracted.matches
        viable_spans = extracted.viable_spans
        not_viable_mask = (~viable_spans.viable).expand_as(matches.ll)
        viable_ll = matches.ll + (-9999.9) * not_viable_mask.float()
        # Get marginal.
        span_log_probs = viable_ll.logsumexp(dim='vocab')
        span_raw_square = raw_reward = None
        span_lengths = get_named_range(self.num_e_tags, 'len_e') + g.min_word_length
        if g.inference_mode == 'new':
            span_lengths = span_lengths.align_as(matches.raw_ll)
            raw_reward = self._get_thresholded_reward(matches.raw_ll / span_lengths)
            span_raw_square = (raw_reward + matches.raw_ll).logsumexp(dim='vocab') - math.log(len(self.vocab))
        else:
            raw_reward = self._get_thresholded_reward(span_log_probs / span_lengths)
        ctc_return = self._run_ctc(batch.lengths, span_log_probs, matches.ll,
                                   matches.raw_ll, span_raw_square, raw_reward)
        # ctc_return = self._run_ctc_v1(batch.lengths, span_log_probs, matches.ll, matches.raw_ll)
        model_ret = ExtractModelReturn(emb_repr,
                                       extracted,
                                       costs,
                                       ctc_return)
        return model_ret

    @global_property
    def cut_off(self):
        pass

    @property
    def num_e_tags(self) -> int:
        return g.max_word_length - g.min_word_length + 1

    @property
    def num_tags(self) -> int:
        return self.num_e_tags + 1

    def _get_thresholded_reward(self, raw: FT) -> Union[FT, LogTensor]:
        diff = raw - self.baseline
        if g.thresh_func == 'linear':
            reward = diff
        elif g.thresh_func == 'trunc':
            reward = diff.clamp_max(math.log(5.0))
        if g.positive_reward_only:
            pos = diff > 0
            reward = torch.where(pos,
                                 reward,
                                 torch.full_like(raw, -9999.9))
        if g.use_log_tensor:
            reward = LogTensor.from_torch(reward, log_scale=True)
        return reward

    # @profile
    def _run_ctc(self, lengths: LT, span_log_probs: FT, vocab_log_probs: FT, raw_vocab_log_probs: FT, span_raw_square: FT, raw_reward: FT) -> CtcReturn:
        r"""To speed up DP, everything is packed into tensors.

        tag \ case
        -------------------------------------------------------------
        O           (l-1, O)    (l-1, E_m)      ...     (l-1, E_M)
        E_m         (l-m, O)  (l-m, E_m)    ...     (l-m, E_M)
        E_m+1       (l-m-1, O)    (l-m-1, E_m)      ...     (l-m-1, E_M)
        ...
        E_M         (l-M, O)  (l-M, E_m)    ...     (l-M, E_M)
        """

        # ------------------ Prepare everything first. ----------------- #

        marginal = dict()
        psi = dict()
        phi = dict()

        max_length = lengths.max().item()
        batch_size = span_log_probs.size('batch')
        bookkeeper = None
        if self.training:
            log_probs = span_log_probs
        else:
            log_probs, best_vocab = vocab_log_probs.max(dim='vocab')
            bookkeeper = CtcBookkeeper(best_vocab)

            # NOTE(j_luo) Only consider words above a certain threshold.
            if g.cut_off is not None:
                warnings.warn('Only taking most confident ones.')
                len_range = get_named_range(log_probs.size('len_e'), 'len_e') + g.min_word_length
                avg = raw_vocab_log_probs.gather('vocab', best_vocab) / len_range.align_as(best_vocab)
                log_probs = torch.where(avg > self.cut_off, log_probs, torch.zeros_like(log_probs).fill_(-9999.9))
        if g.use_log_tensor:
            p_x_z = LogTensor.from_torch(log_probs, log_scale=True)

        def get_init(start: float, init_value: float = -9999.9):
            ret = get_zeros(batch_size, self.num_tags).fill_(init_value)
            ret[:, 0] = start
            ret.rename_('batch', 'tag')
            if g.use_log_tensor:
                ret = LogTensor.from_torch(ret, log_scale=True)
            return ret

        marginal[0] = get_init(0.0)
        if self.training:
            psi[0] = get_init(-20.0)
            phi[0] = get_init(-20.0)
        padding = get_init(-9999.9)

        def expand_vs(t):
            return t.align_to(..., 'vocab').expand(batch_size, self.num_tags, len(self.vocab))

        padding_vs = expand_vs(padding)

        non_span_bias = self.span_bias(False)
        span_biases = self.span_bias(True)
        if g.use_log_tensor:
            non_span_bias = LogTensor.from_torch(non_span_bias, log_scale=True)
            span_biases = {l: LogTensor.from_torch(b, log_scale=True) for l, b in span_biases.items()}

        # ------------------------- Main body. ------------------------- #
        all_phi_scores = dict()
        all_phi_scores[0] = get_init(0.0)  # FIXME(j_luo) This is wrong for log_tensors.
        nh = NameHelper()
        for l in range(1, max_length + 1):
            marginal_tags = list()
            phi_tags = list()
            psi_tags = list()
            phi_scores = list()
            # Case 'O'.
            if g.use_log_tensor:
                non_span_const = self.p_unmatched * non_span_bias
                marginal_tags.append(marginal[l - 1] * non_span_const)
            else:
                non_span_const = self.lp_per_unmatched + non_span_bias
                marginal_tags.append(marginal[l - 1] + non_span_const)
            if self.training:
                if g.use_log_tensor:
                    psi_tags.append((psi[l - 1] * non_span_const).align_to('batch', 'tag', 'new_tag'))
                    phi_tags.append((phi[l - 1] * non_span_const).align_to('batch', 'tag', 'new_tag'))
                else:
                    psi_tags.append((psi[l - 1] + non_span_const).align_to('batch', 'tag', 'new_tag'))
                    phi_tags.append((phi[l - 1] + non_span_const).align_to('batch', 'tag', 'new_tag'))
            else:
                if g.inference_mode == 'new':
                    phi_scores.append(expand_vs(all_phi_scores[l - 1]))
                elif g.inference_mode == 'mixed':
                    phi_scores.append(all_phi_scores[l - 1])
            # Case 'E's.
            phi_lse = list()
            psi_lse = list()
            phi_psi_common = list()
            for word_len in range(g.min_word_length, g.max_word_length + 1):
                span_bias = span_biases[word_len]
                prev_l = l - word_len
                if prev_l in marginal:
                    prev_marginal = marginal[prev_l]
                    # FIXME(j_luo)  This is wrong if oracle spans are used.
                    start_idx = prev_l
                    end_idx = word_len - g.min_word_length
                    if g.use_log_tensor:
                        this_p_x_z = p_x_z[:, start_idx, end_idx].expand_as(prev_marginal)
                        marginal_tags.append(prev_marginal * this_p_x_z * span_bias)
                    else:
                        lp = log_probs[:, start_idx, end_idx].expand_as(prev_marginal)
                        marginal_tags.append(prev_marginal + lp + span_bias)

                    # reward = lp / word_len
                    # reward = self._get_thresholded_reward(reward)
                    reward = raw_reward[:, start_idx, end_idx].expand_as(prev_marginal)

                    if self.training:
                        if g.inference_mode == 'new':
                            phi_psi_common.append(torch.full_like(lp, span_bias))
                            srw = span_raw_square[:, start_idx, end_idx].expand_as(prev_marginal)
                            phi_lse.extend([prev_marginal + srw, phi[prev_l] + lp])
                            psi_lse.extend([prev_marginal + math.log(word_len) + lp, psi[prev_l] + lp])
                        else:
                            if g.use_log_tensor:
                                phi_psi_common.append(span_bias * this_p_x_z)
                                phi_lse.extend([prev_marginal * reward, phi[prev_l]])
                                psi_lse.extend([prev_marginal * word_len, psi[prev_l]])
                            else:
                                phi_psi_common.append(span_bias + lp)
                                # Compute phi-related scores.
                                phi_lse.extend([prev_marginal + reward, phi[prev_l]])
                                # Compute psi-related scores.
                                psi_lse.extend([prev_marginal + math.log(word_len), psi[prev_l]])
                    else:
                        if g.inference_mode == 'new':
                            rr = raw_reward[:, start_idx, end_idx].align_to('batch', 'tag', 'vocab')
                            phi_scores.append(all_phi_scores[prev_l].align_as(rr) + rr.exp())
                        elif g.inference_mode == 'mixed':
                            phi_scores.append(all_phi_scores[prev_l] + reward.exp())
                else:
                    marginal_tags.append(padding)
                    if self.training:
                        phi_psi_common.append(padding)
                        phi_lse.extend([padding, padding])
                        psi_lse.extend([padding, padding])
                    else:
                        if g.inference_mode == 'new':
                            phi_scores.append(padding_vs)
                        elif g.inference_mode == 'mixed':
                            phi_scores.append(padding)

            if self.training:
                if g.use_log_tensor:
                    all_lse = phi_lse + psi_lse
                    part_1 = all_lse[::2]
                    part_2 = all_lse[1::2]
                    part_1 = LogTensor.stack(part_1, new_name='new_tag')
                    part_2 = LogTensor.stack(part_2, new_name='new_tag')
                    phi_psi_lse = part_1 + part_2
                else:
                    all_stacked = torch.stack(phi_lse + psi_lse, new_name='stacked')
                    flat_all_stacked = all_stacked.unflatten('stacked', [('new_tag', self.num_e_tags * 2), ('tmp', 2)])
                    phi_psi_lse = flat_all_stacked.logsumexp(dim='tmp').align_to('batch', 'tag', 'new_tag')
                assert phi_psi_lse.size('new_tag') % 2 == 0
                with NoName(phi_psi_lse):
                    phi_lse, psi_lse = phi_psi_lse.chunk(2, dim=2)
                    phi_lse.rename_('batch', 'tag', 'new_tag')
                    psi_lse.rename_('batch', 'tag', 'new_tag')

                if g.use_log_tensor:
                    stacked_common = LogTensor.stack(phi_psi_common, new_name='new_tag')
                    psi_tags.append(stacked_common * psi_lse)
                    phi_tags.append(stacked_common * phi_lse)
                else:
                    stacked_common = torch.stack(phi_psi_common, new_name='new_tag')
                    psi_tags.append(stacked_common + psi_lse)
                    phi_tags.append(stacked_common + phi_lse)

            # Sum up all alignments.
            if self.training:
                if g.use_log_tensor:
                    marginal[l] = LogTensor.stack(marginal_tags, new_name='new_tag').sum(
                        dim='tag').rename(new_tag='tag')
                    psi[l] = LogTensor.cat(psi_tags, dim='new_tag').sum(dim='tag').rename(new_tag='tag')
                    phi[l] = LogTensor.cat(phi_tags, dim='new_tag').sum(dim='tag').rename(new_tag='tag')
                else:
                    marginal[l] = torch.stack(marginal_tags, new_name='new_tag').logsumexp(
                        dim='tag').rename(new_tag='tag')
                    psi[l] = torch.cat(psi_tags, dim='new_tag').logsumexp(dim='tag').rename(new_tag='tag')
                    phi[l] = torch.cat(phi_tags, dim='new_tag').logsumexp(dim='tag').rename(new_tag='tag')
            else:
                # HACK(j_luo) New way.
                if g.inference_mode in ['new', 'mixed']:
                    phi_stacked = torch.stack(phi_scores, new_name='new_tag')
                    if g.inference_mode == 'new':
                        flat_phi = nh.flatten(phi_stacked, ['tag', 'vocab'], 'tag_X_vocab')
                        best_value, best_index = flat_phi.max(dim='tag_X_vocab')
                        best_prev_tag = best_index // len(self.vocab)
                        best_prev_vocab = best_index % len(self.vocab)
                    else:
                        best_value, best_prev_tag = phi_stacked.max(dim='tag')
                    all_phi_scores[l] = best_value.rename(new_tag='tag')
                    bookkeeper.best_prev_tags[l] = best_prev_tag.rename(new_tag='tag')
                # Old way here.
                if g.use_log_tensor:
                    best_value, best_prev_tag = LogTensor.stack(marginal_tags, new_name='new_tag').max(dim='tag')
                else:
                    best_value, best_prev_tag = torch.stack(marginal_tags, new_name='new_tag').max(dim='tag')
                marginal[l] = best_value.rename(new_tag='tag')
                if g.inference_mode == 'old':
                    bookkeeper.best_prev_tags[l] = best_prev_tag.rename(new_tag='tag')

        # ------------- Get the actual phi and psi values. ------------- #

        if g.use_log_tensor:
            marginal = LogTensor.stack([marginal[l] for l in range(max_length + 1)], new_name='length')
        else:
            marginal = torch.stack([marginal[l] for l in range(max_length + 1)], new_name='length')
        expected_num_spans = expected_avg_log_probs = None
        with NoName(marginal, lengths):
            final_nodes = marginal[range(batch_size), :, lengths]
            final_nodes.rename_('batch', 'tag')

        if not self.training:
            final_score = final_nodes.max(dim='tag')[0]
        else:
            if g.use_log_tensor:
                final_score = final_nodes.sum(dim='tag')
            else:
                final_score = final_nodes.logsumexp(dim='tag')

        if self.training:
            if g.use_log_tensor:
                psi = LogTensor.stack([psi[l] for l in range(max_length + 1)], new_name='length')
                phi = LogTensor.stack([phi[l] for l in range(max_length + 1)], new_name='length')
            else:
                psi = torch.stack([psi[l] for l in range(max_length + 1)], new_name='length')
                phi = torch.stack([phi[l] for l in range(max_length + 1)], new_name='length')
            with NoName(psi, phi, lengths):
                phi_nodes = phi[range(batch_size), :, lengths]
                psi_nodes = psi[range(batch_size), :, lengths]
                if g.use_log_tensor:
                    expected_avg_log_probs = (phi_nodes.sum(dim=-1) / final_score).value
                    expected_num_spans = (psi_nodes.sum(dim=-1) / final_score).value
                else:
                    expected_avg_log_probs = (phi_nodes.logsumexp(dim=-1) - final_score).exp()  # / expected_num_spans
                    expected_num_spans = (psi_nodes.logsumexp(dim=-1) - final_score).exp()
        # HACK(j_luo)
        if g.use_log_tensor:
            return CtcReturn(final_nodes, final_score.storage, expected_num_spans, expected_avg_log_probs, bookkeeper)
        else:
            return CtcReturn(final_nodes, final_score, expected_num_spans, expected_avg_log_probs, bookkeeper)
