import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from cltk.phonology.old_english.orthophonology import \
    OldEnglishOrthophonology as oe
from torch.optim import SGD, Adagrad, Adam

from dev_misc import g
from dev_misc.arglib import add_argument, set_argument
from dev_misc.trainlib import has_gpus
from dev_misc.utils import deprecated
from xib.aligned_corpus.corpus import AlignedCorpus
from xib.aligned_corpus.data_loader import BaseAlignedBatch, DataLoaderRegistry
from xib.aligned_corpus.ipa_sequence import IpaUnit
from xib.aligned_corpus.transcriber import (BaseTranscriber,
                                            DictionaryTranscriber,
                                            MultilingualTranscriber,
                                            PhonemizerTranscriber,
                                            RuleBasedTranscriber,
                                            SimpleTranscriberFactory,
                                            TranscriberWithBackoff)
from xib.batch import convert_to_dense
from xib.ipa import Category, should_include
from xib.model.extract_model import ExtractModel, ExtractModelV2
from xib.training.evaluator import AlignedExtractEvaluator
from xib.training.task import ExtractTask
from xib.training.trainer import ExtractTrainer


class BaseManager(ABC):

    add_argument('saved_model_path', dtype='path', msg='Path to a saved model, skipping the local training phase.')

    @abstractmethod
    def run(self): ...


class ExtractManager(BaseManager):

    # IDEA(j_luo) when to put this in manager/trainer? what about scheduler? annealing? restarting? Probably all in trainer -- you need to track them with pbars.
    add_argument('optim_cls', default='adam', dtype=str, choices=['adam', 'adagrad', 'sgd'], msg='Optimizer class.')
    add_argument('lost_lang', dtype=str)
    add_argument('known_lang', dtype=str)
    add_argument('anneal_factor', default=0.5, dtype=float, msg='Mulplication value for annealing.')
    add_argument('aligner_lr', default=0.1, dtype=float)
    add_argument('num_rounds', default=1000, dtype=int, msg='Number of rounds')
    add_argument('use_oracle', default=False, dtype=bool)
    add_argument('anneal_baseline', default=False, dtype=bool)
    add_argument('use_full_oracle', default=False, dtype=bool)
    add_argument('init_baseline', default=0.05, dtype=float)
    add_argument('max_baseline', default=1.0, dtype=float)
    add_argument('align_mode', default='reg', choices=['init', 'reg'], dtype=str)
    add_argument('evaluate_only', default=False, dtype=bool)
    add_argument('embedding_only', default=False, dtype=bool)
    add_argument('conv_only', default=False, dtype=bool)
    add_argument('use_new_model', dtype=bool, default=False)
    add_argument('vocab_path', dtype='path',
                 msg='Path to a vocabulary file.')

    _name2cls = {'adam': Adam, 'adagrad': Adagrad, 'sgd': SGD}

    def __init__(self):
        if g.evaluate_only:
            ckpts = list()
            for path in Path(g.saved_model_path).glob('*latest'):
                name = path.name
                ckpts.append(int(re.match(r'saved\.0_(\d+)', name).group(1)))
            latest_ckpt = max(ckpts)

            saved_path = f'{g.saved_model_path}/saved.0_{latest_ckpt}.latest'
            saved = torch.load(saved_path)

            g.load_state_dict(saved['g'], keep_new=True)
            set_argument('saved_model_path', saved_path, _force=True)

        train_task = ExtractTask(training=True)
        eval_task = ExtractTask(training=False)
        self.dl_reg = DataLoaderRegistry()
        logging.info('Preparing training data loader.')
        self.dl_reg.register_data_loader(train_task, g.data_path)
        logging.info('Preparing dev data loader.')
        self.dl_reg.register_data_loader(eval_task, g.data_path)

        lu_size = ku_size = None
        char_sets = self.dl_reg[train_task].dataset.corpus.char_sets
        lcs = char_sets[g.lost_lang]
        vocab = BaseAlignedBatch.known_vocab
        kcs = vocab.char_set
        if g.input_format == 'text':
            lu_size = len(lcs)
            ku_size = len(kcs)

        logging.info('Initializing model.')
        model_cls = ExtractModelV2 if g.use_new_model else ExtractModel
        self.model = model_cls(lu_size, ku_size, vocab)
        # HACK(j_luo)
        from xib.aligned_corpus.ipa_sequence import IpaSequence

        def align(lost_char, known_char):
            try:
                lost_id = lcs.unit2id[lost_char]
                if g.use_feature_aligner:
                    assert g.align_mode == 'init', 'reg mode for this not supported'
                    dfms = convert_to_dense(IpaSequence(known_char).feature_matrix.rename(
                        'length', 'feat_group').align_to('length', 'batch', 'feat_group'))
                    for cat in Category:
                        if should_include(g.feat_groups, cat):
                            self.model.feat_aligner.embs[cat.name].data[lost_id].copy_(dfms[cat][0, 0] * 5.0)
                else:
                    known_id = kcs.unit2id[IpaUnit.from_str(g.known_lang, known_char)]
                    if g.align_mode == 'init':
                        self.model.unit_aligner.weight.data[lost_id, known_id] = 10.0
                    else:
                        self.model.align_units.append((lost_id, known_id))
                    # self.model.unit_aligner.weight.data[lost_id, known_id] = 5.0
            except KeyError:
                pass

        # FIXME(j_luo) Move this part to another file.
        if g.use_oracle or g.use_full_oracle:
            logging.imp('Testing some oracle.')
            if g.known_lang in ['ang']:
                oracle = [
                    ('k', 'k'),
                    ('k', 't͡ʃ'),
                    ('l', 'l'),
                    ('m', 'm'),
                    ('n', 'n'),
                    ('p', 'p'),
                    ('s', 's'),
                    ('t', 't')
                ]
            elif g.known_lang in ['pgm', 'non']:
                oracle = [
                    ('k', 'k'),
                    ('l', 'l'),
                    ('m', 'm'),
                    ('n', 'n'),
                    ('p', 'p'),
                    ('s', 's'),
                    ('t', 't')
                ]
            elif g.known_lang == 'lat':
                oracle = [
                    ('a', 'a'),
                    ('b', 'b'),
                    # ('d', 'd'),
                    # ('i', 'i'),
                    ('k', 'k'),
                    # ('k', 't͡ʃ'),
                    ('l', 'l'),
                    ('m', 'm'),
                    ('n', 'n'),
                    # ('o', 'o'),
                    # ('p', 'p'),
                    ('r', 'r'),
                    ('s', 's'),
                    ('t', 't'),
                    # ('g', 'g')

                    # ('þ', 'h'),
                    # ('i', 'r'),
                ]
            elif g.known_lang == 'eu':
                pass
            else:
                raise ValueError

            if g.use_full_oracle:
                if g.known_lang == 'pgm':
                    oracle = [
                        ('b', 'b'),
                        ('b', 'β'),
                        ('d', 'd'),
                        ('d', 'ð'),
                        ('e', 'a'),
                        ('f', 'ɸ'),
                        ('h', 'x'),
                        ('h', 'h'),
                        ('i', 'i'),
                        ('i', 'e'),
                        ('j', 'j'),
                        ('k', 'k'),
                        ('l', 'l'),
                        ('m', 'm'),
                        ('n', 'n'),
                        ('o', 'o'),
                        ('o', 'u'),
                        ('p', 'p'),
                        ('r', 'r'),
                        ('s', 's'),
                        ('s', 'z'),
                        ('t', 't'),
                        ('u', 'u'),
                        ('w', 'w'),
                        ('z', 'z'),
                        ('g', 'ŋ'),
                        ('a', 'a'),
                        ('g', 'ɣ'),
                        ('g', 'g'),
                        ('g', 'ɡ'),
                        ('þ', 'ð'),
                        ('þ', 'θ')
                    ]
                elif g.known_lang == 'ang':
                    oracle = [
                        ('b', 'b'),
                        ('d', 'd'),
                        ('e', 'e'),
                        ('f', 'f'),
                        ('h', 'h'),
                        ('h', 'x'),
                        ('i', 'i'),
                        ('i', 'e'),
                        ('j', 'j'),
                        ('k', 'k'),
                        ('k', 't͡ʃ'),
                        ('k', 'ʃ'),
                        ('l', 'l'),
                        ('m', 'm'),
                        ('n', 'n'),
                        ('o', 'o'),
                        ('o', 'u'),
                        ('p', 'p'),
                        ('r', 'r'),
                        ('s', 's'),
                        ('s', 'z'),
                        ('t', 't'),
                        ('u', 'o'),
                        ('u', 'u'),
                        ('u', 'y'),
                        ('w', 'w'),
                        ('x', 'g'),
                        ('z', 'r'),
                        ('z', 's'),
                        ('z', 'z'),
                        ('g', 'ŋ'),
                        ('a', 'e'),
                        ('a', 'æ'),
                        ('a', 'ɑ'),
                        ('g', 'ɡ'),
                        ('g', 'j'),
                        ('g', 'ɡ'),
                        ('g', 'j'),
                        ('g', 'ɣ'),
                        ('þ', 'ð'),
                        ('þ', 'θ'),
                        ('ƕ', 'h')
                    ]
                elif g.known_lang == 'non':
                    oracle = [
                        ('b', 'b'),
                        ('d', 'd'),
                        ('d', 'ð'),
                        ('e', 'a'),
                        ('e', 'e'),
                        ('e', 'o'),
                        ('e', 'ø'),
                        ('e', 'ɛ'),
                        ('f', 'f'),
                        ('h', 'h'),
                        ('i', 'i'),
                        ('i', 'e'),
                        ('i', 'ø'),
                        ('i', 'y'),
                        ('j', 'j'),
                        ('k', 'k'),
                        ('l', 'l'),
                        ('m', 'm'),
                        ('n', 'n'),
                        ('o', 'o'),
                        ('o', 'y'),
                        ('o', 'ø'),
                        ('o', 'œ'),
                        ('p', 'p'),
                        ('r', 'r'),
                        ('s', 's'),
                        ('s', 'r'),
                        ('t', 't'),
                        ('u', 'o'),
                        ('u', 'u'),
                        ('u', 'y'),
                        ('w', 'v'),
                        ('x', 'g'),
                        ('z', 'r'),
                        ('z', 's'),
                        ('g', 'n'),
                        ('g', 'g'),
                        ('a', 'a'),
                        ('a', 'e'),
                        ('a', 'o'),
                        ('a', 'ø'),
                        ('a', 'œ'),
                        ('a', 'ɒ'),
                        ('g', 'ɣ'),
                        ('g', 'g'),
                        ('g', 'ɡ'),
                        ('þ', 'ð'),
                        ('þ', 'θ')
                    ]
                elif g.known_lang == 'eu':
                    oracle = [
                        ('a', 'ɑ'),
                        ('e', 'ɛ'),
                        ('i', 'i'),
                        ('o', 'o'),
                        ('u', 'u'),

                        ('d', 'd'),

                        ('f', 'f'),
                        ('f', 'b'),

                        ('g', 'ɡ'),
                        ('c', 'ɡ'),

                        ('h', 'h'),
                        ('h', 'k'),

                        ('l', 'l'),
                        ('l', 'r'),
                        ('l', 'n'),
                        ('r', 'l'),
                        ('r', 'r'),
                        ('r', 'n'),
                        ('n', 'l'),
                        ('n', 'n'),
                        ('n', 'r'),

                        ('b', 'b'),
                        ('b', 'm'),
                        ('v', 'b'),
                        ('v', 'm'),
                        ('m', 'b'),
                        ('m', 'm'),

                        ('m', 'n'),

                        ('p', 'p'),

                        ('s', 's'),
                        ('z', 's'),

                        ('t', 't'),

                        ('c', 's'),
                        ('c', 'k'),
                        ('c', 'g'),
                    ]
            for l, k in oracle:
                align(l, k)

        if has_gpus():
            self.model.cuda()
        logging.info(str(self.model))

        self.evaluator = AlignedExtractEvaluator(self.model, self.dl_reg[eval_task], BaseAlignedBatch.known_vocab)

        self.trainer = ExtractTrainer(self.model, [train_task], [1.0], 'total_step',
                                      stage_tnames=['round', 'total_step'],
                                      evaluator=self.evaluator,
                                      check_interval=g.check_interval,
                                      eval_interval=g.eval_interval,
                                      save_interval=g.save_interval)
        if g.saved_model_path:
            self.trainer.load(g.saved_model_path)

    def run(self):
        # HACK(j_luo)
        self.trainer.bij_reg = 0.0
        self.trainer.ent_reg = 0.0
        self.trainer.global_baseline = g.init_baseline + 1e-8
        optim_cls = self._name2cls[g.optim_cls]
        if g.anneal_temperature:
            self.trainer.temperature = g.init_temperature
        else:
            self.trainer.temperature = g.temperature
        if g.anneal_pr_hyper:
            self.trainer.pr_hyper = g.init_pr_hyper
        else:
            self.trainer.pr_hyper = g.pr_hyper
        if g.anneal_context_weight:
            self.trainer.context_weight = g.start_context_weight
        else:
            self.trainer.context_weight = g.context_weight

        # , momentum=0.9, nesterov=True)
        if g.use_feature_aligner:
            self.trainer.optimizer = optim_cls([
                {'params': self.model.feat_aligner.parameters(), 'lr': g.aligner_lr},
                {'params': [param for name, param in self.model.named_parameters() if 'feat_aligner' not in name]}
            ], lr=g.learning_rate)
        elif g.embedding_only:
            self.trainer.optimizer = optim_cls(self.model.base_embeddings.parameters(), lr=g.learning_rate)
        elif g.conv_only:
            from itertools import chain
            self.trainer.optimizer = optim_cls(
                chain(self.model.conv.parameters(), self.model.ins_conv.parameters()), lr=g.learning_rate)
        else:
            self.trainer.optimizer = optim_cls([
                {'params': self.model.unit_aligner.parameters(), 'lr': g.aligner_lr},
                {'params': [param for name, param in self.model.named_parameters() if 'unit_aligner' not in name]}
            ], lr=g.learning_rate)

        self.trainer.er = g.init_expected_ratio

        if g.evaluate_only:
            self.trainer.ins_del_cost = g.min_ins_del_cost
            self.trainer.model.eval()
            self.trainer.evaluate()

        else:
            # Save init parameters.
            out_path = g.log_dir / f'saved.init'
            self.trainer.save_to(out_path)
            for _ in range(g.num_rounds):
                self.trainer.reset()

                self.trainer.train(self.dl_reg)
                self.trainer.tracker.update('round')
