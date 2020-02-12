import logging
import os
import random
import re
from abc import ABC, abstractmethod

import pandas as pd
import torch
import torch.nn as nn
from cltk.phonology.old_english.orthophonology import \
    OldEnglishOrthophonology as oe
from torch.optim import SGD, Adagrad, Adam
from torch.optim.lr_scheduler import CyclicLR

from dev_misc import g
from dev_misc.arglib import add_argument, init_g_attr
from dev_misc.trainlib import Metrics, has_gpus, set_random_seeds
from dev_misc.trainlib.trainer import freeze
from dev_misc.utils import deprecated
from xib.aligned_corpus.corpus import AlignedCorpus
from xib.aligned_corpus.data_loader import BaseAlignedBatch
from xib.aligned_corpus.transcriber import (BaseTranscriber,
                                            DictionaryTranscriber,
                                            MultilingualTranscriber,
                                            PhonemizerTranscriber,
                                            RuleBasedTranscriber,
                                            SimpleTranscriberFactory,
                                            TranscriberWithBackoff)
from xib.data_loader import (ContinuousTextDataLoader, DataLoaderRegistry,
                             DenseIpaDataLoader, IpaDataLoader)
from xib.model.decipher_model import DecipherModel
from xib.model.extract_model import ExtractModel
from xib.model.lm_model import LM, AdaptLM
from xib.search.search_solver import SearchSolver
from xib.search.searcher import BruteForceSearcher
from xib.training.evaluator import (AlignedExtractEvaluator, DecipherEvaluator,
                                    ExtractEvaluator, LMEvaluator,
                                    SearchSolverEvaluator)
from xib.training.task import (AdaptCbowTask, AdaptLMTask, CbowTask,
                               DecipherTask, ExtractTask, LMTask, MlmTask,
                               TransferTask)
from xib.training.trainer import (AdaptLMTrainer, DecipherTrainer,
                                  ExtractTrainer, LMTrainer)

add_argument('task', default='lm', dtype=str,
             choices=['lm', 'cbow', 'adapt_lm', 'adapt_cbow', 'decipher', 'search', 'extract', 'prepare'],
             msg='which task to run')


class BaseManager(ABC):

    @abstractmethod
    def run(self): ...


class LMManager(BaseManager):

    model_cls = LM
    trainer_cls = LMTrainer
    task_cls = LMTask

    def __init__(self):
        self.model = self.model_cls()
        if has_gpus():
            self.model.cuda()
        logging.info(self.model)

        task = self.task_cls()
        self.dl_reg = DataLoaderRegistry()
        self.dl_reg.register_data_loader(task, g.data_path)
        self.evaluator = LMEvaluator(self.model, self.dl_reg[task])
        self.trainer = self.trainer_cls(self.model, [task], [1.0], 'total_step',
                                        evaluator=self.evaluator,
                                        check_interval=g.check_interval,
                                        eval_interval=g.eval_interval)

    def run(self):
        self.trainer.train(self.dl_reg)


class CbowManager(LMManager):

    task_cls = CbowTask


class AdaptLMManager(LMManager):

    model_cls = AdaptLM
    trainer_cls = AdaptLMTrainer
    task_cls = AdaptLMTask


class AdaptCbowManager(CbowManager):

    model_cls = AdaptLM
    trainer_cls = AdaptLMTrainer
    task_cls = AdaptCbowTask


class DecipherManager(BaseManager):

    add_argument('dev_data_path', dtype='path', msg='Path to dev data.')
    add_argument('aux_train_data_path', dtype='path', msg='Path to aux train data.')
    add_argument('in_domain_dev_data_path', dtype='path', msg='Path to in-domain dev data.')
    add_argument('saved_path', dtype='path')
    add_argument('saved_model_path', dtype='path', msg='Path to a saved model, skipping the local training phase.')
    add_argument('train_phi', dtype=bool, default=False,
                 msg='Flag to train phi score. Used only with supervised mode.')
    add_argument('fix_phi', dtype=bool, default=False, msg='Flag fix phi scorer.')
    # add_argument('use_mlm_loss', dtype=bool, default=False, msg='Flag to use mlm loss.')

    def __init__(self):
        self.model = DecipherModel()
        if has_gpus():
            self.model.cuda()

        train_task = DecipherTask('train')
        dev_task = DecipherTask('dev')
        self.dl_reg = DataLoaderRegistry()
        eval_tasks = [train_task, dev_task]
        if g.in_domain_dev_data_path:
            in_domain_dev_task = DecipherTask('in_domain_dev')
            self.dl_reg.register_data_loader(in_domain_dev_task, g.in_domain_dev_data_path)
            eval_tasks.append(in_domain_dev_task)
        train_tasks = [train_task]
        if g.aux_train_data_path:
            aux_train_task = DecipherTask('aux_train')
            self.dl_reg.register_data_loader(aux_train_task, g.aux_train_data_path)
            train_tasks.append(aux_train_task)

        self.dl_reg.register_data_loader(train_task, g.data_path)
        self.dl_reg.register_data_loader(dev_task, g.dev_data_path)
        self.evaluator = DecipherEvaluator(self.model, self.dl_reg, eval_tasks)

        self.trainer = DecipherTrainer(self.model, train_tasks, [1.0] * len(train_tasks), 'total_step',
                                       evaluator=self.evaluator,
                                       check_interval=g.check_interval,
                                       eval_interval=g.eval_interval)
        if g.train_phi:
            freeze(self.model.self_attn_layers)
            freeze(self.model.positional_embedding)
            freeze(self.model.emb_for_label)
            freeze(self.model.label_predictor)
        if g.saved_model_path:
            self.trainer.load(g.saved_model_path, load_phi_scorer=True)
            if g.fix_phi:
                freeze(self.model.phi_scorer)
        #     freeze(self.model.self_attn_layers)
        #     freeze(self.model.positional_embedding)
        #     freeze(self.model.emb_for_label)
        #     freeze(self.model.label_predictor)
            self.trainer.set_optimizer()

    def run(self):
        self.trainer.train(self.dl_reg)


class SearchSolverManager(BaseManager):
    """
    On tmp:
    P: 553 / 950 = 0.5821052631578948
    R: 553 / 978 = 0.565439672801636
    F: 0.5736514522821577

    From 50 samples:
    19 errors in total: 12 from vocab, 4 merged, 3 tricky/maybe solvable by counts.

    After fixing merged:
    P: 559 / 922 = 0.60629067245
    R: 559 / 978 = 0.57157464212
    F: 0.5884210526274193

    After matching prefixes:
    P: 673 / 922 = 0.72993492407
    R: 673 / 978 = 0.6881390593
    F: 0.708421052625276

    Removing #words in the objective #chars - #words only marginally brings down the numbers. 556/665 matches for exact/prefix out of 928 predictions
    compared with 559/673 out of 922. UPDATE: new run gives 567/680 out of 928, actually better without it.
    """

    def run(self):
        dl_reg = DataLoaderRegistry()
        dev_task = DecipherTask('dev')
        dl_reg.register_data_loader(dev_task, g.dev_data_path)
        dl = dl_reg[dev_task]

        with open(g.vocab_path, 'r', encoding='utf8') as fin:
            vocab = set(line.strip() for line in fin)

        solver = SearchSolver(vocab, g.max_num_words)

        evaluator = SearchSolverEvaluator(solver)
        prf_scores = evaluator.evaluate(dl)
        logging.info(prf_scores.get_table())


class ReduceLR:
    # NOTE(j_luo) Seems that there is no such a thing as BaseLRScheduler.

    def __init__(self, optimizer: torch.optim.Optimizer, factor: float):
        self.optimizer = optimizer
        self._factor = factor

    def step(self):
        """Copied from https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self._factor
            param_group['lr'] = new_lr
            logging.imp(f'Learning rate is now {new_lr:.4f}.')


class ExtractManager(BaseManager):

    # IDEA(j_luo) when to put this in manager/trainer? what about scheduler? annealing? restarting? Probably all in trainer -- you need to track them with pbars.
    add_argument('optim_cls', default='adam', dtype=str, choices=['adam', 'adagrad', 'sgd'], msg='Optimizer class.')
    add_argument('anneal_factor', default=0.5, dtype=float, msg='Mulplication value for annealing.')
    add_argument('min_threshold', default=0.01, dtype=float, msg='Min value for threshold')
    add_argument('use_new_data_loader', default=True, dtype=bool, msg='Flag to use the new data loader.')

    _name2cls = {'adam': Adam, 'adagrad': Adagrad, 'sgd': SGD}

    def __init__(self):
        task = ExtractTask()
        self.dl_reg = DataLoaderRegistry()
        self.dl_reg.register_data_loader(task, g.data_path)

        lu_size = ku_size = None
        char_sets = self.dl_reg[task].dataset.corpus.char_sets
        lcs = char_sets[g.lost_lang]
        vocab = BaseAlignedBatch.known_vocab
        kcs = vocab.char_set
        if g.input_format == 'text':
            lu_size = len(lcs)
            ku_size = len(kcs)
        self.model = ExtractModel(lu_size, ku_size, self.dl_reg[task], vocab)
        # HACK(j_luo)
        from xib.aligned_corpus.ipa_sequence import IpaSequence

        def align(lost_char, known_char):
            known_id = kcs.unit2id[IpaSequence(known_char)]
            lost_id = lcs.unit2id[lost_char]
            self.model.unit_aligner.weight.data[lost_id, known_id] = 2.5
        #align('m', 'm')
        #align('k', 't͡ʃ')
        #align('k', 'k')
        #align('d', 'd')
        #align('l', 'l')

        # align('n', 'n')
        # align('t', 't')
        # align('w', 'w')
        # align('h', 'h')
        # align('b', 'b')
        # align('b', 'f')
        # align('b', 'v')
        # align('j', 'j')
        # align('þ', 'θ')

        if has_gpus():
            self.model.cuda()
        logging.info(str(self.model))

        eval_cls = AlignedExtractEvaluator if g.use_new_data_loader else ExtractEvaluator
        self.evaluator = eval_cls(self.model, self.dl_reg[task], BaseAlignedBatch.known_vocab)

        self.trainer = ExtractTrainer(self.model, [task], [1.0], 'total_step',
                                      stage_tnames=['round', 'total_step'],
                                      evaluator=self.evaluator,
                                      check_interval=g.check_interval,
                                      eval_interval=g.eval_interval,
                                      save_interval=g.save_interval)
        if g.saved_model_path:
            self.trainer.load(g.saved_model_path)
        # self.trainer.set_optimizer(Adam, lr=g.learning_rate)

    def run(self):
        optim_cls = self._name2cls[g.optim_cls]
        self.trainer.threshold = g.init_threshold
        # , momentum=0.9, nesterov=True)
        self.trainer.set_optimizer(optim_cls, lr=g.learning_rate, weight_decay=g.weight_hyper)
        # Save init parameters.

        out_path = g.log_dir / f'saved.init'
        self.trainer.save_to(out_path)
        # # HACK(j_luo)
        # self.trainer.reset(reset_params=True)
        self.trainer.er = 1.0
        while self.trainer.threshold > g.min_threshold:
            if g.update_p_weights:
                # HACK(j_luo)
                self.trainer.update_p_weights()
            self.trainer.reset()
            self.trainer.set_optimizer(optim_cls, lr=g.learning_rate, weight_decay=g.weight_hyper)

            self.trainer.train(self.dl_reg)
            self.trainer.tracker.update('round')

            # HACK(j_luo)
            self.trainer.er *= 0.9
            self.trainer.er = max(self.trainer.er, g.expected_ratio)


class PrepareManager(BaseManager):

    add_argument('lost_lang', dtype=str)
    add_argument('known_lang', dtype=str)
    add_argument('dictionary_path', dtype='path', default='data/de.csv')

    def _get_transcriber(self, lang: str) -> BaseTranscriber:

        def converter(s: str) -> str:
            s = re.sub(r'\s+', '', s)
            s = s.replace('ʔ', '')
            s = s.replace('l̩', 'əl')
            s = s.replace('n̩', 'ən')
            s = s.replace('m̩', 'əm')
            s = s.replace('ç', 'ç')
            s = s.replace('ˈ', '')
            s = s.replace('ˌ', '')
            return s

        stf = SimpleTranscriberFactory()

        if lang == 'nhd':
            simple = stf.get_transcriber('phonemizer')
            dt = stf.get_transcriber('dictionary', csv_path=g.dictionary_path, converter=converter)
            tr = TranscriberWithBackoff(dt, simple)
        elif lang in ['got', 'germ']:
            tr = stf.get_transcriber('rule', lang='got')
        elif lang == 'ae':
            tr = stf.get_transcriber('third_party', func=oe)
        else:
            raise ValueError(f'Unsupported language {lang}.')

        return tr

    def run(self):
        transcriber = MultilingualTranscriber()
        transcriber.register_lang(g.lost_lang, self._get_transcriber(g.lost_lang))
        transcriber.register_lang(g.known_lang, self._get_transcriber(g.known_lang))
        corpus = AlignedCorpus.from_data_path(g.lost_lang, g.known_lang, g.data_path, transcriber)
        out_path = f'data/{g.lost_lang}-{g.known_lang}.corpus.tsv'
        corpus.to_tsv(out_path)
