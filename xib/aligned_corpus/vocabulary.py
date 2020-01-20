from typing import IO, Iterable

import numpy as np
import torch

from dev_misc import add_argument, g
from dev_misc.devlib import get_array, get_length_mask
from dev_misc.devlib.named_tensor import Rename
from xib.aligned_corpus.char_set import (DELETE_SYM, EMPTY_SYM, INSERT_SYM,
                                         CharSetFactory)
from xib.aligned_corpus.corpus import Content, Stem, Word
from xib.batch import convert_to_dense


class Vocabulary:

    add_argument('add_infinitive', dtype=bool, default=False)
    add_argument('use_stem', dtype=bool, default=False)

    def __init__(self):

        def has_proper_length(content: Content) -> bool:
            l = len(content)
            return g.min_word_length <= l <= g.max_word_length

        word_cls = Stem if g.use_stem else Word
        def gen_word(io: IO):

            for line in io:
                try:
                    word = word_cls.from_saved_string(line.strip())
                    yield from word.ipa
                except ValueError:
                    pass

        def expand(vocab: Iterable[Content]):
            if g.add_infinitive:
                for content in vocab:
                    if str(content).endswith('ən'):
                        yield from [content, content[:-2]]
                    else:
                        yield content
            else:
                yield from vocab

        with open(g.vocab_path, 'r', encoding='utf8') as fin:
            self.vocab = get_array(sorted(set(expand(filter(has_proper_length, gen_word(fin)))),
                                          key=str))

            self.vocab_length = torch.LongTensor(list(map(len, self.vocab)))
            max_len = self.vocab_length.max().item()
            self.vocab_source_padding = ~get_length_mask(self.vocab_length, max_len)
            self.vocab_length.rename_('vocab')
            self.vocab_source_padding.rename_('vocab', 'length')

            feat_matrix = [word.feat_matrix for word in self.vocab]
            self.vocab_feat_matrix = torch.nn.utils.rnn.pad_sequence(feat_matrix, batch_first=True)
            self.vocab_feat_matrix.rename_('vocab', 'length', 'feat_group')

            with Rename(self.vocab_feat_matrix, vocab='batch'):
                vocab_dense_feat_matrix = convert_to_dense(self.vocab_feat_matrix)
            self.vocab_dense_feat_matrix = {k: v.rename(batch='vocab') for k, v in vocab_dense_feat_matrix.items()}

            # Get the entire set of units from vocab. Always use IPA for the known vocab.
            csf = CharSetFactory()
            self.char_set = csf.get_char_set(self.vocab, g.known_lang, True)

            # Now indexify the vocab. Gather feature matrices for units as well.
            indexed_segments = np.zeros([len(self.vocab), max_len], dtype='int64')
            if g.one2two:
                unit_feat_matrix = {INSERT_SYM: torch.zeros(0).long(), DELETE_SYM: torch.zeros(0).long()}
            elif g.use_empty_symbol:
                unit_feat_matrix = {EMPTY_SYM: torch.zeros(0).long()}
            else:
                unit_feat_matrix = dict()
            for i, segment in enumerate(self.vocab):
                indexed_segments[i, range(len(segment))] = [self.char_set.to_id(u) for u in segment.cv_list]
                for j, u in enumerate(segment.cv_list):
                    if u not in unit_feat_matrix:
                        unit_feat_matrix[u] = segment.feat_matrix[j]
            unit_feat_matrix = [unit_feat_matrix[u] for u in self.char_set]
            unit_feat_matrix = torch.nn.utils.rnn.pad_sequence(unit_feat_matrix, batch_first=True)
            self.unit_feat_matrix = unit_feat_matrix.unsqueeze(dim=1)
            self.indexed_segments = torch.from_numpy(indexed_segments)

            # Use dummy length to avoid the trouble later on.
            # HACK(j_luo) Have to provide 'length'.
            self.unit_feat_matrix.rename_('unit', 'length', 'feat_group')
            self.indexed_segments.rename_('vocab', 'length')
            with Rename(self.unit_feat_matrix, unit='batch'):
                unit_dense_feat_matrix = convert_to_dense(self.unit_feat_matrix)
            self.unit_dense_feat_matrix = {
                k: v.rename(batch='unit')
                for k, v in unit_dense_feat_matrix.items()
            }

    def __len__(self):
        return len(self.vocab)
