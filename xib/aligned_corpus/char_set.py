"""CharSet is responsible for converting between ids and units."""

import logging
from typing import (ClassVar, Dict, Iterable, Iterator, List, NewType, Set,
                    Tuple)

from dev_misc import g
from dev_misc.utils import Singleton
from xib.aligned_corpus.ipa_sequence import Content

Lang = NewType('Lang', str)
Key = Tuple[Lang, bool]


class CharSet:

    def __init__(self, units: Set[Content], lang: Lang, is_ipa: bool):
        self.lang = lang
        self.is_ipa = is_ipa
        self.id2unit = sorted(units, key=str)
        if g.use_empty_symbol:
            id2unit = ['<EMPTY>'] + id2unit
        self.unit2id = {u: i for i, u in enumerate(self.id2unit)}

    def to_id(self, unit: Content):
        return self.unit2id[unit]

    def to_unit(self, idx: int):
        return self.id2unit[idx]

    def __iter__(self) -> Iterator[Content]:
        yield from self.id2unit

    def __len__(self):
        return len(self.id2unit)


class CharSetFactory(Singleton):

    _char_sets: ClassVar[Dict[Key, CharSet]] = dict()

    def get_char_set(self, contents: Iterable[Content], lang: Lang, is_ipa: bool) -> CharSet:
        cls = type(self)
        key = (lang, is_ipa)
        if key in cls._char_sets:
            char_set = cls._char_sets[key]
            logging.imp(f'Reusing the char set for {key}.')
            return char_set

        all_units = set()
        for content in contents:
            if is_ipa:
                all_units.update(content.cv_list)
            else:
                all_units.update(content)

        logging.imp(f'Getting the char set for {key}.')
        char_set = CharSet(all_units, lang, is_ipa)
        cls._char_sets[key] = char_set
        return char_set
