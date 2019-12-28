import subprocess
from abc import ABC, ABCMeta, abstractmethod
from typing import Callable, ClassVar, Dict, Optional, Sequence, Set

import pandas as pd
from ipapy.ipastring import IPAString

from dev_misc.utils import Singleton
from xib.gothic.core import _get_sub_func


class _CacheMetaclass(type):

    def __new__(cls, name, bases, attrs):
        # NOTE(j_luo) Create a cache for each subclass.
        attrs['_cache'] = dict()
        return type.__new__(cls, name, bases, attrs)


class ABCWithCacheMetaclass(ABCMeta, _CacheMetaclass):

    def __new__(cls, name, bases, attrs):
        obj = _CacheMetaclass.__new__(cls, name, bases, attrs)  # pylint: disable=too-many-function-args
        name = obj.__name__
        bases = tuple(obj.mro())
        attrs = obj.__dict__.copy()
        obj = ABCMeta.__new__(cls, name, bases, attrs)
        return obj

    def __init__(cls, name, bases, attrs):
        # call both parents
        ABCMeta.__init__(cls, name, bases, attrs)
        _CacheMetaclass.__init__(cls, name, bases, attrs)


class BaseTransliterator(metaclass=ABCWithCacheMetaclass):

    _cache: ClassVar[Dict[str, Set[IPAString]]]

    @abstractmethod
    def _transliterate(self, grapheme: str) -> Sequence[str]: ...

    def transliterate(self, grapheme: str) -> Set[IPAString]:
        cls = type(self)
        if grapheme in cls._cache:
            return cls._cache[grapheme]

        phonemes = self._transliterate(grapheme)
        ret = set()
        for phoneme in phonemes:
            ret.add(IPAString(unicode_string=phoneme))

        cls._cache[grapheme] = ret
        return ret


class PhonemizerTransliterator(BaseTransliterator):

    def _transliterate(self, grapheme: str) -> Sequence[str]:
        out = subprocess.run(f'echo "{grapheme}" | phonemize -l de', shell=True,
                             encoding='utf8', capture_output=True, check=True)
        return [out.stdout.strip()]


_got2ipa_map = {
    "ah": "aːh",
    "aih": "ɛh",
    "air": "ɛr",
    "ai": "ɛː",
    "auh": "ɔh",
    "aur": "ɔr",
    "au": "ɔː",
    "ei": "iː",
    "e": "eː",
    "o": "oː",
    "ur": "uːr",
    "uh": "uːh",
    "ab": "aβ",
    "ɛb": "ɛβ",
    "ɔb": "ɔβ",
    "ib": "iβ",
    "eb": "eβ",
    "ob": "oβ",
    "ub": "uβ",
    "bd": "βd",
    "bn": "βn",
    "bm": "βm",
    "bg": "βg",
    "bl": "βl",
    "bj": "βj",
    "br": "βr",
    "bw": "βw",
    "bz": "βz",
    " β": " b",
    "ad": "að",
    "ɛd": "ɛð",
    "ɔd": "ɔð",
    "id": "ið",
    "ed": "eð",
    "od": "oð",
    "ud": "uð",
    "db": "ðb",
    "dβ": "ðβ",
    "dn": "ðn",
    "dm": "ðm",
    "dg": "ðg",
    "dl": "ðl",
    "dj": "ðj",
    "dr": "ðr",
    "dw": "ðw",
    "dz": "ðz",
    " ð": " d",
    "f": "ɸ",
    "gw": "ɡʷ",
    "hw": "hʷ",
    "ag": "aɣ",
    "ɛg": "ɛɣ",
    "ɔg": "ɔɣ",
    "ig": "iɣ",
    "eg": "eɣ",
    "og": "oɣ",
    "ug": "uɣ",
    "gb": "ɣb",
    "gβ": "ɣβ",
    "gn": "ɣn",
    "gm": "ɣm",
    "gg": "ŋg",
    "gl": "ɣl",
    "gj": "ɣj",
    "gr": "ɣr",
    "gw": "ɣw",
    "gz": "ɣz",
    "gp": "xp",
    "gt": "xt",
    "gk": "ŋk",
    "gɸ": "xɸ",
    "gh": "xh",
    "gs": "xs",
    "gþ": "xþ",
    "gq": "xq",
    " ɣ": " g",
    " x": " g",
    "qw": "kʷ",
    "þ": "θ",
    'ƕ': 'hʷ'
}
_ipa_dict = {
    'got': _got2ipa_map
}


class RuleBasedTransliterator(BaseTransliterator):

    def __init__(self, lang: str):
        self._sub_func = _get_sub_func(_ipa_dict[lang])

    def _transliterate(self, grapheme: str) -> Sequence[str]:
        return [self._sub_func(grapheme)]


class BaseTransliteratorError(Exception):
    """Base exception for all transliteration-related errors."""


class EntryNotFound(BaseTransliteratorError):
    """Raise this error if an entry is not found."""


class DictionaryTransliterator(BaseTransliterator):

    def __init__(self, csv_path: str, converter: Optional[Callable] = None):
        df = pd.read_csv(csv_path)
        df = df.dropna()
        df['headword'] = df['headword'].str.lower()
        if converter is not None:
            df['pronunciation'] = df['pronunciation'].apply(converter)
        self.data = pd.pivot_table(df, values='pronunciation', index='headword', aggfunc=set)

    def _transliterate(self, grapheme: str) -> Sequence[str]:
        try:
            return self.data.loc[grapheme].values[0]
        except KeyError:
            raise EntryNotFound(f'Entry not found for {grapheme}.')


class SimpleTransliteratorFactory(Singleton):

    def get_transliterator(self, kind: str, csv_path: Optional[str] = None, converter: Optional[Callable] = None):
        if kind == 'phonemizer':
            obj = PhonemizerTransliterator()
        elif kind == 'dictionary':
            obj = DictionaryTransliterator(csv_path, converter=converter)
        else:
            raise ValueError(f'Unrecognized kind {kind}.')
        return obj


class TransliteratorWithBackoff(BaseTransliterator):

    def __init__(self, main: BaseTransliterator, backoff: BaseTransliterator):
        self.main = main
        self.backoff = backoff

    def _transliterate(self, grapheme: str) -> Sequence[str]:
        try:
            return self.main._transliterate(grapheme)
        except BaseTransliteratorError:
            return self.backoff._transliterate(grapheme)


class MultilingualTranliterator:

    def __init__(self):
        self._transliterators: Dict[str, BaseTransliterator] = dict()

    def register_lang(self, lang: str, transliterator: BaseTransliterator):
        if lang in self._transliterators:
            raise ValueError(f'Language {lang} already has a transliterator.')
        self._transliterators[lang] = transliterator

    def transliterate(self, grapheme: str, lang: str) -> Set[IPAString]:
        transliterator = self._transliterators[lang]
        return transliterator.transliterate(grapheme)
