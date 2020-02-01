import logging
import re
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


_reverse_GVS = {
    'aɪ': 'iː',
    'iː': 'eː',
    'eɪ': 'aː',
    'aʊ': 'uː',
    'uː': 'oː',
    'oʊ': 'ɔː'
}


def postprocess(phoneme: str, prefix: str = '', mode: str = 'none') -> str:

    if mode == 'none':
        return prefix + phoneme

    if mode == 'redup':
        return prefix + re.sub(r'(.)ː', r'\1\1', phoneme)

    for k, v in _reverse_GVS.items():
        phoneme = phoneme.replace(k, v)

    return prefix + phoneme


class BaseTranscriber(metaclass=ABCWithCacheMetaclass):

    _cache: ClassVar[Dict[str, Set[IPAString]]]

    def __init__(self, postprocess_mode: str = 'none'):
        assert postprocess_mode in ['none', 'gvs', 'redup']
        self.postprocess_mode = postprocess_mode

    @abstractmethod
    def _transcribe(self, grapheme: str) -> Sequence[str]: ...

    def postprocess(self, phoneme: str, prefix: str = '') -> str:
        """Postprocess the phoneme."""
        # Standardize phonemes.
        try:
            ipa = IPAString(unicode_string=phoneme)
        except ValueError:
            logging.exception(f'Invalid IPAString for {phoneme}. Invalid characters will be ignored.')
            ipa = IPAString(unicode_string=phoneme, ignore=True)
        phoneme = str(ipa)
        return postprocess(phoneme, prefix=prefix, mode=self.postprocess_mode)

    def transcribe(self, grapheme: str) -> Set[str]:
        cls = type(self)
        if grapheme in cls._cache:
            return cls._cache[grapheme]

        phonemes = self._transcribe(grapheme)
        ret = set()
        for phoneme in phonemes:
            prefix = ''
            if '?' in phoneme:
                match = re.match(r'^(\[[\?\]\[]+\])(.+)$', phoneme)
                prefix = match.group(1)
                phoneme = match.group(2)
            ret.add(self.postprocess(phoneme, prefix=prefix))

            # ret.add(IPAString(unicode_string=phoneme))

        cls._cache[grapheme] = ret
        return ret


class PhonemizerTranscriber(BaseTranscriber):

    def _transcribe(self, grapheme: str) -> Sequence[str]:
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
    'got': _got2ipa_map,
    'germ': _got2ipa_map  # HACK(j_luo)
}


class RuleBasedTranscriber(BaseTranscriber):

    def __init__(self, lang: str, postprocess_mode: str = 'none'):
        # HACK(j_luo) Some issues with using super() here due to mixin metaclasses.
        BaseTranscriber.__init__(self, postprocess_mode=postprocess_mode)
        self._sub_func = _get_sub_func(_ipa_dict[lang])

    def _transcribe(self, grapheme: str) -> Sequence[str]:
        return [self._sub_func(grapheme)]


class ThirdPartyTranscriber(BaseTranscriber):

    def __init__(self, func, postprocess_mode: str = 'none'):
        BaseTranscriber.__init__(self, postprocess_mode=postprocess_mode)
        self.func = func

    def _transcribe(self, grapheme: str) -> Sequence[str]:
        return [self.func(grapheme)]


class BaseTranscriberError(Exception):
    """Base exception for all transliteration-related errors."""


class EntryNotFound(BaseTranscriberError):
    """Raise this error if an entry is not found."""


class DictionaryTranscriber(BaseTranscriber):

    def __init__(self, csv_path: str, converter: Optional[Callable] = None, postprocess_mode: str = 'none'):
        BaseTranscriber.__init__(self, postprocess_mode=postprocess_mode)
        df = pd.read_csv(csv_path)
        df = df.dropna()
        df['headword'] = df['headword'].str.lower()
        if converter is not None:
            df['pronunciation'] = df['pronunciation'].apply(converter)
        self.data = pd.pivot_table(df, values='pronunciation', index='headword', aggfunc=set)

    def _transcribe(self, grapheme: str) -> Sequence[str]:
        try:
            return self.data.loc[grapheme].values[0]
        except KeyError:
            raise EntryNotFound(f'Entry not found for {grapheme}.')


class SimpleTranscriberFactory(Singleton):

    def get_transcriber(self,
                        kind: str,
                        postprocess_mode: str = 'none',
                        lang: Optional[str] = None,
                        csv_path: Optional[str] = None,
                        converter: Optional[Callable] = None,
                        func: Optional[Callable] = None):
        if kind == 'phonemizer':
            obj = PhonemizerTranscriber(postprocess_mode=postprocess_mode)
        elif kind == 'dictionary':
            obj = DictionaryTranscriber(csv_path, converter=converter, postprocess_mode=postprocess_mode)
        elif kind == 'rule':
            obj = RuleBasedTranscriber(lang, postprocess_mode=postprocess_mode)
        elif kind == 'third_party':
            obj = ThirdPartyTranscriber(func, postprocess_mode=postprocess_mode)
        else:
            raise ValueError(f'Unrecognized kind {kind}.')
        return obj


class TranscriberWithBackoff(BaseTranscriber):

    def __init__(self, main: BaseTranscriber, backoff: BaseTranscriber, postprocess_mode: str = 'none'):
        BaseTranscriber.__init__(self, postprocess_mode=postprocess_mode)
        self.main = main
        self.backoff = backoff

    def _transcribe(self, grapheme: str) -> Sequence[str]:
        try:
            return self.main._transcribe(grapheme)
        except BaseTranscriberError:
            return self.backoff._transcribe(grapheme)


class MultilingualTranscriber:

    def __init__(self):
        self._transcribers: Dict[str, BaseTranscriber] = dict()

    def register_lang(self, lang: str, transcriber: BaseTranscriber):
        if lang in self._transcribers:
            raise ValueError(f'Language {lang} already has a transcriber.')
        self._transcribers[lang] = transcriber

    def transcribe(self, grapheme: str, lang: str) -> Set[IPAString]:
        transcriber = self._transcribers[lang]
        return transcriber.transcribe(grapheme)
