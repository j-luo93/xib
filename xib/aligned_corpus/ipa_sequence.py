from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from typing import ClassVar, Dict, List, Optional, Union

from ipapy.ipastring import IPAString

from dev_misc.devlib import LT
from dev_misc.utils import cached_property, concat_lists
from xib.ipa.process import Segment


class IpaSequence(SequenceABC):

    _cache: ClassVar[Dict[str, Segment]] = dict()

    def __init__(self, raw_string: str):
        self._seg = self._get_segment(raw_string)
        self.data: List[IPAString] = [IPAString(ipa_lst) for ipa_lst in self._seg.merged_ipa]
        self._canonical_string = ''.join(map(str, concat_lists(self.data)))

    @property
    def feat_matrix(self) -> LT:
        return self._seg.feat_matrix

    @property
    def cv_list(self) -> List[IpaSequence]:
        """Return a list of consonants and vowels. This should be the same as `cv_list` of `Segment` but done in a different way."""
        return [IpaSequence(str(ipa_str[0])) for ipa_str in self.data]

    def _get_segment(self, raw_string: str) -> Segment:
        cls = type(self)
        if raw_string in cls._cache:
            return cls._cache[raw_string]
        seg = Segment(raw_string)
        cls._cache[raw_string] = seg
        return seg


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: Union[int, slice]) -> IpaSequence:
        if isinstance(idx, int):
            raw_string = str(self.data[idx])
        else:
            raw_string = str(sum(self.data[idx], IPAString()))
        return IpaSequence(raw_string)

    def __str__(self):
        return self._canonical_string

    def __repr__(self):
        return f'IpaSequence({self})'

    def __hash__(self):
        return hash(self._canonical_string)

    def __eq__(self, other: IpaSequence):
        if not isinstance(other, IpaSequence):
            return False

        return self._canonical_string == other._canonical_string
