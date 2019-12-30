from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from typing import ClassVar, Dict, List, Optional

from ipapy.ipastring import IPAString

from dev_misc.utils import concat_lists
from xib.ipa.process import Segment


class IpaSequence(SequenceABC):

    _cache: ClassVar[Dict[str, Segment]] = dict()

    def __init__(self, raw_string: Optional[str] = None, data: Optional[List[IPAString]] = None):
        if data is not None:
            self.data = data
        else:
            seg = self._get_segment(raw_string)
            self.data: List[IPAString] = [IPAString(ipa_lst) for ipa_lst in seg.merged_ipa]
        self._canonical_string = ''.join(map(str, concat_lists(self.data)))

    def save(self) -> str:
        """Save as a loadable string."""
        return '-'.join(map(str, self.data))

    @classmethod
    def from_saved_string(self, saved_string: str) -> IpaSequence:
        data = [IPAString(unicode_string=x) for x in saved_string.split('-')]
        return IpaSequence(data=data)

    def _get_segment(self, raw_string: str) -> Segment:
        cls = type(self)
        if raw_string in cls._cache:
            return cls._cache[raw_string]
        seg = Segment(raw_string)
        cls._cache[raw_string] = seg
        return seg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> str:
        return str(self.data[idx])

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
