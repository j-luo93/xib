from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from dev_misc import g

from xib.ipa.process import Span, Segmentation


class _State:

    def __init__(self, length: int, value: Optional[Tuple[int]] = None, spans: Optional[List[Span]] = None):
        self._length = length
        self._value = value or tuple([0] * length)
        self.spans = spans or list()

    def __hash__(self):
        return hash(self._value)

    @property
    def value(self):
        return self._value

    def __eq__(self, other: _State):
        if not isinstance(other, _State):
            return False
        return self._value == other.value

    def fit(self, word: str, segment: str) -> List[_State]:
        ret = list()
        for match in re.finditer(word, segment):
            start = match.start()
            end = match.end()
            if any(self._value[i] for i in range(start, end)):
                continue
            new_value = list(self._value)
            for i in range(start, end):
                new_value[i] = 1
            new_value = tuple(new_value)
            new_span = Span(word, start, end - 1)
            new_spans = self.spans + [new_span]
            ret.append(_State(self._length, value=new_value, spans=new_spans))
        return ret


class SearchSolver:

    def __init__(self, vocab: Set[str], max_num_words: int):
        self._vocab = {k for k in vocab if len(k) >= g.min_word_length}
        self._max_num_words = max_num_words

    def find_best(self, segment: str) -> Tuple[int, _State]:
        fs = defaultdict(dict)
        empty_state = _State(len(segment))
        fs[0][empty_state] = 0
        best_value = 0
        best_state = empty_state

        vocab = {v for v in self._vocab if empty_state.fit(v, segment)}

        for i in range(1, self._max_num_words + 1):
            prev_fs = fs[i - 1]
            for prev_state, prev_value in prev_fs.items():
                for word in vocab:
                    new_states = prev_state.fit(word, segment)
                    for new_state in new_states:
                        new_value = len(word) + prev_value
                        fs[i][new_state] = new_value
                        if new_value - i > best_value:
                            best_value = new_value - i
                            best_state = new_state
        return best_value, best_state
