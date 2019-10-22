"""
Mostly based on https://en.wikipedia.org/wiki/Help:IPA/Basque and https://en.wikipedia.org/wiki/Basque_language#Phonology.

Some notes:
1. /β/, /ð/ and /ɣ/ are ignored since they are mostly present in southern dialects.
2. Only -r- is mapped to tap, all other instances or r's are treated as trills.
3. Processing goes like this:
        3.1 Identifying all possible locations to replace a digraph key.
        3.2 If there is no conflicting in applying these changes, replace them with corresponding values. Otherwise, discard this segment.
            Conflicts might happen if you can something like 'baitta': 'b-a-it-t-a' or 'b-a-i-tt-a'.
        3.3 Replace simple keys.

With this script, eventually I ended up with 115 unknown errors and 3 conflict errors.
"""

import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from tqdm import tqdm

simple = {
    'a': 'ɑ', 'e': 'ɛ', 'i': 'i', 'o': 'o', 'u': 'u', 'ü': 'y',
    'b': 'b', 'd': 'd', 'f': 'f', 'g': 'ɡ', 'h': 'h', 'j': 'j',
    'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 's': 's̺',
    'z': 's̻', 't': 't', 'x': 'ʃ', 'ñ': 'ɲ', 'y': 'y',
    '-r': 'r', 'r-': 'r', '-r-': 'ɾ'
}

digraph = {
    'tt': 'c', 'dd': 'ɟ', 'ts': 't̺s̺', 'tz': 't̪s̻', 'tx': 'tʃ',
    'll': 'ʎ', '-in-': 'ɲ', '-it-': 'c', '-id-': 'ɟ',
    '-il-': 'ʎ', '-rr-': 'r'
}

g_char_set = set(''.join(list(simple.keys()) + list(digraph.keys())))
g_char_set.remove('-')

p_char_set = set(''.join(list(simple.values()) + list(digraph.values())))

vowels = {'a', 'e', 'i', 'o', 'u', 'y'}


def is_covered(segment):
    """Check every character of segment is valid."""
    return set(segment) <= g_char_set


def is_valid_ipa(segment):
    """Check everything is in IPA."""
    return set(segment) <= p_char_set


@dataclass(order=True)
class Match:
    """Only use `end` for ordering."""

    segment: str = field(compare=False)
    pattern: str = field(compare=False)
    start: int = field(compare=False)
    end: int
    di_key: str = field(compare=False)
    di_value: str = field(compare=False)

    def is_valid(self):
        if '-' not in self.di_key:
            return True
        ret = True
        if self.di_key.startswith('-'):
            if self.start == 0:
                ret = False
            elif self.segment[self.start - 1] not in vowels:
                ret = False
        if self.di_key.endswith('-'):
            if self.end == len(self.segment):
                ret = False
            elif self.segment[self.end] not in vowels:
                ret = False
        return ret


def no_conflict(matches: List[Match]):
    occupied = set()
    for match in matches:
        for position in range(match.start, match.end):
            if position in occupied:
            occupied.add(position)
                return False
    return True


class MatchConflict(Exception):
    pass


class UnknownCharacter(Exception):
    pass


class InvalidIpa(Exception):
    pass


def g2p(word: str):
    if is_covered(word):
        # Identifying all possible locations for digraph keys.
        matches = list()
        for di_key, di_value in digraph.items():
            pattern = di_key.strip('-')
            for match in re.finditer(pattern, word):
                match = Match(word, pattern, match.start(), match.end(), di_key, di_value)
                if match.is_valid():
                    matches.append(match)

        # Apply changes if possible
        if no_conflict(matches):
            segment = word
            for match in sorted(matches, reverse=True):  # NOTE(j_luo) Replace from the end to the start.
                segment = segment.replace(match.pattern, match.di_value)

            # Now we can proceed to deal with simple mappings.
            for key, value in simple.items():
                segment = segment.replace(key, value)

            if is_valid_ipa(segment):
                return segment
            else:
                raise InvalidIpa()
        else:
            raise MatchConflict()
    else:
        raise UnknownCharacter()


# Unused for now.
def b2ipa(word, basque2ipa, basque2ipa_midVowels):
    vowels = ['a', 'e', 'i', 'o', 'u', 'y']

    # REPLACING CONTEXTUALIZED
    for c in basque2ipa_midVowels:
        if c in word:
            for s in vowels:
                for e in vowels:
                    word = word.replace(s + c + e, ''.join([s, basque2ipa_midVowels[c], e]))

    # REPLACING ALL OTHERS
    for c in basque2ipa:
        word = word.replace(c, basque2ipa[c])

    return word


def clean_html(line):
    pattern = re.compile('<.*?>')
    return re.sub(pattern, '', line)

# print(b2ipa('uilattey', basque2ipa, basque2ipa_midVowels))


if __name__ == "__main__":
    in_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    errors = Counter()
    total = 0
    with in_path.open('r', encoding='utf8') as fin, out_path.open('w', encoding='utf8') as fout:
        for line in tqdm(fin):
            line = line.lower()
            line = clean_html(line)
            if '-' in line or '(' in line or ')' in line:
                continue
            total += 1
            out = list()
            for word in line.strip().split():
                try:
                    segment = g2p(word)
                    out.append(segment)
                except MatchConflict:
                    logging.debug(f'Conflicting matches for {word}, discarding it.')
                    errors['conflict'] += 1
                except InvalidIpa:
                    logging.debug(f'Invalid IPA found in {word}.')
                    errors['invalid'] += 1
                except UnknownCharacter:
                    logging.debug(f'Found unknown characters in {word}, discarding it.')
                    errors['unknown'] += 1

            if out:
                fout.write(' '.join(out) + '\n')
    print(total)
    print(errors)
