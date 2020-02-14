import re
from collections import defaultdict

import cltk.corpus.utils.importer
import cltk.lemmatize.old_english.lemma as oe_l
import cltk.phonology.old_english.orthophonology
import cltk.phonology.orthophonology
import pandas as pd
import tqdm
from cltk.phonology.old_english.orthophonology import LetterNotFound
from cltk.phonology.old_english.orthophonology import \
    OldEnglishOrthophonology as oe
from ipapy.ipastring import IPAString
from unidecode import unidecode

from dev_misc.utils import concat_lists
from xib.aligned_corpus.transcriber import (MultilingualTranscriber,
                                            SimpleTranscriberFactory,
                                            ThirdPartyTranscriber,
                                            _get_sub_func)
from xib.data_prepare.table import (CogSet, Lemmas, Stems, Tokens,
                                    generate_data_file)
from xib.gothic.core import _get_sub_func


def gothic2latin_script_transformer(sentence):  # can only be applied to the non-ipa transliteration
    sentence = re.sub(r"𐌰", "a", sentence)
    sentence = re.sub(r"𐌴", "e", sentence)
    sentence = re.sub(r"𐌹", "i", sentence)
    sentence = re.sub(r"𐍉", "o", sentence)
    sentence = re.sub(r"𐌿", "u", sentence)
    sentence = re.sub(r"𐌱", "b", sentence)
    sentence = re.sub(r"𐌳", "d", sentence)
    sentence = re.sub(r"𐍆", "f", sentence)
    sentence = re.sub(r"𐌲", "g", sentence)
    sentence = re.sub(r"𐍈", "ƕ", sentence)
    sentence = re.sub(r"𐌷", "h", sentence)
    sentence = re.sub(r"𐌾", "j", sentence)
    sentence = re.sub(r"𐌺", "k", sentence)
    sentence = re.sub(r"𐌻", "l", sentence)
    sentence = re.sub(r"𐌼", "m", sentence)
    sentence = re.sub(r"𐌽", "n", sentence)
    sentence = re.sub(r"𐍀", "p", sentence)
    sentence = re.sub(r"𐌵", "q", sentence)
    sentence = re.sub(r"𐍂", "r", sentence)
    sentence = re.sub(r"𐍃", "s", sentence)
    sentence = re.sub(r"𐍄", "t", sentence)
    sentence = re.sub(r"𐌸", "þ", sentence)
    sentence = re.sub(r"𐍅", "w", sentence)
    sentence = re.sub(r"𐌶", "z", sentence)
    return sentence


def clean(sentence, aggressive=False):
    """Clean up the sentence by removing some weird characters. If `aggressive`, many diacritics will be removed as well (sometimes improperly)."""
    if aggressive:
        sentence = unidecode(sentence)
    return re.sub(r'[-*?\s\[\]]', '', sentence)


def transliterate_got(item):
    lemma, lang = item
    if lang == 'got':
        lemma = gothic2latin_script_transformer(clean(lemma))
    return lemma


standardize_map = {
    'û': 'u',
    'ï': 'i',
    '~': '',
    '-': '',
    '[': '',
    ']': '',
    '(': '',
    ')': '',
    # Present in gem-pro.
    'ā': 'a',
    'ą': 'a',
    'ē': 'e',
    'ī': 'i',
    'į': 'i',
    'ō': 'o',
    'ū': 'u',
    'ǭ': 'o',
    '̄': '',
    '₂': '',
    'ê': 'e',
    'ô': 'o',
    # Present in ang.
    'é': 'e',
    'ġ': 'g',
    'ċ': 'c',
    'ǣ': 'æ',
    'ȳ': 'y'
}
no_tilde_standardize_map = standardize_map.copy()
del no_tilde_standardize_map['~']
# `_get_sub_func` converts a dictionary specifying character mappings (could be multiple-character string mappings), into a function that does the mapping.
standardize_func = _get_sub_func(standardize_map)
no_tilde_standardize_func = _get_sub_func(no_tilde_standardize_map)


def clean_special_chars(s, tilde=True):
    if not pd.isnull(s):
        if tilde:
            return standardize_func(s)
        else:
            return no_tilde_standardize_func(s)


def transcribe(s, transcriber):
    """Transcribe a single token/lemma."""
    if pd.isnull(s):
        return None
    try:
        ipa = transcriber.transcribe(s)
    except LetterNotFound:
        return None
    try:
        assert len(ipa) == 1
        return list(ipa)[0]
    except KeyError:
        return None


def transcribe_stem(s, transcriber, lang, ignore_error=False, raw=False):
    """Transcribe a single stem."""
    if pd.isnull(s):
        return None
    ret = list()

    stems, full_form = s.split(':')
    stems = stems.split(',')
    for stem in stems:
        digits, word = stem.split('@')
        start, end = map(int, digits.split('~'))
        ipa = transcribe(word, transcriber)
        if ipa is None:
            continue
            raise RuntimeError(f'Error when transcribing {word} for {s}.')
        if raw:
            ret.append((stem, ipa))
        else:
            ret.append(stem + ':' + '{' + ipa + '}')
    if not ret:
        return None
    if raw:
        return (lang, ret, full_form)
    else:
        return lang + ';' + ','.join(ret) + ';' + full_form


def transcribe_multiple(s, transcriber):
    """Transcribe multiple tokens/lemmas separated by `|`."""
    try:
        ret = list()
        for seg in s.split('|'):
            res = transcribe(seg, transcriber)
            if res is not None:
                ret.append(res)
            else:
                ret.append('')
        return '|'.join(ret)
    except AttributeError:
        return None


def transcribe_multiple_stems(s, transcriber, lang, ignore_error=False):
    """Transcribe multiple stems separated by `|`."""
    try:
        ret = list()
        for seg in s.split('|'):
            res = transcribe_stem(seg, transcriber, lang, ignore_error=ignore_error, raw=True)
            if res is not None:
                ret.append(res)

        full_form2stem_ret = defaultdict(set)
        for lang, stem_ret, full_form in ret:
            full_form2stem_ret[full_form].update(stem_ret)
        ret = list()
        for full_form, stem_ret in full_form2stem_ret.items():
            ret.append(lang + ';' + ','.join([f'{stem}:{{{ipa}}}' for stem, ipa in stem_ret]) + ';' + full_form)
        return '|'.join(ret)
    except AttributeError:
        return None


if __name__ == "__main__":
    tqdm.tqdm.pandas()

    # Get transcribers.
    stf = SimpleTranscriberFactory()
    got_tr = stf.get_transcriber('rule', lang='got')
    ang_tr = ThirdPartyTranscriber(oe)

    # Get all relevant tables.
    got_stems = pd.read_csv('data/wulfila/processed/got_stems.tsv', sep='\t')
    got_stems_table = Stems(got_stems, 'got')

    got_lemmas = pd.read_csv('data/wulfila/processed/got_lemmas.tsv', sep='\t')
    got_lemmas_table = Lemmas(got_lemmas, 'got')

    ang_stems = pd.read_csv('data/wulfila/processed/ang_stems.tsv', sep='\t')
    ang_stems_table = Stems(ang_stems, 'ang')

    got_tokens = pd.read_csv('data/wulfila/processed/got_tokens.tsv', sep='\t')
    got_tokens_table = Tokens(got_tokens, 'got')

    cog_set = pd.read_csv('data/wiktionary/pgmc.tsv', sep='\t')
    cog_set['Language'] = cog_set['Language'].str.replace(r'^de$', 'nhd', regex=True)
    cog_set['Lemma'] = cog_set[['Lemma', 'Language']].apply(transliterate_got, axis=1)
    cog_set['Source'] = cog_set['Source'].str.strip('*').str.strip('-')
    cog_set_table = CogSet(cog_set, 'gem-pro')

    # Generate temporary data file by merging every table first.
    df = generate_data_file(got_tokens_table, got_lemmas_table, got_stems_table,
                            ang_stems_table, cog_set_table, 'data/wulfila/processed/tmp.tsv')

    # Postprocess -- cleaning, lowercasing and transcribing.
    df['Token'] = df['Token'].apply(clean_special_chars).str.lower()
    df['Stems_x'] = df['Stems_x'].apply(clean_special_chars, tilde=False).str.lower()
    df['Lemma'] = df['Lemma'].apply(clean_special_chars).str.lower()
    df['Stems_y'] = df['Stems_y'].str.lower()
    df['known_lemma'] = df['known_lemma'].str.lower()

    got_token_ipa = df.Token.apply(transcribe, transcriber=got_tr)
    got_lemma_ipa = df.Lemma.apply(transcribe, transcriber=got_tr)
    got_stem_ipa = df.Stems_x.apply(transcribe_stem, transcriber=got_tr, lang='got')

    ang_lemma_ipa = df.known_lemma.progress_apply(transcribe_multiple, transcriber=ang_tr)
    ang_stem_ipa = df.Stems_y.progress_apply(transcribe_multiple_stems, transcriber=ang_tr, lang='ang')

    df['lost_lang'] = 'got'
    df['lost_ipa'] = got_token_ipa
    df['lost_lemma_ipa'] = got_lemma_ipa
    df['known_lemma_ipa'] = ang_lemma_ipa
    df['known_lang'] = 'ang'

    # Combine multiple relevant columns into a serialized string to be loaded by the full model later.
    def safe_combine(item):
        lang, token, ipa = item
        if not pd.isnull(lang) and not pd.isnull(token) and not pd.isnull(ipa):
            t_segs = token.split("|")
            ipa_segs = ipa.split('|')
            assert len(t_segs) == len(ipa_segs)
            return '|'.join(f'{lang};{t};{{{i}}}' for t, i in zip(t_segs, ipa_segs) if len(i) > 0)

    def combine(item):
        lang, token, ipa = item
        assert not pd.isnull(lang) and not pd.isnull(token) and not pd.isnull(ipa) and len(ipa) > 0
        return f'{lang};{token};{{{ipa}}}'

    df = df[df['lost_ipa'].apply(len) > 0]

    df['lost_token'] = df[['lost_lang', 'Token', 'lost_ipa']].apply(combine, axis=1)
    df['lost_lemma'] = df[['lost_lang', 'Lemma', 'lost_lemma_ipa']].apply(safe_combine, axis=1)
    df['known_tokens'] = df[['known_lang', 'known_lemma', 'known_lemma_ipa']].apply(safe_combine, axis=1)

    # Get the output df.
    out_df = pd.DataFrame({
        'sentence_idx': df.SegmentID,
        'word_idx': df.Position - 1,
        'lost_token': df.lost_token,
        'lost_lemma': df.lost_lemma,
        'lost_stems': got_stem_ipa,
        # These two are redundant now.
        'known_tokens': df.known_tokens,
        'known_lemmas': df.known_tokens,
        'known_stems': ang_stem_ipa
    })

    out_df['sentence_idx'] = out_df['sentence_idx'].astype('Int64')
    out_df['word_idx'] = out_df['word_idx'].astype('Int64')

    out_df.to_csv('data/wulfila/processed/corpus.got-ang.tsv', sep='\t', index=None)

    # Get a small output df.
    small_out_df = out_df[out_df.sentence_idx < 500]
    small_out_df.to_csv('data/wulfila/processed/corpus.small.got-ang.tsv', sep='\t', index=None)

    # Write out all known words into a file -- they form the known vocabulary.
    with open('tmp.stems', 'w', encoding='utf8') as fout:
        for stem in set(concat_lists(out_df.known_stems.str.split('|').dropna().values)):
            fout.write(stem + '\n')

    # A smaller vocabulary file.
    with open('data/wulfila/processed/ang.small.matched.stems', 'w', encoding='utf8') as fout:
        for stem in set(concat_lists(small_out_df.known_stems.str.split('|').dropna().values)):
            fout.write(stem + '\n')
