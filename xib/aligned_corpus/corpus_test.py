from unittest.mock import MagicMock

from ipapy.ipastring import IPAString

from dev_misc import TestCase, test_with_arguments

from .corpus import (AlignedSentence, AlignedWord, OverlappingAnnotation,
                     UnsegmentedSentence, Word, WordFactory)


class BaseTest(TestCase):

    def setUp(self):
        super().setUp()
        test_with_arguments(max_word_length=10, min_word_length=1, postprocess_mode='none', _force=True)
        WordFactory.clear_cache()


class TestWord(BaseTest):

    def test_basic(self):
        wf = WordFactory()
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.side_effect = [
            {'a'},
            {'an', 'ann'}
        ]
        w0 = wf.get_word('en', 'a', mock_transcriber)
        w1 = wf.get_word('en', 'an', mock_transcriber)
        self.assertEqual(len(w0.ipa), 1)
        self.assertEqual(len(w1.ipa), 2)


class TestAlignedWord(BaseTest):

    def setUp(self):
        super().setUp()
        self.mock_transcriber = MagicMock()

        def mock_transcribe(form, lang):
            if lang == 'en':
                unicode_string = 'aaaa'
            else:
                unicode_string = 'bb'
            return {unicode_string}

        self.mock_transcriber.transcribe.side_effect = mock_transcribe

    def test_basic(self):
        aligned_word = AlignedWord.from_raw_string('en', 'de', 'good|good|gut|', self.mock_transcriber)
        self.assertEqual(len(aligned_word.lost_token.main_ipa), 4)
        self.assertEqual(len(list(aligned_word.known_tokens)[0].main_ipa), 2)

    def test_missing_known_form(self):
        aligned_word = AlignedWord.from_raw_string('en', 'de', 'good|good||', self.mock_transcriber)
        self.assertEqual(len(aligned_word.lost_token.main_ipa), 4)
        self.assertEqual(len(aligned_word.known_tokens), 0)


class TestUnsegmentedSentence(BaseTest):

    def test_basic(self):
        uss = UnsegmentedSentence('abcdefghi', False)
        self.assertEqual(len(uss), 9)
        uss.annotate(0, 3, 'ABCD')
        uss.annotate(4, 5, 'EF')
        with self.assertRaises(OverlappingAnnotation):
            uss.annotate(5, 6, 'FG')
        self.assertSetEqual(uss.annotated, {0, 1, 2, 3, 4, 5})


class TestAlignedSentence(BaseTest):

    def setUp(self):
        super().setUp()
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = {'abc'}
        self.aligned_sentence = AlignedSentence.from_raw_string(
            'en', 'de', 'good|good|gut|gut bad|bad||', mock_transcriber)

    def test_basic(self):
        self.assertEqual(len(self.aligned_sentence.words), 2)

    def test_to_unsegmented(self):
        for annotated in [True, False]:
            uss = self.aligned_sentence.to_unsegmented(annotated=annotated, is_ipa=False)
            self.assertEqual(len(uss), 7)

        uss = self.aligned_sentence.to_unsegmented(annotated=True, is_ipa=False)
        self.assertSetEqual(uss.annotated, {0, 1, 2, 3})
