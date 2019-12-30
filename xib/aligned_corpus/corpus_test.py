from unittest.mock import MagicMock

from ipapy.ipastring import IPAString

from dev_misc import TestCase, test_with_arguments

from .corpus import (AlignedSentence, AlignedWord, OverlappingAnnotation,
                     UnsegmentedSentence, Word, WordFactory)


class BaseTest(TestCase):

    def setUp(self):
        super().setUp()
        WordFactory.clear_cache()


class TestWord(BaseTest):

    def test_basic(self):
        wf = WordFactory()
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.side_effect = [
            {IPAString(unicode_string='a')},
            {IPAString(unicode_string='an'), IPAString(unicode_string='ann')}
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
            return {IPAString(unicode_string=unicode_string)}
        self.mock_transcriber.transcribe.side_effect = mock_transcribe

    def test_basic(self):
        aligned_word = AlignedWord.from_raw_string('en', 'de', 'good|gut', self.mock_transcriber)
        self.assertEqual(len(list(aligned_word.lost_word.ipa)[0]), 4)
        self.assertEqual(len(list(aligned_word.known_word.ipa)[0]), 2)

    def test_missing_known_form(self):
        aligned_word = AlignedWord.from_raw_string('en', 'de', 'good', self.mock_transcriber)
        self.assertEqual(len(list(aligned_word.lost_word.ipa)[0]), 4)
        self.assertEqual(aligned_word.known_word, None)


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
        mock_transcriber.transcribe.return_value = {IPAString(unicode_string='abc')}
        self.aligned_sentence = AlignedSentence.from_raw_string('en', 'de', 'good|gut bad', mock_transcriber)

    def test_basic(self):
        self.assertEqual(len(self.aligned_sentence), 2)

    def test_to_unsegmented(self):
        for annotated in [True, False]:
            uss = self.aligned_sentence.to_unsegmented(annotated=annotated, is_ipa=False)
            self.assertEqual(len(uss), 7)

        uss = self.aligned_sentence.to_unsegmented(annotated=True, is_ipa=False)
        self.assertSetEqual(uss.annotated, {0, 1, 2, 3})
