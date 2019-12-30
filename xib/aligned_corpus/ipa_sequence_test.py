from ipapy.ipastring import IPAString

from dev_misc import TestCase, test_with_arguments

from .ipa_sequence import IpaSequence


class TestIpaSequence(TestCase):

    def test_basic(self):
        raw_string = 'a:bc'
        ipa_seq = IpaSequence(raw_string)
        self.assertEqual(len(ipa_seq), 3)
        self.assertEqual(str(ipa_seq), 'aːbc')
        self.assertEqual(ipa_seq[0], str(IPAString(unicode_string='aː')))
        self.assertEqual(ipa_seq[1], str(IPAString(unicode_string='b')))
        self.assertEqual(ipa_seq[2], str(IPAString(unicode_string='c')))
