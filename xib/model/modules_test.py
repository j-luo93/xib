from unittest import TestCase

import torch

from dev_misc.arglib import reset_repo
from xib.ipa import Name, ipa, ipax

from .modules import Predictor


class TestPredictor(TestCase):

    def setUp(self):
        reset_repo()

    def test_predictor_new_style(self):
        pred = Predictor(20, 'pcv', True)
        h = torch.randn(32, 20)
        ret = pred(h)
        enums = {
            ipax.PtypeX, ipax.CVoicingX, ipax.CMannerX,
            ipax.CPlaceX, ipax.VHeightX, ipax.VBacknessX,
            ipax.VRoundnessX
        }
        names = {e.get_name().value: e.get_name() for e in enums}
        self.assertSetEqual(
            set(ret.keys()),
            set(names.values())
        )
        self.assertTupleEqual(ret[names['PtypeX']].shape, (32, 2))
        self.assertTupleEqual(ret[names['CVoicingX']].shape, (32, 3))
        self.assertTupleEqual(ret[names['CMannerX']].shape, (32, 21))
        self.assertTupleEqual(ret[names['CPlaceX']].shape, (32, 17))
        self.assertTupleEqual(ret[names['VHeightX']].shape, (32, 8))
        self.assertTupleEqual(ret[names['VBacknessX']].shape, (32, 6))
        self.assertTupleEqual(ret[names['VRoundnessX']].shape, (32, 3))

    def test_predictor_old_style(self):
        pred = Predictor(20, 'pcv', False)
        h = torch.randn(32, 20)
        ret = pred(h)
        enums = {
            ipa.Ptype, ipa.CVoicing, ipa.CManner,
            ipa.CPlace, ipa.VHeight, ipa.VBackness,
            ipa.VRoundness
        }
        names = {e.get_name().value: e.get_name() for e in enums}
        self.assertSetEqual(
            set(ret.keys()),
            set(names.values())
        )
        self.assertTupleEqual(ret[names['Ptype']].shape, (32, 2))
        self.assertTupleEqual(ret[names['CVoicing']].shape, (32, 3))
        self.assertTupleEqual(ret[names['CManner']].shape, (32, 21))
        self.assertTupleEqual(ret[names['CPlace']].shape, (32, 17))
        self.assertTupleEqual(ret[names['VHeight']].shape, (32, 8))
        self.assertTupleEqual(ret[names['VBackness']].shape, (32, 6))
        self.assertTupleEqual(ret[names['VRoundness']].shape, (32, 3))
