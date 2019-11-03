from unittest import TestCase

import numpy as np

from . import ipa, ipax


def _test_routine(e: ipax.DistEnum, ans: np.ndarray):
    dist_mat = e.get_distance_matrix()
    np.testing.assert_array_almost_equal(dist_mat, ans)


def _get_dist(value1, value2):
    dist = 0.0
    for name, field in value1.__dataclass_fields__.items():
        cls = field.type
        feat1 = getattr(value1, name)
        feat2 = getattr(value2, name)
        dist += cls.get_distance(feat1, feat2)
    return dist


class TestIpax(TestCase):

    def test_basic_discrete(self):
        to_test = [
            ipax.PtypeX, ipax.CVoicingX,
            ipax.VRoundnessX, ipax.CMannerNasality,
            ipax.CMannerLaterality, ipax.CMannerAirstream,
            ipax.CMannerSibilance, ipax.CMannerVibrancy
        ]
        ans = np.asarray(
            [[0.0, 1.0, 1.0],
             [1.0, 0.0, 1.0],
             [1.0, 1.0, 0.0]]
        )
        for e in []:
            _test_routine(e, ans)

    def test_basic_continuous(self):
        to_test = [
            ipax.VBacknessX, ipax.CPlaceActiveArticulator,
            ipax.CPlacePassiveArticulator, ipax.CMannerSonority,
            ipax.VHeightX, ipax.VBacknessX
        ]
        for e in to_test:
            v = 1 / (len(e) - 2)
            if e in [ipax.CPlaceActiveArticulator, ipax.CPlacePassiveArticulator]:
                v = 1 / (len(e) - 6)
            ans = np.zeros([len(e), len(e)])
            ans[0] = 1.0
            ans[:, 0] = 1.0
            ans[0, 0] = 0.0
            for i in range(1, len(e)):
                for j in range(1, len(e)):
                    if e(i).name.startswith('DISC_') or e(j).name.startswith('DISC_'):
                        if i == j:
                            ans[i, j] = 0
                        else:
                            ans[i, j] = 1.0
                    else:
                        ans[i, j] = abs(i - j) * v
            _test_routine(e, ans)

    def test_complex_continuous(self):
        for e in [ipax.CPlaceX]:  # , ipax.CMannerX]:
            n = len(e)
            ans = np.zeros([n, n])
            for i, f1 in enumerate(e):
                for j, f2 in enumerate(e):
                    ans[i, j] = _get_dist(f1.value, f2.value)
            inf_mask = ans == np.inf
            ans[inf_mask] = 0.0
            ans = ans / ans.max()
            ans[inf_mask] = 1.0
            _test_routine(e, ans)

    def test_conversion(self):
        for cat in ipax.CategoryX:
            e = ipax.CategoryX.get_enum(cat.name)
            old_e = ipa.get_enum_by_cat(ipa.Category[cat.name.rstrip('_X')])
            for feat in e:
                idx = old_e[feat.name].value
                self.assertEqual(ipax.conversions[idx], feat)
                self.assertEqual(ipax.reverse_conversions[feat], idx)
