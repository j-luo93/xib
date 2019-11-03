from unittest import TestCase

from . import ipax

import numpy as np


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
            ipax.Ptype, ipax.CVoicingX,
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
            ans = np.zeros([len(e), len(e)])
            ans[0] = 1.0
            ans[:, 0] = 1.0
            ans[0, 0] = 0.0
            for i in range(1, len(e)):
                for j in range(1, len(e)):
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
            print(ans)
