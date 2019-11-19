from unittest import TestCase

import torch

from .process import Segment


class TestSegment(TestCase):

    def test_basic(self):
        seg = Segment('θ-ɹ-iː')
        ans = torch.LongTensor(
            [[0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110]]
        )
        self.assertListEqual(seg.feat_matrix.cpu().numpy().tolist(), ans.cpu().numpy().tolist())
