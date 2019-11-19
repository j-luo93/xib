from unittest import TestCase

import torch

from import Segment, B, I, O, SegmentWindow


class TestSegment(TestCase):

    def test_basic(self):
        seg = Segment('θɹiː')
        ans = torch.LongTensor(
            [[0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110]]
        )
        self.assertListEqual(seg.feat_matrix.cpu().numpy().tolist(), ans.cpu().numpy().tolist())

    def test_gold_tag_seq(self):
        seg = Segment('θɹiː')
        self.assertListEqual(seg.gold_tag_seq.cpu().numpy().tolist(), [B, I, I])

    def test_str(self):
        seg = Segment('θɹiː')
        self.assertEqual(str(seg), 'θ-ɹ-iː')


class TestSegmentWindow(TestCase):

    def test_basic(self):
        seg1 = Segment('θɹiː')
        seg2 = Segment('θɹiː')
        sw = SegmentWindow([seg1, seg2])
        ans = torch.LongTensor(
            [[0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110],
             [0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110]]
        )
        self.assertListEqual(sw.feat_matrix.cpu().numpy().tolist(), ans.cpu().numpy().tolist())

    def test_gold_tag_seq(self):
        seg1 = Segment('θɹiː')
        seg2 = Segment('θɹiː')
        sw = SegmentWindow([seg1, seg2])
        self.assertListEqual(sw.gold_tag_seq.cpu().numpy().tolist(), [B, I, I, B, I, I])
        self.assertEqual(str(sw), 'θ-ɹ-iː θ-ɹ-iː')
