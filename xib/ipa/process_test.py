from unittest import TestCase

import torch

from .process import Segment, B, I, O, SegmentWindow


class TestSegment(TestCase):

    def setUp(self):
        self.seg = Segment('θɹiː')

    def test_basic(self):
        ans = torch.LongTensor(
            [[0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110]]
        )
        self.assertListEqual(self.seg.feat_matrix.cpu().numpy().tolist(), ans.cpu().numpy().tolist())

    def test_gold_tag_seq(self):
        self.assertListEqual(self.seg.gold_tag_seq.cpu().numpy().tolist(), [B, I, I])

    def test_str(self):
        self.assertEqual(str(self.seg), 'θ-ɹ-iː')

    def test_getitem(self):
        self.assertEqual(self.seg[2], 'iː')

    def test_to_span(self):
        span = self.seg.to_span()
        self.assertEqual(span.start, 0)
        self.assertEqual(span.end, 2)


class TestSegmentWindow(TestCase):

    def setUp(self):
        seg1 = Segment('θɹiː')
        seg2 = Segment('θɹiː')
        self.sw = SegmentWindow([seg1, seg2])

    def test_basic(self):
        ans = torch.LongTensor(
            [[0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110],
             [0, 4, 9, 38, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [0, 3, 6, 23, 43, 51, 57, 60, 87, 89, 93, 96, 102, 110],
             [1, 2, 5, 22, 44, 54, 59, 60, 87, 92, 93, 96, 102, 110]]
        )
        self.assertListEqual(self.sw.feat_matrix.cpu().numpy().tolist(), ans.cpu().numpy().tolist())

    def test_gold_tag_seq(self):
        self.assertListEqual(self.sw.gold_tag_seq.cpu().numpy().tolist(), [B, I, I, B, I, I])
        self.assertEqual(str(self.sw), 'θ-ɹ-iː θ-ɹ-iː')

    def test_getitem(self):
        self.assertEqual(self.sw[3], 'θ')

    def test_getitem(self):
        self.assertEqual(self.sw[3], 'θ')

    def test_to_segmentation(self):
        segmentation = self.sw.to_segmentation()
        span1, span2 = segmentation.spans
        self.assertEqual(span1.start, 0)
        self.assertEqual(span1.end, 2)
        self.assertEqual(span2.start, 3)
        self.assertEqual(span2.end, 5)
