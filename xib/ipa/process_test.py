from unittest import TestCase
from unittest.mock import patch

import torch

from dev_misc import TestCase, test_with_arguments

from .process import B, I, O, Segment, SegmentWindow


class TestSegment(TestCase):

    def setUp(self):
        super().setUp()
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

    def test_noisy(self):
        seg = Segment('#θɹiː')
        self.assertTrue(seg.is_noise)
        self.assertArrayEqual(seg.gold_tag_seq, [O, O, O])


class TestSegmentWindow(TestCase):

    def setUp(self):
        super().setUp()
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

    def test_to_segmentation(self):
        segmentation = self.sw.to_segmentation()
        span1, span2 = segmentation.spans
        self.assertEqual(span1.start, 0)
        self.assertEqual(span1.end, 2)
        self.assertEqual(span2.start, 3)
        self.assertEqual(span2.end, 5)

    def test_noisy(self):
        seg1 = Segment('#θɹiː')
        seg2 = Segment('θɹiː')
        sw = SegmentWindow([seg1, seg2])
        self.assertArrayEqual(sw.gold_tag_seq, [O, O, O, B, I, I])
        spans = sw.to_segmentation().spans
        self.assertEqual(len(spans), 1)
        span = spans[0]
        self.assertEqual(span.start, 3)
        self.assertEqual(span.end, 5)

    def test_get_segmentation_from_tags(self):
        tags = [B, I, O, I, B, O]
        seg = self.sw.get_segmentation_from_tags(tags)
        self.assertEqual(len(seg), 3)
        span1, span2, span3 = seg.spans
        self.assertEqual(span1.start, 0)
        self.assertEqual(span1.end, 1)
        self.assertEqual(span2.start, 3)
        self.assertEqual(span2.end, 3)
        self.assertEqual(span3.start, 4)
        self.assertEqual(span3.end, 4)

    def test_perturb(self):
        with patch('xib.ipa.process.random.randint') as mock_rand_func:
            mock_rand_func.return_value = 3
            seg = self.sw.perturb()
            self.assertListEqual(seg.segment_list, ['θ', 'ɹ', 'iː', 'ɹ', 'θ', 'iː'])
            self.assertArrayEqual(seg.feat_matrix[3], seg.feat_matrix[1])

            mock_rand_func.return_value = 0
            seg = self.sw.perturb()
            self.assertListEqual(seg.segment_list, ['ɹ', 'θ', 'iː', 'θ', 'ɹ', 'iː'])
            self.assertArrayEqual(seg.feat_matrix[4], seg.feat_matrix[0])

            mock_rand_func.return_value = 4
            seg = self.sw.perturb()
            self.assertListEqual(seg.segment_list, ['θ', 'ɹ', 'iː', 'θ', 'iː', 'ɹ'])
            self.assertArrayEqual(seg.feat_matrix[5], seg.feat_matrix[1])
