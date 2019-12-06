from unittest import TestCase
from unittest.mock import patch

import torch

from dev_misc import TestCase, test_with_arguments

from .process import B, I, O, Segment, SegmentWindow


class TestSegment(TestCase):

    def setUp(self):
        super().setUp()
        self.seg = Segment('θɹiː')
        test_with_arguments(min_word_length=1, _force=True)

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

    def test_broken_segment(self):
        new_seg = self.seg.break_segment(1, 2)

        self.assertListEqual(new_seg.segment_list, ['ɹ', 'iː'])
        self.assertArrayEqual(new_seg.gold_tag_seq, [O, O])

        new_seg = self.seg.break_segment(0, 1)

        self.assertListEqual(new_seg.segment_list, ['θ', 'ɹ'])
        self.assertArrayEqual(new_seg.gold_tag_seq, [O, O])

        new_seg = self.seg.break_segment(0, 2)
        self.assertIs(new_seg, self.seg)

    def test_cv_list(self):
        self.assertListEqual(self.seg.cv_list, ['θ', 'ɹ', 'i'])


class TestSegmentWindow(TestCase):

    def setUp(self):
        super().setUp()
        seg1 = Segment('θɹiː')
        seg2 = Segment('θɹiː')
        self.sw = SegmentWindow([seg1, seg2])
        test_with_arguments(min_word_length=1, _force=True)

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

    def test_perturb_swap(self):
        with patch('xib.ipa.process.random.randint') as mock_rand_func:
            mock_rand_func.return_value = 3
            seg = self.sw.perturb_swap()
            self.assertListEqual(seg.segment_list, ['θ', 'ɹ', 'iː', 'ɹ', 'θ', 'iː'])
            self.assertArrayEqual(seg.feat_matrix[3], seg.feat_matrix[1])

            mock_rand_func.return_value = 0
            seg = self.sw.perturb_swap()
            self.assertListEqual(seg.segment_list, ['ɹ', 'θ', 'iː', 'θ', 'ɹ', 'iː'])
            self.assertArrayEqual(seg.feat_matrix[4], seg.feat_matrix[0])

            mock_rand_func.return_value = 4
            seg = self.sw.perturb_swap()
            self.assertListEqual(seg.segment_list, ['θ', 'ɹ', 'iː', 'θ', 'iː', 'ɹ'])
            self.assertArrayEqual(seg.feat_matrix[5], seg.feat_matrix[1])

    def test_perturb_shift(self):
        with patch('xib.ipa.process.random.randint') as mock_rand_func:
            mock_rand_func.return_value = 3
            seg = self.sw.perturb_shift()
            self.assertListEqual(seg.segment_list, ['θ', 'ɹ', 'iː', 'θ', 'ɹ', 'iː'])
            self.assertArrayEqual(seg.feat_matrix[3], seg.feat_matrix[0])

            mock_rand_func.return_value = 1
            seg = self.sw.perturb_shift()
            self.assertListEqual(seg.segment_list, ['iː', 'θ', 'ɹ', 'iː', 'θ', 'ɹ'])
            self.assertArrayEqual(seg.feat_matrix[1], seg.feat_matrix[4])

            mock_rand_func.return_value = 5
            seg = self.sw.perturb_shift()
            self.assertListEqual(seg.segment_list, ['ɹ', 'iː', 'θ', 'ɹ', 'iː', 'θ'])
            self.assertArrayEqual(seg.feat_matrix[5], seg.feat_matrix[2])

    def test_perturb_n_times(self):
        with patch('xib.ipa.process.random.randint') as mock_rand_func:
            mock_rand_func.side_effect = [3, 3, 3, 3, 3]
            segments, duplicated = self.sw.perturb_n_times(2)
            self.assertListEqual(segments[0].segment_list, ['θ', 'ɹ', 'iː', 'θ', 'ɹ', 'iː'])
            self.assertListEqual(segments[1].segment_list, ['θ', 'ɹ', 'iː', 'ɹ', 'θ', 'iː'])
            self.assertListEqual(segments[2].segment_list, ['θ', 'ɹ', 'iː', 'θ', 'ɹ', 'iː'])
            self.assertListEqual(segments[3].segment_list, ['θ', 'ɹ', 'iː', 'ɹ', 'θ', 'iː'])
            self.assertListEqual(segments[4].segment_list, ['θ', 'ɹ', 'iː', 'θ', 'ɹ', 'iː'])
            self.assertListEqual(duplicated, [False, False, True, True, True])

    def test_broken_segment(self):
        seg1 = Segment('θɹiː')
        seg2 = Segment('θɹiː')
        seg3 = Segment('θɹiː')
        sw = SegmentWindow([seg1, seg2, seg3])

        new_sw = sw.break_segment(1, 2)

        self.assertListEqual(new_sw.segment_list, ['ɹ', 'iː'])
        self.assertArrayEqual(new_sw.gold_tag_seq, [O, O])

        new_sw = sw.break_segment(0, 3)

        self.assertListEqual(new_sw.segment_list, ['θ', 'ɹ', 'iː', 'θ'])
        self.assertArrayEqual(new_sw.gold_tag_seq, [B, I, I, O])

        new_seg = sw.break_segment(1, 8)

        self.assertListEqual(new_seg.segment_list, ['ɹ', 'iː', 'θ', 'ɹ', 'iː', 'θ', 'ɹ', 'iː'])
        self.assertArrayEqual(new_seg.gold_tag_seq, [O, O, B, I, I, B, I, I])

    def test_cv_list(self):
        self.assertListEqual(self.sw.cv_list, ['θ', 'ɹ', 'i', 'θ', 'ɹ', 'i'])
