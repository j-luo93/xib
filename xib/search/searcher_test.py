import torch

from dev_misc import TestCase, test_with_arguments
from xib.search.searcher import BeamSearcher


class TestBeamSearcher(TestCase):

    def test_basic(self):
        test_with_arguments(beam_size=3, _force=True)
        lengths = torch.LongTensor([3, 4]).rename('batch')
        label_log_probs = torch.FloatTensor(
            [
                [
                    [70, 20, 10],
                    [60, 30, 5],
                    [70, 20, 10],
                    [70, 20, 10],
                ],
                [
                    [10, 70, 20],
                    [30, 5, 60],
                    [10, 70, 10],
                    [10, 70, 10],
                ]
            ]
        ).rename('batch', 'length', 'label')
        searcher = BeamSearcher()
        samples, sample_log_probs = searcher.search(lengths, label_log_probs)
        ans_samples = [
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
            ],
            [
                [1, 2, 1, 1],
                [1, 0, 1, 1],
                [2, 2, 1, 1]
            ]
        ]
        self.assertArrayEqual(samples, ans_samples)
        self.assertArrayEqual(sample_log_probs, [[270, 240, 220], [270, 240, 220]])
