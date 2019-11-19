from unittest import TestCase
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from dev_misc.arglib import reset_repo, test_with_arguments
from dev_misc.devlib.named_tensor import (patch_named_tensors,
                                          unpatch_named_tensors)

from .data_loader import ContinuousTextDataLoader, IpaDataLoader

from pathlib import Path


class BaseDataLoaderTestCase(TestCase):

    def setUp(self):
        reset_repo()
        patch_named_tensors()

    def tearDown(self):
        unpatch_named_tensors()

    @patch('xib.data_loader.BatchSampler')
    def _test_routinue(self, dl_cls, content, mock_sampler_cls):
        data_path = MagicMock(autospec=Path)

        def get_mock_file(*args, **kwargs):
            mock_file = MagicMock()
            mock_file.__enter__.return_value = iter(content)
            mock_file.__exit__.return_value = True
            return mock_file

        data_path.open.side_effect = get_mock_file
        data_path.__class__ = Path

        mock_sampler = MagicMock()
        mock_sampler.__iter__.return_value = iter([[0, 1]])
        mock_sampler_cls.return_value = mock_sampler
        test_with_arguments(data_path=data_path, char_per_batch=100,
                            num_workers=0, feat_groups='pcv', _force=True)
        dl = dl_cls()
        cnt = 0
        for batch in dl:
            cnt += 1
        self.assertEqual(cnt, 1)
        return dl


class TestIpaDataLoader(BaseDataLoaderTestCase):

    def test_basic(self):
        self._test_routinue(IpaDataLoader, ('ab:c', 'a:bd'))


class TestContinuousIpaDataLoader(BaseDataLoaderTestCase):

    def test_basic(self):
        test_with_arguments(max_segment_length=10, _force=True)
        self._test_routinue(ContinuousTextDataLoader, ('ab:c', 'a:bd'))

    def test_window(self):
        test_with_arguments(max_segment_length=3, _force=True)
        dl = self._test_routinue(ContinuousTextDataLoader, ('ab:c ab b ac bcd b', 'a:bd'))

        def aeq(idx, ans):
            self.assertEqual(str(dl.dataset[idx]['segment']), ans)

        aeq(0, 'a-bː-c')
        aeq(1, 'a-b b')
        aeq(2, 'b a-c')
        aeq(3, 'b-c-d')
        aeq(4, 'b')
        aeq(5, 'aː-b-d')