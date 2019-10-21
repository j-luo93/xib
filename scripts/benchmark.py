import cProfile
import pstats
import timeit

import numpy as np
import pyximport

from devlib import pad_to_dense
from xib.extract_words import B, I, O, extract_words_py, where
from xib.extract_words_impl import extract_words_v5


def get_random_test(size):
    np.random.seed(1234)
    arr = np.random.randint(0, high=3, size=size, dtype=np.int32)
    batch_size, num_samples, max_len = arr.shape

    batch_indices = list()
    sample_indices = list()
    word_positions = list()
    word_lengths = list()

    for i in range(batch_size):
        for j in range(num_samples):
            last_value = None
            next_value = arr[i, j, 0]
            word = list()
            for k in range(max_len):
                value = next_value
                if k < max_len - 1:
                    next_value = arr[i, j, k + 1]
                else:
                    next_value = None

                start, add, wrap_up = where(value, last_value, next_value)
                if start:
                    batch_indices.append(i)
                    sample_indices.append(j)
                    word = list()
                if add:
                    word.append(k)
                if wrap_up:
                    assert word
                    word_positions.append(word)
                    word_lengths.append(len(word))
                    word = list()

                last_value = value

    batch_indices = np.asarray(batch_indices)
    sample_indices = np.asarray(sample_indices)
    word_positions = pad_to_dense(word_positions, dtype=np.int32)
    word_lengths = np.asarray(word_lengths)

    return arr, batch_indices, sample_indices, word_positions, word_lengths


if __name__ == "__main__":

    # print(timeit.timeit('get_random_test([200, 100, 20])', 'from __main__ import get_random_test', number=10))

    # This is for linetracing. Don't forget to add linetrace=1 to pyx file.
    # pyximport.install()

    # arr, _, _, _, _ = get_random_test([200, 100, 20])
    # cProfile.runctx(
    #     "extract_words_v5(arr)", globals(), locals(), "Profile.prof")

    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()

    print(timeit.timeit(
        'extract_words_v5(arr)',
        'from __main__ import extract_words_v5, get_random_test; arr, _, _, _, _ = get_random_test([200, 100, 20])', number=10))

    # arr, batch_indices, sample_indices, word_positions, word_lengths = get_random_test([200, 100, 20])

    # ret_batch_indices, ret_sample_indices, ret_word_positions, ret_word_lengths = extract_words_v5(arr)

    # if not np.array_equal(batch_indices, ret_batch_indices) or not np.array_equal(sample_indices, ret_sample_indices) or not np.array_equal(word_positions, ret_word_positions)or not np.array_equal(word_lengths, ret_word_lengths):
    #     raise RuntimeError('Did not pass!')
    # else:
    #     print('Passed!')
