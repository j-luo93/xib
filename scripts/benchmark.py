import cProfile
import pstats
import sys
import timeit

import numpy as np
import pyximport

from devlib import pad_to_dense
from xib.extract_words_impl import extract_words_v8


B, I, O, N = 0, 1, 2, 3


def where(value, last_value, next_value):
    start = (value == B) or (value == I and (last_value == None or last_value == O))
    add = (value == B) or (value == I)
    wrap_up = add and (next_value != I)
    return start, add, wrap_up


def get_random_test(size, array_only=False):
    np.random.seed(1234)
    arr = np.random.randint(0, high=3, size=size, dtype=np.int32)
    sample_lengths = np.random.randint(0, high=3, size=size[:-1], dtype=np.int32) + 1
    if array_only:
        return arr, sample_lengths
    batch_size, num_samples, max_len = arr.shape

    batch_indices = list()
    sample_indices = list()
    word_positions = list()
    word_lengths = list()
    is_unique = list()

    for i in range(batch_size):
        all_samples = set()
        unique = list()
        for j in range(num_samples):
            sample = tuple(arr[i, j, :sample_lengths[i, j]])
            if sample not in all_samples:
                unique.append(1)
                all_samples.add(sample)
            else:
                unique.append(0)

            last_value = None
            next_value = arr[i, j, 0]
            word = list()
            sample_length = sample_lengths[i, j]
            for k in range(sample_length):
                value = next_value
                if k < sample_length - 1:
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
        is_unique.append(unique)

    batch_indices = np.asarray(batch_indices)
    sample_indices = np.asarray(sample_indices)
    word_positions = pad_to_dense(word_positions, dtype=np.int32)
    word_lengths = np.asarray(word_lengths)
    is_unique = np.asarray(is_unique, dtype=np.int32)

    return arr, sample_lengths, batch_indices, sample_indices, word_positions, word_lengths, is_unique


if __name__ == "__main__":

    # print(timeit.timeit('get_random_test([200, 100, 20])', 'from __main__ import get_random_test', number=10))

    # This is for linetracing. Don't forget to add linetrace=1 to pyx file.
    # pyximport.install()

    # arr, _, _, _, _ = get_random_test([200, 100, 20])
    # cProfile.runctx(
    #     "extract_words_v7(arr)", globals(), locals(), "Profile.prof")

    # s = pstats.Stats("Profile.prof")
    # s.strip_dirs().sort_stats("time").print_stats()
    num_threads = int(sys.argv[1])

    print(timeit.timeit(
        f'extract_words_v8(arr, sample_lengths, num_threads={num_threads})',
        'from __main__ import extract_words_v8, get_random_test; arr, sample_lengths = get_random_test([2000, 100, 20], array_only=True)', number=100))

    arr, sample_lengths, batch_indices, sample_indices, word_positions, word_lengths, is_unique = get_random_test([
                                                                                                                  200, 10, 3])

    ret_batch_indices, ret_sample_indices, ret_word_positions, ret_word_lengths, ret_is_unique = extract_words_v8(
        arr, sample_lengths)

    # or not np.array_equal(is_unique, ret_is_unique):
    if not np.array_equal(batch_indices, ret_batch_indices) or not np.array_equal(sample_indices, ret_sample_indices) or not np.array_equal(word_positions, ret_word_positions)or not np.array_equal(word_lengths, ret_word_lengths) or not np.array_equal(is_unique, ret_is_unique):
        raise RuntimeError('Did not pass!')
    else:
        print('Passed!')
