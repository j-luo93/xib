import numpy as np
cimport numpy as np


B = <int>(0)
I = <int>(1)
O = <int>(2)
N = <int>(3)

cdef inline where(int value, int last_value, int next_value):
    start = (value == B) or (value == I and (last_value == N or last_value == O))
    add = (value == B) or (value == I)
    wrap_up = add and (next_value != I)
    return start, add, wrap_up

cdef first_pass(samples):
    batch_size, num_samples, max_len = samples.shape
    word_counts = np.zeros([batch_size, num_samples], dtype=int)
    max_lengths = np.zeros([batch_size, num_samples], dtype=int)
    offsets = np.zeros([batch_size * num_samples], dtype=int)

    cdef long length
    cdef long max_length
    cdef long count
    cdef long last_value
    cdef long next_value

    for i in range(batch_size):
        for j in range(num_samples):
            length = 0
            max_length = 0
            count = 0
            last_value = N
            next_value = samples[i, j, 0]
            for k in range(max_len):
                value = next_value
                if k < max_len - 1:
                    next_value = samples[i, j, k + 1]
                else:
                    next_value = N

                start, add, wrap_up = where(value, last_value, next_value)
                if start:
                    count += 1
                    length = 0
                if add:
                    length += 1
                if wrap_up:
                    max_length = max(max_length, length)
                    length = 0

                last_value = value

            word_counts[i, j] = count
            max_lengths[i, j] = max_length

    # Compute offsets.
    accum_counts = np.cumsum(word_counts.reshape(-1))

    offsets[1:] = accum_counts[:-1]
    offsets = offsets.reshape([batch_size, num_samples])
    return word_counts, max_lengths, offsets


cdef second_pass(samples, offsets, int total_num_words, int max_word_len):
    batch_indices = np.zeros([total_num_words], dtype=int)
    sample_indices = np.zeros([total_num_words], dtype=int)
    word_positions = np.zeros([total_num_words, max_word_len], dtype=int)
    word_lengths = np.zeros([total_num_words], dtype=np.uint32)

    batch_size, num_samples, max_len = samples.shape
    for i in range(batch_size):
        for j in range(num_samples):
            offset = offsets[i, j]
            length = 0
            last_value = N
            next_value = samples[i, j, 0]
            for k in range(max_len):
                value = next_value
                if k < max_len - 1:
                    next_value = samples[i, j, k + 1]
                else:
                    next_value = N
                start, add, wrap_up = where(value, last_value, next_value)
                if start:
                    batch_indices[offset] = i
                    sample_indices[offset] = j
                    length = 0
                if add:
                    word_positions[offset, length] = k
                    length += 1
                if wrap_up:
                    word_lengths[offset] = length
                    length = 0
                    offset += 1

                last_value = value
    return batch_indices, sample_indices, word_positions, word_lengths


def extract_words_v2(samples):
    # First pass to calculate total number of words and max length of words.
    word_counts, max_lengths, offsets = first_pass(samples)
    cdef long total_num_words = word_counts.sum()
    cdef long max_word_len = max_lengths.max()

    # Now we can extract words.
    batch_indices, sample_indices, word_positions, word_lengths = second_pass(
        samples, offsets, total_num_words, max_word_len)

    return batch_indices, sample_indices, word_positions, word_lengths
