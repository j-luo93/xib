import numpy as np
cimport numpy as np

DTYPE = np.intc

B = <int>(0)
I = <int>(1)
O = <int>(2)
N = <int>(3)

cdef inline where(int value, int last_value, int next_value):
    start = (value == B) or (value == I and (last_value == N or last_value == O))
    add = (value == B) or (value == I)
    wrap_up = add and (next_value != I)
    return start, add, wrap_up

cdef first_pass(int[:, :, :] samples):
    cdef Py_ssize_t batch_size = samples.shape[0]
    cdef Py_ssize_t num_samples = samples.shape[1]
    cdef Py_ssize_t max_len = samples.shape[2]

    word_counts_storage = np.zeros([batch_size, num_samples], dtype=DTYPE)
    max_lengths_storage = np.zeros([batch_size, num_samples], dtype=DTYPE)
    cdef int[:, :] word_counts = word_counts_storage
    cdef int[:, :] max_lengths = max_lengths_storage

    cdef int length
    cdef int max_length
    cdef int count
    cdef int last_value
    cdef int next_value

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
    offsets_storage = np.zeros([batch_size * num_samples], dtype=np.int_)
    cdef long[:] offsets = offsets_storage
    cdef long[:] accum_counts = np.cumsum(word_counts_storage.reshape(-1))

    offsets[1:] = accum_counts[:-1]
    offsets_2d_storage = offsets_storage.reshape([batch_size, num_samples])
    return word_counts_storage, max_lengths_storage, offsets_2d_storage


cdef second_pass(int[:, :, :] samples, long[:, :] offsets, long total_num_words, int max_word_len):
    batch_indices_storage = np.zeros([total_num_words], dtype=DTYPE)
    sample_indices_storage = np.zeros([total_num_words], dtype=DTYPE)
    word_positions_storage = np.zeros([total_num_words, max_word_len], dtype=DTYPE)
    word_lengths_storage = np.zeros([total_num_words], dtype=DTYPE)
    cdef int[:] batch_indices = batch_indices_storage
    cdef int[:] sample_indices = sample_indices_storage
    cdef int[:, :] word_positions = word_positions_storage
    cdef int[:] word_lengths = word_lengths_storage

    cdef Py_ssize_t batch_size = samples.shape[0]
    cdef Py_ssize_t num_samples = samples.shape[1]
    cdef Py_ssize_t max_len = samples.shape[2]

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
    return batch_indices_storage, sample_indices_storage, word_positions_storage, word_lengths_storage


def extract_words_v3(int[:, :, :] samples):
    # First pass to calculate total number of words and max length of words.
    word_counts, max_lengths, offsets = first_pass(samples)
    cdef long total_num_words = word_counts.sum()
    cdef int max_word_len = max_lengths.max()

    # Now we can extract words.
    batch_indices, sample_indices, word_positions, word_lengths = second_pass(
        samples, offsets, total_num_words, max_word_len)

    return batch_indices, sample_indices, word_positions, word_lengths
