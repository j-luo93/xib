# distutils: language = c++

import cython
from cython.parallel import prange

import numpy as np
cimport numpy as np
cimport openmp

from libcpp.set cimport set
from libcpp.vector cimport vector

DTYPE = np.intc

cdef int B = 0
cdef int I = 1
cdef int O = 2
cdef int N = 3

cdef inline (bint, bint, bint) where(int value, int last_value, int next_value) nogil:
    cdef bint start = (value == B) or (value == I and (last_value == N or last_value == O))
    cdef bint add = (value == B) or (value == I)
    cdef bint wrap_up = add and (next_value != I)
    return start, add, wrap_up

cdef first_pass(const int[:, :, ::1] samples, const int[:, ::1] sample_lengths, int num_threads):
    cdef Py_ssize_t batch_size = samples.shape[0]
    cdef Py_ssize_t num_samples = samples.shape[1]
    cdef Py_ssize_t max_len = samples.shape[2]

    word_counts_storage = np.zeros([batch_size, num_samples], dtype=DTYPE)
    max_lengths_storage = np.zeros([batch_size, num_samples], dtype=DTYPE)
    cdef int[:, ::1] word_counts = word_counts_storage
    cdef int[:, ::1] max_lengths = max_lengths_storage

    cdef int length
    cdef int max_length
    cdef int count
    cdef int last_value
    cdef int next_value
    cdef int value

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef bint start
    cdef bint add
    cdef bint wrap_up

    is_unique_storage = np.zeros([batch_size, num_samples], dtype=DTYPE)
    cdef int[:, ::1] is_unique = is_unique_storage

    cdef vector[set[vector[int]]] all_samples_by_thread = vector[set[vector[int]]]()
    cdef vector[vector[int]] this_sample_by_thread =  vector[vector[int]]()
    cdef int thread_number
    cdef int sample_length
    cdef bint in_set

    for _ in range(num_threads):
        all_samples_by_thread.push_back(set[vector[int]]())
        this_sample_by_thread.push_back(vector[int]())

    for i in prange(batch_size, nogil=True):
        # Clear the sample set for the thread.
        thread_number = openmp.omp_get_thread_num()
        all_samples_by_thread.at(thread_number).clear()

        for j in range(num_samples):
            this_sample_by_thread.at(thread_number).clear()
            sample_length = sample_lengths[i, j]
            length = 0
            max_length = 0
            count = 0
            last_value = N
            next_value = samples[i, j, 0]
            for k in range(sample_length):
                value = next_value
                if k + 1 < sample_length:
                    next_value = samples[i, j, k + 1]
                else:
                    next_value = N

                start, add, wrap_up = where(value, last_value, next_value)
                if start:
                    count = count + 1
                    length = 0
                if add:
                    length = length + 1
                if wrap_up:
                    max_length = max(max_length, length)
                    length = 0

                last_value = value

                this_sample_by_thread.at(thread_number).push_back(value)

            word_counts[i, j] = count
            max_lengths[i, j] = max_length

            # Compute whether this sample is unique.
            in_set = all_samples_by_thread.at(thread_number).count(this_sample_by_thread.at(thread_number))
            if not in_set:
                is_unique[i, j] = 1
                all_samples_by_thread.at(thread_number).insert(this_sample_by_thread.at(thread_number))

    # Compute offsets.
    offsets_storage = np.zeros([batch_size * num_samples], dtype=np.int_)
    cdef long[::1] offsets = offsets_storage
    cdef long[:] accum_counts = np.cumsum(word_counts_storage.reshape(-1))

    offsets[1:] = accum_counts[:-1]
    offsets_2d_storage = offsets_storage.reshape([batch_size, num_samples])
    return word_counts_storage, max_lengths_storage, offsets_2d_storage, is_unique_storage

cdef second_pass(
        const int[:, :, ::1] samples,
        const int [:, ::1] sample_lengths,
        const long[:, ::1] offsets,
        long total_num_words,
        int max_word_len):
    batch_indices_storage = np.zeros([total_num_words], dtype=DTYPE)
    sample_indices_storage = np.zeros([total_num_words], dtype=DTYPE)
    word_positions_storage = np.zeros([total_num_words, max_word_len], dtype=DTYPE)
    word_lengths_storage = np.zeros([total_num_words], dtype=DTYPE)
    cdef int[::1] batch_indices = batch_indices_storage
    cdef int[::1] sample_indices = sample_indices_storage
    cdef int[:, ::1] word_positions = word_positions_storage
    cdef int[::1] word_lengths = word_lengths_storage

    cdef Py_ssize_t batch_size = samples.shape[0]
    cdef Py_ssize_t num_samples = samples.shape[1]

    cdef Py_ssize_t offset
    cdef Py_ssize_t length
    cdef int last_value
    cdef int next_value
    cdef int value
    cdef int sample_length

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef bint start
    cdef bint add
    cdef bint wrap_up

    for i in prange(batch_size, nogil=True):
        for j in range(num_samples):
            offset = offsets[i, j]
            length = 0
            last_value = N
            next_value = samples[i, j, 0]
            sample_length = sample_lengths[i, j]
            for k in range(sample_length):
                value = next_value
                if k < sample_length - 1:
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
                    length = length + 1
                if wrap_up:
                    word_lengths[offset] = length
                    length = 0
                    offset = offset + 1

                last_value = value
    return batch_indices_storage, sample_indices_storage, word_positions_storage, word_lengths_storage


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_words_v8(const int[:, :, ::1] samples, const int[:, ::1] sample_lengths, int num_threads=1):
    openmp.omp_set_num_threads(num_threads)
    # First pass to calculate total number of words and max length of words.
    word_counts, max_lengths, offsets, is_unique = first_pass(samples, sample_lengths, num_threads)
    cdef long total_num_words = word_counts.sum()
    cdef int max_word_len = max_lengths.max()

    # Now we can extract words.
    batch_indices, sample_indices, word_positions, word_lengths = second_pass(
        samples, sample_lengths, offsets, total_num_words, max_word_len)

    return batch_indices, sample_indices, word_positions, word_lengths, is_unique
