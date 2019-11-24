# distutils: language = c++
# cython: c_string_type = unicode
# cython: c_string_encoding = default

import cython
from cython.parallel import prange

import numpy as np
cimport numpy as np
cimport openmp

from libcpp.set cimport set
from libcpp.vector cimport vector
from libcpp.string cimport string


DTYPE = np.intc

cdef check_in_vocab_kernel(
        const int[::1] batch_indices,
        const int[:, ::1] word_positions,
        const int[::1] word_lengths,
        const vector[vector[string]] segments,
        const set[string] vocab):

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef int bi
    cdef int si
    cdef int wl
    cdef int start
    cdef int end
    cdef string word

    cdef Py_ssize_t num_words = batch_indices.shape[0]

    in_vocab_storage = np.zeros([num_words], dtype=DTYPE)
    cdef int[::1] in_vocab = in_vocab_storage

    for i in range(num_words):
        bi = batch_indices[i]
        wl = word_lengths[i]
        start = word_positions[i, 0]
        end = start + wl

        word = ''
        for j in range(start, end):
            word.append(segments.at(bi).at(j))

        in_vocab[i] = vocab.count(word)

    return in_vocab_storage

@cython.boundscheck(False)
@cython.wraparound(False)
def check_in_vocab(const int[::1] batch_indices, const int[:, ::1] word_positions, const int[::1] word_lengths, segments, vocab, int num_threads=1):
    openmp.omp_set_num_threads(num_threads)
    cdef vector[vector[string]] cpp_segments = segments
    cdef set[string] cpp_vocab = vocab
    in_vocab = check_in_vocab_kernel(batch_indices, word_positions, word_lengths, cpp_segments, cpp_vocab)
    return in_vocab