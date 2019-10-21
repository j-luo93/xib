import numpy as np
cimport numpy as np


from xib.extract_words import where, B, I, O


def first_pass(samples):
    batch_size, num_samples, max_len = samples.shape
    word_counts = np.zeros([batch_size, num_samples], dtype=np.uint32)
    max_lengths = np.zeros([batch_size, num_samples], dtype=np.uint32)
    offsets = np.zeros([batch_size * num_samples], dtype=np.uint32)

    for i in range(batch_size):
        for j in range(num_samples):
            length = 0
            max_length = 0
            count = 0
            last_value = None
            next_value = samples[i, j, 0]
            for k in range(max_len):
                value = next_value
                if k < max_len - 1:
                    next_value = samples[i, j, k + 1]
                else:
                    next_value = None

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


def second_pass(samples, offsets, total_num_words, max_word_len):
    batch_indices = np.zeros([total_num_words], dtype=np.uint32)
    sample_indices = np.zeros([total_num_words], dtype=np.uint32)
    word_positions = np.zeros([total_num_words, max_word_len], dtype=np.uint32)
    word_lengths = np.zeros([total_num_words], dtype=np.uint32)

    batch_size, num_samples, max_len = samples.shape
    for i in range(batch_size):
        for j in range(num_samples):
            offset = offsets[i, j]
            length = 0
            last_value = None
            next_value = samples[i, j, 0]
            for k in range(max_len):
                value = next_value
                if k < max_len - 1:
                    next_value = samples[i, j, k + 1]
                else:
                    next_value = None
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


def extract_words_v1(samples):
    # First pass to calculate total number of words and max length of words.
    word_counts, max_lengths, offsets = first_pass(samples)
    total_num_words = word_counts.sum()
    max_word_len = max_lengths.max()

    # Now we can extract words.
    batch_indices, sample_indices, word_positions, word_lengths = second_pass(
        samples, offsets, total_num_words, max_word_len)

    return batch_indices, sample_indices, word_positions, word_lengths

