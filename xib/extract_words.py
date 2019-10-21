import numpy as np

B, I, O = 0, 1, 2


def where(value, last_value, next_value):
    """
    Return three values:
    1. start: whether to start a new word with this position.
    2. add: whether to add this position. Note that start always implies add.
    3. wrap_up: whether to wrap up this position.

    last_value  value   next_value  start   add     wrap_up
    None        B       None        True    True    True
    None        B       B           True    True    True
    None        B       I           True    True    False
    None        B       O           True    True    True
    None        I       None        True    True    True
    None        I       B           True    True    True
    None        I       I           True    True    False
    None        I       O           True    True    True
    None        O       None        False   False   False
    None        O       B           False   False   False
    None        O       I           False   False   False
    None        O       O           False   False   False
    -------------------------------------------------------
    B           B       None        True    True    True
    B           B       B           True    True    True
    B           B       I           True    True    False
    B           B       O           True    True    True
    B           I       None        False   True    True
    B           I       B           False   True    True
    B           I       I           False   True    False
    B           I       O           False   True    True
    B           O       None        False   False   False
    B           O       B           False   False   False
    B           O       I           False   False   False
    B           O       O           False   False   False
    -------------------------------------------------------
    I           B       None        True    True    True
    I           B       B           True    True    True
    I           B       I           True    True    False
    I           B       O           True    True    True
    I           I       None        False   True    True
    I           I       B           False   True    True
    I           I       I           False   True    False
    I           I       O           False   True    True
    I           O       None        False   False   False
    I           O       B           False   False   False
    I           O       I           False   False   False
    I           O       O           False   False   False
    -------------------------------------------------------
    O           B       None        True    True    True
    O           B       B           True    True    True
    O           B       I           True    True    False
    O           B       O           True    True    True
    O           I       None        True    True    True
    O           I       B           True    True    True
    O           I       I           True    True    False
    O           I       O           True    True    True
    O           O       None        False   False   False
    O           O       B           False   False   False
    O           O       I           False   False   False
    O           O       O           False   False   False

    In short:
    Start a word iff:
        (value == B) or (value == I and (last_value is None or last_value = O))
    Add a word iff:
        (value == B) or (value == I)
    Wrap up a word iff:
        add and (next_value != I)

    Also note that O is besically equivalent to None.
    """
    start = (value == B) or (value == I and (last_value is None or last_value == O))
    add = (value == B) or (value == I)
    wrap_up = add and (next_value != I)
    return start, add, wrap_up


def first_pass(samples):
    batch_size, num_samples, max_len = samples.shape
    word_counts = np.zeros([batch_size, num_samples], dtype=np.uint32)
    max_lengths = np.zeros([batch_size, num_samples], dtype=np.uint32)
    offsets = np.zeros([batch_size, num_samples], dtype=np.uint32)

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
    accum_counts = np.cumsum(word_counts.reshape(-1)).reshape([batch_size, num_samples])
    offsets[:, 1:] = accum_counts[:, :-1]
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


def extract_words_py(samples):
    # First pass to calculate total number of words and max length of words.
    word_counts, max_lengths, offsets = first_pass(samples)
    total_num_words = word_counts.sum()
    max_word_len = max_lengths.max()

    # Now we can extract words.
    batch_indices, sample_indices, word_positions, word_lengths = second_pass(
        samples, offsets, total_num_words, max_word_len)

    return batch_indices, sample_indices, word_positions, word_lengths
