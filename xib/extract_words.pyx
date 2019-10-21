import numpy as np
cimport numpy as np

B, I, O = 0, 1, 2

def extract_words(samples):
    """
    `samples` is of size ('batch', 'sample', 'length'), and we are expecting four outputs:
    1. `batch_indices`: size ('batch_word', ), which stores the batch indices for all extract words.
    2. `sample_indices`: size ('batch_word', ), similar to `batch_indices` but stores sample indices.
    3. `word_positions`: size ('batch_word', 'position'), which stores the original position in `samples` for each extracted word.
    4. `word_lengths`: size ('batch_word', ), similar to `batch_indices` but stores word lengths.
    """
    return extract_words_v1(samples)
