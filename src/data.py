"""
Tiny Shakespeare data pipeline: character-level tokenizer, train/validation split, batches.
"""

from pathlib import Path

import numpy as np

# data/input.txt, found relative to this file (src/data.py -> project root -> data/)
DATA_PATH = Path(__file__).parent.parent / "data" / "input.txt"


class CharTokenizer:
    """
    Character-level tokenizer: every distinct character in the text is one token.
    IDs are 0 .. V-1, assigned in sorted order so they are the same on every run.
    """

    def __init__(self, text):
        # Vocabulary: every distinct character, sorted (a set has no fixed order)
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)   # V

        # Lookup tables in both directions
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}   # encode
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}   # decode

    def encode(self, text):
        """
        text:    string
        returns: list of token IDs, one per character
        """
        return [self.char_to_id[char] for char in text]

    def decode(self, ids):
        """
        ids:     sequence of token IDs (list or 1D NumPy array)
        returns: the string they represent
        """
        return "".join(self.id_to_char[int(i)] for i in ids)


def load_data(train_fraction=0.9):
    """
    Reads Tiny Shakespeare, builds the tokenizer, and splits the encoded text.
    train_fraction: share of the text used for training (the rest is validation)
    returns: tokenizer, train_ids (N_train,), val_ids (N_val,)  -- 1D int arrays
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download it with:\n"
            "  mkdir -p data && curl -o data/input.txt "
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        )

    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)

    # The whole text as one long array of token IDs
    all_ids = np.array(tokenizer.encode(text), dtype=np.int64)   # (total characters,)

    # One contiguous cut: first part for training, last part held out for validation
    split = int(train_fraction * len(all_ids))
    train_ids = all_ids[:split]
    val_ids = all_ids[split:]

    return tokenizer, train_ids, val_ids


def get_batch(ids, batch_size, context_length, rng):
    """
    Random chunks of the data as inputs; the same chunks shifted by one as targets.
    ids:            (N_ids,) 1D token IDs (train_ids or val_ids)
    batch_size:     B, number of chunks
    context_length: T, tokens per chunk
    rng:            np.random.Generator, for reproducible randomness
    returns:        X (B, T) inputs, Y (B, T) targets, with Y[b, t] = the token after X[b, t]
    """
    # B random start positions; the last allowed start leaves room for the +1 shift in Y
    starts = rng.integers(0, len(ids) - context_length, size=batch_size)   # (B,)

    # Index grid: row b is starts[b], starts[b]+1, ..., starts[b]+T-1
    offsets = np.arange(context_length)                                   # (T,)
    index_grid = starts[:, None] + offsets                                # (B, T) by broadcasting

    X = ids[index_grid]          # (B, T) inputs
    Y = ids[index_grid + 1]      # (B, T) targets: every index shifted one later
    return X, Y
