"""Tiny Shakespeare: character-level tokenizer, train/val split, random batches."""

from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent.parent / "data" / "input.txt"


class CharTokenizer:
    """Character-level tokenizer: each distinct character is one token."""

    def __init__(self, text):
        self.chars = sorted(set(text))                                   # the vocabulary
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}     # "stoi"
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}     # "itos"

    def encode(self, text):
        """string -> list of token IDs"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        """token IDs -> string"""
        return "".join(self.id_to_char[int(i)] for i in ids)


def load_data(train_fraction=0.9):
    """Returns (tokenizer, train_ids, val_ids). IDs are 1D int arrays."""
    text = DATA_PATH.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    data = np.array(tokenizer.encode(text), dtype=np.int64)

    # First 90% for training, last 10% held out for validation
    split = int(train_fraction * len(data))
    return tokenizer, data[:split], data[split:]


def get_batch(data, batch_size, context_length, rng):
    """
    Random chunks of the data as inputs, the same chunks shifted by one as targets.
    returns: X (B, T), Y (B, T) with T = context_length
    """
    # Random start positions; leave room for the +1 shift of the targets
    starts = rng.integers(0, len(data) - context_length, size=batch_size)

    X = np.stack([data[s : s + context_length] for s in starts])            # inputs
    Y = np.stack([data[s + 1 : s + context_length + 1] for s in starts])    # next tokens
    return X, Y