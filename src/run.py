"""Load Shakespeare, run the GPT forward pass, compute the loss, sample some text."""

import numpy as np

from data import get_batch, load_data
from gpt import GPT, CrossEntropyWithSoftmax, Softmax

# Hyperparameters (small, so NumPy on a CPU is quick)
BATCH_SIZE = 8
CONTEXT_LENGTH = 64
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2


def generate(model, token_ids, max_new_tokens, rng):
    """Autoregressive sampling: predict one token, append it, repeat."""
    softmax = Softmax()
    for _ in range(max_new_tokens):
        context = token_ids[:, -model.context_length:]        # crop to the context window
        logits = model.forward(context)[:, -1, :]             # last position only -> (B, vocab_size)
        probs = softmax.forward(logits)
        next_ids = np.array([[rng.choice(len(p), p=p)] for p in probs])   # sample from probs
        token_ids = np.concatenate([token_ids, next_ids], axis=1)
    return token_ids


def main():
    np.random.seed(0)                    # model weights use np.random -> reproducible
    rng = np.random.default_rng(0)       # batches and sampling

    tokenizer, train_ids, val_ids = load_data()
    print(f"vocab_size: {tokenizer.vocab_size}, train tokens: {len(train_ids):,}")

    model = GPT(
        vocab_size=tokenizer.vocab_size,
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )

    # One forward pass on a real batch
    X, Y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, rng)

    # Show the batch: one row per sequence. repr (!r) makes newlines visible as \n
    print(f"\nInput batch X: {BATCH_SIZE} sequences x {CONTEXT_LENGTH} characters")
    for i, row in enumerate(X):
        print(f"  [{i}] {tokenizer.decode(row)!r}")

    # Targets are the same text shifted one character later
    print("\nSequence 0 inputs vs targets (each target is the next character):")
    print(f"  X[0]: {tokenizer.decode(X[0])!r}")
    print(f"  Y[0]: {tokenizer.decode(Y[0])!r}")

    logits = model.forward(X)
    loss = CrossEntropyWithSoftmax().forward(logits, Y)

    print(f"\nX: {X.shape}  logits: {logits.shape}")
    print(f"loss: {loss:.4f}  (random guessing = ln(vocab_size) = {np.log(tokenizer.vocab_size):.4f})")

    # Generate from a prompt (untrained, so expect gibberish)
    prompt = "\n"
    start = np.array([tokenizer.encode(prompt)])     # (1, len(prompt))
    out = generate(model, start, max_new_tokens=100, rng=rng)

    print(f"\nPrompt: {prompt!r}")
    print(f"Generated {out.shape[1] - start.shape[1]} new characters:")
    print(tokenizer.decode(out[0]))


if __name__ == "__main__":
    main()