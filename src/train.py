"""
Trains the GPT on Tiny Shakespeare with mini-batch Adam, then saves the weights.
Run from the project root:  python src/train.py
"""

import json
import time
from pathlib import Path

import numpy as np

from data import get_batch, load_data
from gpt import GPT, CrossEntropyWithSoftmax
from optim import Adam

# ---------------------------------------------------------------- hyperparameters
BATCH_SIZE = 16          # B: sequences per batch
CONTEXT_LENGTH = 64      # T: tokens per sequence
D_MODEL = 64             # model dimension
NUM_HEADS = 4            # attention heads (d_k = 64 / 4 = 16)
NUM_LAYERS = 4           # transformer blocks
LEARNING_RATE = 1e-3     # Adam's eta
MAX_STEPS = 5000         # training steps
EVAL_INTERVAL = 250      # measure and print the loss every this many steps
EVAL_ITERS = 20          # batches averaged per loss estimate
SEED = 0                 # for reproducibility

# checkpoints/ in the project root (src/train.py -> project root -> checkpoints/)
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def estimate_loss(model, loss_fn, ids, rng):
    """
    Average loss over EVAL_ITERS random batches from ids (train or val).
    Forward only: no backward, no update, so the model is not changed.
    """
    losses = []
    for _ in range(EVAL_ITERS):
        X, Y = get_batch(ids, BATCH_SIZE, CONTEXT_LENGTH, rng)
        losses.append(loss_fn.forward(model.forward(X), Y))
    return np.mean(losses)


def save_checkpoint(model, tokenizer, config):
    """
    Saves everything generate.py needs to rebuild the trained model:
      model.npz:   every parameter array, keyed by name (e.g. 'blocks.0.attn.W_q.W')
      config.json: the model's settings + the tokenizer's characters
    """
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # name -> parameter array (the gradients aren't needed)
    weights = {name: param for name, param, _ in model.params_and_grads()}
    np.savez(CHECKPOINT_DIR / "model.npz", **weights)

    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump({**config, "chars": tokenizer.chars}, f, indent=2)

    print(f"saved {len(weights)} parameter arrays to {CHECKPOINT_DIR}")


def main():
    # Reproducibility: the weight initialisation uses np.random; batches use their own generators.
    # Separate generators for training and evaluation, so changing how often we evaluate
    # doesn't change which training batches are drawn.
    np.random.seed(SEED)
    train_rng = np.random.default_rng(SEED)
    eval_rng = np.random.default_rng(SEED + 1)

    # Data and tokenizer
    tokenizer, train_ids, val_ids = load_data()
    print(f"vocab_size {tokenizer.vocab_size} | train tokens {len(train_ids):,} | val tokens {len(val_ids):,}")

    # Model, loss and optimiser
    config = dict(
        vocab_size=tokenizer.vocab_size,   # from the tokenizer, so it always matches the data
        context_length=CONTEXT_LENGTH,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    )
    model = GPT(**config)
    loss_fn = CrossEntropyWithSoftmax()
    optimiser = Adam(model, lr=LEARNING_RATE)

    start = time.time()
    for step in range(MAX_STEPS):
        # Every EVAL_INTERVAL steps (including step 0): averaged train and val loss
        if step % EVAL_INTERVAL == 0:
            train_loss = estimate_loss(model, loss_fn, train_ids, eval_rng)
            val_loss = estimate_loss(model, loss_fn, val_ids, eval_rng)
            elapsed = time.time() - start
            print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:7.1f}s")

        # One training step: mini-batch -> forward -> backward -> update
        X, Y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, train_rng)   # (B, T) each
        loss_fn.forward(model.forward(X), Y)                                 # loss on this batch
        model.backward(loss_fn.backward())                                   # grads of this batch's loss
        optimiser.step()                                                     # update every parameter

    # Final measurement
    train_loss = estimate_loss(model, loss_fn, train_ids, eval_rng)
    val_loss = estimate_loss(model, loss_fn, val_ids, eval_rng)
    elapsed = time.time() - start
    print(f"step {MAX_STEPS:5d} | train {train_loss:.4f} | val {val_loss:.4f} | {elapsed:7.1f}s")

    num_params = sum(param.size for _, param, _ in model.params_and_grads())
    print(f"trained {num_params:,} parameters")

    save_checkpoint(model, tokenizer, config)


if __name__ == "__main__":
    main()