"""
Generates text from the trained GPT, one character at a time (autoregressive sampling).
Run from the project root, after training:  python src/generate.py
"""

import json
from pathlib import Path

import numpy as np

from data import CharTokenizer
from gpt import GPT, Softmax

# ---------------------------------------------------------------- settings
PROMPT = "ROMEO:\n"      # text to start from (every character must be in the vocabulary)
MAX_NEW_TOKENS = 500     # how many characters to generate
TEMPERATURE = 0.8        # < 1: safer, more predictable;  > 1: more varied, more mistakes
SEED = 0                 # same seed -> same generated text

CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def load_model():
    """
    Rebuilds the trained model and tokenizer from checkpoints/:
      config.json -> model settings + vocabulary,  model.npz -> trained weights
    returns: model, tokenizer
    """
    if not (CHECKPOINT_DIR / "model.npz").exists():
        raise FileNotFoundError(f"No checkpoint in {CHECKPOINT_DIR}. Run python src/train.py first.")

    with open(CHECKPOINT_DIR / "config.json") as f:
        config = json.load(f)

    # The vocabulary rebuilds the tokenizer: sorted(set(...)) gives back the same characters in the same order
    chars = config.pop("chars")
    tokenizer = CharTokenizer("".join(chars))

    # Same settings as training, so every parameter has the same shape as the saved one
    model = GPT(**config)

    # Copy each saved array INTO the model's own array (in place, so the model changes)
    weights = np.load(CHECKPOINT_DIR / "model.npz")
    for name, param, _ in model.params_and_grads():
        param[...] = weights[name]

    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens, temperature, rng):
    """
    Autoregressive sampling: predict the next character, sample it, append it, repeat.
    Prints each character as it is generated, and returns the full text.
    """
    softmax = Softmax()
    ids = np.array([tokenizer.encode(prompt)])            # (1, T): one sequence
    print(prompt, end="", flush=True)

    for _ in range(max_new_tokens):
        # Only the last context_length tokens fit through the model (positional embeddings stop there)
        context = ids[:, -model.context_length:]          # (1, T_context)

        # Forward pass; only the LAST position predicts the next token
        logits = model.forward(context)                   # (1, T_context, V)
        last_logits = logits[:, -1, :] / temperature      # (1, V): temperature rescales the logits

        # Probabilities over the next character, then sample one in proportion to them
        probs = softmax.forward(last_logits)[0]           # (V,)
        next_id = rng.choice(len(probs), p=probs)

        # Append and continue from the longer sequence
        ids = np.append(ids, [[next_id]], axis=1)         # (1, T + 1)
        print(tokenizer.decode([next_id]), end="", flush=True)

    print()
    return tokenizer.decode(ids[0])


def main():
    rng = np.random.default_rng(SEED)
    model, tokenizer = load_model()
    generate(model, tokenizer, PROMPT, MAX_NEW_TOKENS, TEMPERATURE, rng)


if __name__ == "__main__":
    main()
