"""
Gradient check: compares every backward pass against central-difference estimates.
    dL/dtheta_i  ~=  ( L(theta_i + h) - L(theta_i - h) ) / (2h)
Run from the project root:  python src/test_gradients.py
"""

import numpy as np

from gpt import GPT, CrossEntropyWithSoftmax


def compute_loss(model, loss_fn, X, Y):
    """Forward pass only: token IDs -> scalar loss L(theta)."""
    return loss_fn.forward(model.forward(X), Y)


def relative_error(analytic, numerical, eps=1e-12):
    """|a - n| / max(|a|, |n|): works for gradients of any size; eps avoids 0 / 0."""
    return abs(analytic - numerical) / max(abs(analytic), abs(numerical), eps)


def check_gradients(checks_per_param=None, h=1e-5, seed=0):
    """
    Builds a tiny GPT, computes analytic gradients with backward, then compares
    them with the central difference, parameter by parameter.
    checks_per_param: None = check every entry of every parameter;
                      an integer = check that many random entries per parameter
    """
    np.random.seed(seed)               # the model's weight initialisation uses np.random
    rng = np.random.default_rng(seed)  # random data and random entries to check

    # Tiny model and batch: every layer type, but each forward pass is fast
    V, B, T = 11, 2, 5
    model = GPT(vocab_size=V, context_length=T, d_model=8, num_heads=2, num_layers=2)
    loss_fn = CrossEntropyWithSoftmax()

    # Fixed random batch, built like get_batch: T + 1 tokens per sequence,
    # X = the first T, Y = the same tokens shifted one later (the next-token targets)
    sequence = rng.integers(0, V, size=(B, T + 1))   # (B, T + 1)
    X = sequence[:, :-1]                             # (B, T) inputs
    Y = sequence[:, 1:]                              # (B, T) targets

    # Analytic gradients a_i: one forward pass, then one backward pass
    compute_loss(model, loss_fn, X, Y)
    model.backward(loss_fn.backward())

    all_ok = True
    for name, param, grad in model.params_and_grads():
        # Which entries theta_i of this parameter to check, as index tuples like (i, j)
        if checks_per_param is None:
            indices = list(np.ndindex(param.shape))                       # every entry
        else:
            indices = [np.unravel_index(rng.integers(param.size), param.shape)
                       for _ in range(checks_per_param)]                  # random entries

        worst_rel, worst_abs = 0.0, 0.0
        for idx in indices:
            original = param[idx]

            # L+ and L-: nudge this one entry up, then down, then restore it
            param[idx] = original + h
            loss_plus = compute_loss(model, loss_fn, X, Y)
            param[idx] = original - h
            loss_minus = compute_loss(model, loss_fn, X, Y)
            param[idx] = original

            # Central difference n_i, and the analytic a_i from backward
            numerical = (loss_plus - loss_minus) / (2 * h)
            analytic = grad[idx]

            worst_rel = max(worst_rel, relative_error(analytic, numerical))
            worst_abs = max(worst_abs, abs(analytic - numerical))

        # Pass if relatively close, or if both are so tiny that only rounding noise differs
        ok = worst_rel < 1e-5 or worst_abs < 1e-9
        all_ok = all_ok and ok
        print(f"{name:45s} rel {worst_rel:.1e}  abs {worst_abs:.1e}  {'ok' if ok else 'FAIL'}")

    print("\nALL GRADIENTS OK" if all_ok else "\nSOME GRADIENTS FAILED: check the FAIL lines")


if __name__ == "__main__":
    check_gradients()
