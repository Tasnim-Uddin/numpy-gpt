# Documentation

How the NumPy GPT works, from the architecture to the hand-derived gradients.

| document | covers |
|---|---|
| [forward-pass.md](forward-pass.md) | the architecture and data flow, with diagrams and the shape at every step |
| [backward-pass.md](backward-pass.md) | the backward pass of every layer, derived with index notation and the chain rule |
| [numerical-stability.md](numerical-stability.md) | how the code avoids overflow, `log(0)` and division by zero |
| [gradient-checking.md](gradient-checking.md) | verifying every gradient with central differences, and why the method works |
| [optimisers.md](optimisers.md) | mini-batch SGD and Adam: what they do and why |
| [data-and-generation.md](data-and-generation.md) | tokenizer, batches, the training loop, checkpoints, and text generation |

**Suggested reading order:** forward pass → backward pass → numerical stability → gradient checking → optimisers → data and generation.

## Notation used throughout

| symbol | meaning |
|---|---|
| $B$, $T$, $N = BT$ | batch size, sequence length, total positions |
| $d$ (`d_model`) | model dimension |
| $h$, $d_k = d / h$ | number of heads, per-head dimension |
| $V$ | vocabulary size |
| $\bar{A}$ (`dA`) | the adjoint of $A$: $\partial L / \partial A$, same shape as $A$ |
| $\odot$ | element-wise (Hadamard) product |