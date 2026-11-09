# Data, training and generation

Everything around the model: turning text into batches, the training loop, saving the model, and generating text.

## Contents

1. [Tokenization](#1-tokenization)
2. [Train and validation split](#2-train-and-validation-split)
3. [Mini-batches](#3-mini-batches)
4. [The training loop](#4-the-training-loop)
5. [Checkpoints](#5-checkpoints)
6. [Generation](#6-generation)
7. [Temperature](#7-temperature)

---

## 1. Tokenization

The model works with integer **token IDs**, not text. A **character-level** tokenizer makes every distinct character one token:

1. **Vocabulary:** the distinct characters in the text, **sorted**, so every run assigns the same IDs (a Python `set` has no fixed order). Tiny Shakespeare has $V = 65$.
2. **encode:** text → IDs, one per character.
3. **decode:** IDs → text.

`decode(encode(s)) == s` for any text made of known characters.

| strategy | example: "unbelievable" | vocabulary size |
|---|---|---|
| character (used here) | 12 tokens: `u n b e l i e v a b l e` | about 65 |
| word | 1 token (if it's in the vocabulary) | hundreds of thousands |
| subword (BPE, used by GPT-2 onwards) | e.g. `un believ able` | about 50,000 to 200,000 |

Character-level keeps the vocabulary tiny and never meets an unknown token, at the cost of long sequences and having to learn spelling.

---

## 2. Train and validation split

The whole text is encoded into one long array of IDs, then cut once:

- the first 90% is **training** data (`train_ids`), which the model learns from
- the last 10% is **validation** data (`val_ids`), which is never used for gradients

The cut is contiguous, because every example is a run of consecutive characters. The validation loss measures how well the model handles **unseen** text.

---

## 3. Mini-batches

`get_batch` picks $B$ random starting positions and cuts a chunk of $T$ consecutive tokens from each. The targets are the same chunks shifted one position later:

$$
X[b, t] = \text{ids}[s_b + t],
\qquad
Y[b, t] = \text{ids}[s_b + t + 1]
$$

so $Y[b, t]$ is the token that follows $X[b, t]$. With $T = 4$ and a start at 1:

```
ids:  [ 5, 12,  9,  3, 41,  7, ... ]
X   = ids[1:5] = [12,  9,  3, 41]
Y   = ids[2:6] = [ 9,  3, 41,  7]
```

One chunk gives $T$ predictions, and one batch gives $N = BT$. Starts are drawn from `0` up to `len(ids) - T - 1`, leaving room for the extra token at the end of `Y`.

**The index grid.** `starts[:, None] + np.arange(T)` broadcasts a $(B, 1)$ column of starts against a row of offsets into a $(B, T)$ grid of positions. `ids[grid]` looks them all up at once, and `ids[grid + 1]` gives the targets.

Chunks often start mid-word. That's intended: generation also sees a sliding window that starts anywhere.

---

## 4. The training loop

One training **step** is one forward pass, one backward pass and one optimiser update:

```python
X, Y = get_batch(train_ids, BATCH_SIZE, CONTEXT_LENGTH, train_rng)
loss_fn.forward(model.forward(X), Y)       # layers save their activations
model.backward(loss_fn.backward())         # layers store their gradients
optimiser.step()                           # every parameter updated in place
```

The values pass through the objects: `forward` stores activations inside each layer, `backward` reads them and stores gradients inside each layer, and `step` reads the gradients through `params_and_grads()`.

**Measuring progress.** A single batch's loss is noisy, so every `EVAL_INTERVAL` steps, `estimate_loss` averages the loss over `EVAL_ITERS` random batches (forward only, so the model isn't changed). It runs on both sets:

| pattern | meaning |
|---|---|
| training and validation both falling, close together | learning general patterns |
| training falling, validation flat or rising | **overfitting** |
| both high | **underfitting** |

**Perplexity.** $e^{L}$ is the effective number of equally likely choices the model is deciding between. At the start, $L \approx \ln 65$, so the perplexity is 65. With the default settings, the validation loss reaches about 1.75, a perplexity of about 5.7.

**Seeds.** `np.random.seed` fixes the weight initialisation, and separate generators fix the training and evaluation batches, so every run is reproducible.

---

## 5. Checkpoints

After training, `save_checkpoint` writes two files to `checkpoints/`:

| file | contents |
|---|---|
| `model.npz` | every parameter array, keyed by name, via `np.savez` |
| `config.json` | the model's settings (`vocab_size`, `context_length`, `d_model`, `num_heads`, `num_layers`) and the vocabulary characters |

Gradients aren't saved: they only exist to compute the next update, and they're specific to the last batch. Generation only needs the weights. (Resuming training exactly would also need Adam's $\mathbf{m}$, $\mathbf{v}$ and $t$.)

`generate.py` rebuilds the model with the same settings, then copies each saved array **into** the model's array with `param[...] = weights[name]`.

---

## 6. Generation

The model predicts one token at a time, so text is generated **autoregressively**:

1. Crop the text so far to the last `context_length` tokens (the positional embeddings go no further).
2. Run the forward pass, and take the logits at the **last** position: they predict the next token.
3. Divide by the temperature and apply softmax.
4. **Sample** the next token from those probabilities.
5. Append it, and repeat.

**Why sample instead of always taking the most likely token?** Always taking the top token (*greedy decoding*) tends to loop and repeat itself, and never produces the less likely but valid continuations that real text contains. Sampling draws from the model's learned distribution over the next token.

---

## 7. Temperature

The logits are divided by a temperature $T > 0$ before softmax:

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

The ratio of any two probabilities depends only on the gap between their logits, scaled by $\frac{1}{T}$:

$$
\frac{p_i}{p_j} = \exp\!\left(\frac{z_i - z_j}{T}\right)
$$

A low temperature widens every gap, making the distribution sharper. A high temperature shrinks the gaps, making it flatter. For logits $[2, 1, 0]$:

| $T$ | probabilities | text |
|---|---|---|
| 0.5 | $[0.87, 0.12, 0.02]$ | safer, more repetitive |
| 1 | $[0.67, 0.24, 0.09]$ | the model's learned distribution |
| 2 | $[0.51, 0.31, 0.19]$ | more varied, more mistakes |

**The limits.** As $T \to 0$, all the probability moves to the top token (greedy decoding). As $T \to \infty$, every $z_i / T \to 0$, so every token becomes equally likely ($p_i \to 1 / V$).

The name comes from the Boltzmann distribution in physics, $p_i \propto \exp(-E_i / kT)$, with the logit playing the role of negative energy.