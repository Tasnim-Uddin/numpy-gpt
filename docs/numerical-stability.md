# Numerical stability

Computers store numbers with limited range and precision. This document covers each place where the code avoids overflow, `log(0)` or division by zero, and why each fix doesn't change the maths.

## Contents

1. [The limits of floating point](#1-the-limits-of-floating-point)
2. [Softmax: subtract the max](#2-softmax-subtract-the-max)
3. [Cross-entropy: compute log-probabilities directly](#3-cross-entropy-compute-log-probabilities-directly)
4. [The causal mask: minus infinity](#4-the-causal-mask-minus-infinity)
5. [Layer normalisation: epsilon](#5-layer-normalisation-epsilon)
6. [Attention: scaling by the square root of d_k](#6-attention-scaling-by-the-square-root-of-d_k)
7. [Adam: epsilon](#7-adam-epsilon)
8. [Summary](#8-summary)

---

## 1. The limits of floating point

NumPy uses 64-bit floats (`float64`) by default:

| limit | value | consequence |
|---|---|---|
| largest number | about $1.8 \times 10^{308}$ | $e^x$ **overflows** to `inf` for $x$ above about $709.8$ |
| smallest positive number | about $5 \times 10^{-324}$ | $e^x$ **underflows** to `0.0` for $x$ below about $-745$ |
| relative precision ($\varepsilon$) | about $2.2 \times 10^{-16}$ | every result has a tiny rounding error |

Once an `inf` or `nan` appears, it spreads through every later calculation (`inf / inf = nan`, `nan * 0 = nan`), so a single overflow can ruin a whole training run.

---

## 2. Softmax: subtract the max

### The problem

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

- **Large scores overflow.** For $\mathbf{z} = [1000, 999]$, both exponentials are `inf`, and `inf / inf = nan`.
- **Very negative scores underflow.** For $\mathbf{z} = [-1000, -1001]$, both exponentials are `0`, and `0 / 0 = nan`.

### The fix

Softmax doesn't change when the same constant $c$ is subtracted from every score, because $e^{-c}$ cancels:

$$
\frac{\exp(z_i - c)}{\sum_j \exp(z_j - c)}
= \frac{e^{-c}\exp(z_i)}{e^{-c}\sum_j \exp(z_j)}
= \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

Choose $c = m = \max_j z_j$. Every shifted score is then $\le 0$, so:

- the largest exponential is exactly $e^0 = 1$, so **nothing can overflow**
- the denominator is at least 1, so **it can never be 0**

A shifted score can still underflow to 0, but only when it's hundreds below the maximum, where its true probability really is negligible.

| scores | naive | shifted scores | stable result |
|---|---|---|---|
| $[1000, 999]$ | `nan` | $[0, -1]$ | $[0.731, 0.269]$ |
| $[-1000, -1001]$ | `nan` | $[0, -1]$ | $[0.731, 0.269]$ |

**In code:** `shifted = X - X.max(axis=-1, keepdims=True)` in `Softmax.forward`.

---

## 3. Cross-entropy: compute log-probabilities directly

### The problem

Even with a stable softmax, a very wrong prediction can give the correct token a probability so small that it rounds to exactly 0, and then $-\log 0 = \infty$:

| logits, correct token 0 | softmax | $-\log p_0$ |
|---|---|---|
| $[0, 1000]$ | $[0.0, 1.0]$ | `inf` |

### The fix: log-softmax

Take the log of the shifted softmax formula. The division becomes a subtraction:

$$
\log p_i = (z_i - m) - \log\sum_{j}\exp(z_j - m)
$$

Nothing is divided, and the sum inside the log contains $e^0 = 1$, so it can never be 0. For the example above, $\log p_0 = (0 - 1000) - \log(1 + e^{-1000}) = -1000$, so the loss is $1000$: very large, as it should be for a confident wrong prediction, but finite.

The term $\log\sum_j \exp(z_j)$ is called **log-sum-exp**, and the shifted version is the standard stable way to compute it.

### The gradient is unaffected

The shift $m$ depends on $\mathbf{z}$, but its effect cancels exactly in the derivative. With $\ell = -(z_y - m) + \log\sum_j \exp(z_j - m)$:

$$
\frac{\partial \ell}{\partial z_k}
= -\delta_{yk} + \frac{\partial m}{\partial z_k} + p_k - \frac{\partial m}{\partial z_k}\sum_j p_j
= p_k - \delta_{yk}
$$

since $\sum_j p_j = 1$. So the backward pass uses $\frac{1}{N}(P - Y)$ with no reference to the shift.

**In code:** `log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))` in `CrossEntropyWithSoftmax.forward`.

---

## 4. The causal mask: minus infinity

Future positions are blocked by setting their scores to $-\infty$ before softmax:

- $e^{-\infty} = 0$, so masked positions get **exactly zero** weight
- the diagonal (a token attending to itself) is never masked, so every row has at least one finite score, and the row's maximum is finite. The max-shift therefore never computes $-\infty - (-\infty)$, which would be `nan`
- in the backward pass, the softmax gradient is $\mathbf{p} \odot (\ldots)$, which is 0 wherever $p = 0$, so masked positions automatically get zero gradient

In lower-precision formats (16-bit floats), $-\infty$ can be risky, and implementations often use a large negative number such as $-10^9$ instead.

**In code:** `S = np.where(self.mask[:T, :T], -np.inf, S)` in `MultiHeadAttention.forward`.

---

## 5. Layer normalisation: epsilon

LayerNorm divides by the standard deviation of each vector. If every entry is equal, the variance is 0, and the division would be $0 / 0$. A small $\epsilon$ prevents this:

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \epsilon = 10^{-5}
$$

$\epsilon$ is a fixed constant, not a learned parameter. It only matters when the variance is close to 0.

**In code:** `self.std = np.sqrt(var + self.eps)` in `LayerNorm.forward`.

---

## 6. Attention: scaling by the square root of d_k

This prevents a *training* instability rather than an overflow. If the components of $\mathbf{q}$ and $\mathbf{k}$ have mean 0 and variance 1, their dot product sums $d_k$ such products:

$$
\operatorname{Var}(\mathbf{q} \cdot \mathbf{k}) = d_k
$$

Large scores make softmax nearly one-hot, and its gradients close to zero (*saturation*). Dividing by $\sqrt{d_k}$ restores variance 1, since scaling by $a$ scales the variance by $a^2$:

$$
\operatorname{Var}\!\left(\frac{\mathbf{q} \cdot \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

Dividing by $d_k$ instead would overshoot, giving variance $1 / d_k$ and a nearly uniform softmax.

**In code:** `S = self.Q @ self.K.mT / np.sqrt(self.d_k)`.

---

## 7. Adam: epsilon

Adam divides each parameter's step by $\sqrt{\hat{v}}$, its typical gradient size. For a parameter whose gradients have all been 0, $\hat{v} = 0$. A small $\epsilon$ outside the square root prevents dividing by zero:

$$
\theta \leftarrow \theta - \eta\,\frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}, \qquad \epsilon = 10^{-8}
$$

It belongs **outside** the square root: inside, $\sqrt{10^{-8}} = 10^{-4}$ would set a much larger floor and change the step size for parameters with small gradients.

---

## 8. Summary

| where | problem | fix |
|---|---|---|
| softmax | $e^x$ overflow; $0 / 0$ | subtract each row's max |
| cross-entropy | $\log 0 = -\infty$ | log-softmax via shifted log-sum-exp |
| causal mask | blocking future positions safely | $-\infty$ scores; the diagonal is never masked |
| LayerNorm | division by zero variance | $\sqrt{\sigma^2 + \epsilon}$ |
| attention | softmax saturation | divide scores by $\sqrt{d_k}$ |
| Adam | division by zero | $\sqrt{\hat{v}} + \epsilon$ |

None of these change the maths: each gives the same result as the original formula wherever that formula is well-defined.