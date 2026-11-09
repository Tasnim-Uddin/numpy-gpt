# Optimisers

How the model's parameters are updated using their gradients: mini-batch SGD, then Adam.

## Contents

1. [The goal](#1-the-goal)
2. [Mini-batches and why they work](#2-mini-batches-and-why-they-work)
3. [Stochastic gradient descent](#3-stochastic-gradient-descent)
4. [The problems with SGD](#4-the-problems-with-sgd)
5. [Exponentially weighted moving averages](#5-exponentially-weighted-moving-averages)
6. [Adam](#6-adam)
7. [In code](#7-in-code)

---

## 1. The goal

Minimise the loss over the **whole training set** of $M$ positions:

$$
L_{\text{full}}(\boldsymbol{\theta}) = \frac{1}{M}\sum_{m=1}^{M}\ell_m(\boldsymbol{\theta})
$$

Gradient descent would repeat $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\,\nabla L_{\text{full}}$, but every step would need a forward and backward pass over all $M$ positions (about a million for Tiny Shakespeare).

---

## 2. Mini-batches and why they work

Each step instead uses a random **mini-batch** of $N = BT$ positions:

$$
L_{\text{batch}}(\boldsymbol{\theta}) = \frac{1}{N}\sum_{n \in \text{batch}}\ell_n(\boldsymbol{\theta})
$$

### The batch gradient is unbiased

If each batch position is drawn uniformly from the $M$ positions, its gradient is a random variable taking each value $\nabla\ell_m$ with probability $\frac{1}{M}$:

$$
\mathbb{E}\big[\nabla\ell_n\big] = \sum_{m=1}^{M}\frac{1}{M}\,\nabla\ell_m = \nabla L_{\text{full}}
$$

By linearity of expectation (which holds whether or not the positions are independent):

$$
\mathbb{E}\big[\nabla L_{\text{batch}}\big] = \frac{1}{N}\sum_{n}\mathbb{E}\big[\nabla\ell_n\big] = \nabla L_{\text{full}}
$$

So the batch gradient is an **unbiased estimator** of the full gradient: it points the right way on average. Its error is all **variance** (noise), which shrinks as the batch grows. If one position's gradient has variance $\sigma^2$, the average of $N$ independent positions has variance $\sigma^2 / N$.

Random sampling is essential: batches drawn from only one part of the data would give a biased gradient.

### Terminology

| name | data per step |
|---|---|
| (full) batch gradient descent | all $M$ positions |
| mini-batch SGD (used here) | $N$ random positions |
| "pure" SGD | one random example |

In deep learning, "batch" and "SGD" almost always mean the mini-batch versions.

---

## 3. Stochastic gradient descent

$$
\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \eta\,\bar{\boldsymbol{\theta}}^{(t)}
$$

where $\bar{\boldsymbol{\theta}}^{(t)}$ is the adjoint of **this step's** mini-batch loss and $\eta$ is the **learning rate**. It is applied to every parameter array, entry by entry.

**Why it reduces the loss.** To first order, for a small step $\Delta\boldsymbol{\theta}$:

$$
L(\boldsymbol{\theta} + \Delta\boldsymbol{\theta}) \approx L(\boldsymbol{\theta}) + \nabla L\,\Delta\boldsymbol{\theta}
$$

With $\Delta\boldsymbol{\theta} = -\eta\,\bar{\boldsymbol{\theta}}$, the change is $-\eta\lVert\bar{\boldsymbol{\theta}}\rVert^2 \le 0$.

**The learning rate.** Too small, and training is slow. Too large, and steps overshoot and the loss can blow up. For a quadratic loss, gradient descent converges only if $\eta < 2 / \lambda_{\max}$, where $\lambda_{\max}$ is the largest eigenvalue of the curvature.

---

## 4. The problems with SGD

1. **One learning rate for every parameter.** Each step is $\eta\lvert g_i \rvert$, so it depends on the gradient's size. If one parameter's gradients are around $0.001$ and another's around $10$, no single $\eta$ suits both: one barely moves, or the other overshoots.
2. **Noise.** Each mini-batch gradient is noisy, and SGD reacts fully to every one, so the parameters zig-zag.

Adam fixes problem 2 with **momentum** and problem 1 with **per-parameter scaling**. Both are built from exponentially weighted moving averages.

---

## 5. Exponentially weighted moving averages

A running average of a stream of values $x_1, x_2, \ldots$, starting from $a_0 = 0$:

$$
a_t = \beta\,a_{t-1} + (1 - \beta)\,x_t
$$

Unrolling the recursion:

$$
a_t = (1 - \beta)\sum_{k=1}^{t}\beta^{\,t-k}\,x_k
$$

Each value's weight shrinks by a factor of $\beta$ for every step it ages, so recent values count most. The average effectively covers about the last $\frac{1}{1 - \beta}$ values:

| $\beta$ | effective window |
|---|---|
| 0.9 | about 10 values |
| 0.999 | about 1,000 values |

It smooths noise without storing past values: just one running array per parameter.

---

## 6. Adam

### Momentum (first moment)

An exponential average of the gradients, $\mathbf{g}_t$:

$$
\mathbf{m}_t = \beta_1\,\mathbf{m}_{t-1} + (1 - \beta_1)\,\mathbf{g}_t
$$

Noise points different ways on different steps and cancels in the average, while the true direction adds up. $\mathbf{m}_t$ estimates the gradient's mean, its **first moment**.

### Second moment

An exponential average of the squared gradients (element-wise):

$$
\mathbf{v}_t = \beta_2\,\mathbf{v}_{t-1} + (1 - \beta_2)\,\mathbf{g}_t \odot \mathbf{g}_t
$$

$\sqrt{v_i}$ estimates the typical size of parameter $i$'s gradients.

### Per-parameter step sizes

Dividing the momentum by $\sqrt{v_i}$ normalises each parameter's step. If the gradients are consistent, $m_i \approx g$ and $\sqrt{v_i} \approx \lvert g \rvert$, so:

$$
\frac{m_i}{\sqrt{v_i}} \approx \pm 1
$$

and the step is about $\eta$ whatever the gradient's scale. Each parameter effectively gets its own learning rate, $\eta / \sqrt{v_i}$. If the gradients are mostly noise, $m_i$ averages towards 0 while $v_i$ stays large, so the step is small: confident where the direction is consistent, cautious where it isn't.

### Bias correction

Both averages start at 0, so early on they underestimate. If the gradients have a steady mean, then using the unrolled sum and a geometric series:

$$
\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1)\sum_{k=1}^{t}\beta_1^{\,t-k}\,\mathbb{E}[\mathbf{g}] = \left(1 - \beta_1^{\,t}\right)\mathbb{E}[\mathbf{g}]
$$

so dividing by $1 - \beta_1^{\,t}$ removes the bias (and likewise for $\mathbf{v}_t$):

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^{\,t}},
\qquad
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^{\,t}}
$$

Without the correction, $\mathbf{v}_t$ (with $\beta_2 = 0.999$) is far more underestimated than $\mathbf{m}_t$ in the first steps, which would make the first updates several times too large. As $t$ grows, $\beta^{\,t} \to 0$ and the correction fades away.

### The update

$$
\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta\,\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}
$$

All operations are element-wise.

| setting | default | meaning |
|---|---|---|
| $\eta$ | $10^{-3}$ | roughly how far each parameter moves per step |
| $\beta_1$ | 0.9 | momentum: about the last 10 gradients |
| $\beta_2$ | 0.999 | second moment: about the last 1,000 (large LLMs often use 0.95) |
| $\epsilon$ | $10^{-8}$ | avoids division by zero |

Adam stores an $\mathbf{m}$ and $\mathbf{v}$ for every parameter, plus a step counter $t$: about three times the memory of the parameters alone.

Reference: Kingma & Ba (2014), [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980).

---

## 7. In code

`src/optim.py`. Each optimiser's `step()` is called once per training step, after `model.backward()`:

- It calls `model.params_and_grads()` **every step**, because each backward pass creates new gradient arrays.
- It updates parameters **in place** (`param -= ...`), so the model's own arrays change. `param = param - ...` would only rebind the name.
- `Adam` keeps `self.m` and `self.v` as dictionaries from parameter name to array, created as zeros the first time each parameter is seen, and `self.t` as one counter shared by all parameters.

| maths | `Adam.step` |
|---|---|
| $t \leftarrow t + 1$ | `self.t += 1` (once per step, before the loop) |
| $\mathbf{m} \leftarrow \beta_1\mathbf{m} + (1 - \beta_1)\mathbf{g}$ | `self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad` |
| $\mathbf{v} \leftarrow \beta_2\mathbf{v} + (1 - \beta_2)\,\mathbf{g} \odot \mathbf{g}$ | `self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * grad * grad` |
| $\hat{\mathbf{m}}$, $\hat{\mathbf{v}}$ | `m_hat = self.m[name] / bias_correction1`, `v_hat = self.v[name] / bias_correction2` |
| update | `param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)` |