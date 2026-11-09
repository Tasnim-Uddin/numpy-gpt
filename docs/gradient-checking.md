# Gradient checking

How `src/test_gradients.py` verifies every backward pass, and why the method works.

## Contents

1. [The idea](#1-the-idea)
2. [The forward difference and its error](#2-the-forward-difference-and-its-error)
3. [The central difference](#3-the-central-difference)
4. [Choosing h](#4-choosing-h)
5. [Comparing the gradients](#5-comparing-the-gradients)
6. [The procedure](#6-the-procedure)
7. [Caveats](#7-caveats)

---

## 1. The idea

Each `backward` computes an **analytic** gradient $a_i = \partial L / \partial \theta_i$ for every parameter entry $\theta_i$, using hand-derived formulas. Gradient checking computes the same derivative an independent way, straight from the definition of a derivative, using only forward passes. If the two agree, the backward pass is correct.

A wrong backward pass doesn't raise an error: training just quietly fails to improve. Gradient checking catches the mistake before it costs a training run.

---

## 2. The forward difference and its error

A partial derivative is defined by nudging one entry along its basis direction $\mathbf{e}_i$:

$$
\frac{\partial L}{\partial \theta_i} = \lim_{h \to 0}\frac{L(\boldsymbol{\theta} + h\,\mathbf{e}_i) - L(\boldsymbol{\theta})}{h}
$$

With a small finite $h$, this is the **forward difference**. Writing $L(\theta_i)$ for the loss as a function of the single entry $\theta_i$, a Taylor expansion shows its error:

$$
L(\theta_i + h) = L + h\,L' + \frac{h^2}{2}L'' + \frac{h^3}{6}L''' + O(h^4)
$$

$$
\frac{L(\theta_i + h) - L(\theta_i)}{h} = L' + \frac{h}{2}L'' + O(h^2)
$$

The error is proportional to $h$: around $10^{-5}$ for $h = 10^{-5}$, too large to separate small bugs from approximation error.

---

## 3. The central difference

Nudging in both directions cancels the even-order terms:

$$
L(\theta_i - h) = L - h\,L' + \frac{h^2}{2}L'' - \frac{h^3}{6}L''' + O(h^4)
$$

$$
L(\theta_i + h) - L(\theta_i - h) = 2h\,L' + \frac{h^3}{3}L''' + O(h^5)
$$

$$
\frac{L(\theta_i + h) - L(\theta_i - h)}{2h} = L' + \frac{h^2}{6}L''' + O(h^4)
$$

The error is now proportional to $h^2$: around $10^{-10}$ for $h = 10^{-5}$. It costs two forward passes per entry instead of one.

---

## 4. Choosing h

A smaller $h$ reduces the Taylor error, but increases **rounding error**. Each computed loss has a relative error of about $\varepsilon \approx 2.2 \times 10^{-16}$ (`float64`), and subtracting two nearly equal losses then dividing by $2h$ amplifies it. Roughly:

$$
E(h) \approx \underbrace{\frac{h^2}{6}\lvert L''' \rvert}_{\text{Taylor}} + \underbrace{\frac{\varepsilon\lvert L \rvert}{h}}_{\text{rounding}}
$$

Setting $dE / dh = 0$:

$$
\frac{h}{3}\lvert L''' \rvert = \frac{\varepsilon\lvert L \rvert}{h^2}
\qquad\Rightarrow\qquad
h^3 = \frac{3\varepsilon\lvert L \rvert}{\lvert L''' \rvert}
$$

With $\lvert L \rvert$ and $\lvert L''' \rvert$ around 1, $h \approx (6.6 \times 10^{-16})^{1/3} \approx 9 \times 10^{-6}$. So $h = 10^{-5}$ is close to optimal, giving a total error around $10^{-10}$.

This is also why the check runs in `float64`: in `float32`, $\varepsilon \approx 10^{-7}$, and the best achievable error would be around $10^{-5}$.

---

## 5. Comparing the gradients

With analytic gradient $a$ and numerical gradient $n$:

$$
\text{absolute error} = \lvert a - n \rvert,
\qquad
\text{relative error} = \frac{\lvert a - n \rvert}{\max\big(\lvert a \rvert, \lvert n \rvert\big)}
$$

Each one has a blind spot that the other covers:

| case | $a$ | $n$ | absolute | relative | correct? |
|---|---|---|---|---|---|
| normal, correct | $0.0123456$ | $0.0123457$ | $10^{-7}$ | $8 \times 10^{-6}$ ✓ | yes |
| normal, bug | $0.0123456$ | $0.0246912$ | $10^{-2}$ ✗ | $0.5$ ✗ | no |
| tiny, correct | $10^{-12}$ | $3 \times 10^{-12}$ | $2 \times 10^{-12}$ ✓ | $0.67$ ✗ | yes (rounding noise) |
| large, correct | $0.50000001$ | $0.50000002$ | $10^{-8}$ ✗ | $2 \times 10^{-8}$ ✓ | yes |

- **Relative error** judges the difference against the size of the gradient, but gives false alarms when both values are tiny and dominated by rounding noise.
- **Absolute error** catches that case, but on its own would give false alarms for large gradients.

So a check **passes if either error is small**: relative below $10^{-5}$, or absolute below $10^{-9}$. A real bug fails both.

---

## 6. The procedure

1. Build a **tiny model** ($V = 11$, $d = 8$, 2 heads, 2 blocks) and a fixed random batch ($B = 2$, $T = 5$). Every layer type is present, and each forward pass is fast.
2. Run one forward and one backward pass, to get every analytic gradient.
3. For every parameter array, and every entry $\theta_i$ in it:
   - set $\theta_i \to \theta_i + h$ and compute $L^+$
   - set $\theta_i \to \theta_i - h$ and compute $L^-$
   - restore $\theta_i$
   - compare $n_i = (L^+ - L^-) / 2h$ with $a_i$
4. Report the worst errors per parameter array, so any failure points to a specific layer.

The parameter arrays come from `model.params_and_grads()`. They are the model's own arrays, not copies, so changing an entry changes the model's next forward pass.

**Run it:**

```bash
python src/test_gradients.py
```

It prints one line per parameter array and ends with `ALL GRADIENTS OK`.

---

## 7. Caveats

- **Cost.** Each entry needs two forward passes. The tiny model has about 2,000 entries, so checking all of them is quick. For large models, check a random sample of entries per parameter (`checks_per_param=5`).
- **Kinks.** ReLU's derivative jumps at 0. If a nudge of $\pm h$ pushes a hidden value across 0, the check can fail even though the code is correct. With $h = 10^{-5}$ this is rare. If one ReLU-related check fails while everything else passes, rerun with a different seed.
- **Depth.** Rounding errors accumulate through many layers, so very deep models show somewhat larger relative errors even when correct.