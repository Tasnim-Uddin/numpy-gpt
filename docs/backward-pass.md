# Derivations

Hand derivations of every backward pass in `src/gpt.py`. Each section gives the forward pass, derives the backward pass with index notation and the chain rule, and shows how the result maps to the code.

## Contents

1. [Conventions](#1-conventions)
2. [Linear layer](#2-linear-layer)
3. [Cross-entropy with softmax](#3-cross-entropy-with-softmax)
4. [Softmax](#4-softmax)
5. [ReLU and the feed-forward network](#5-relu-and-the-feed-forward-network)
6. [Layer normalisation](#6-layer-normalisation)
7. [Causal multi-head attention](#7-causal-multi-head-attention)
8. [Residual connections and the transformer block](#8-residual-connections-and-the-transformer-block)
9. [Token and positional embeddings](#9-token-and-positional-embeddings)
10. [The full backward pass](#10-the-full-backward-pass)

---

## 1. Conventions

**Positions as rows.** A batch has $B$ sequences of $T$ tokens, so $N = BT$ positions. Activations are written as matrices with one row per position: $X \in \mathbb{R}^{N \times d}$. In code, the same values are stored as arrays of shape `(B, T, d)`.

**Adjoints.** For any variable $A$, its *adjoint* $\bar{A}$ holds the derivative of the scalar loss $L$ with respect to each entry, in the same shape as $A$:

$$
\bar{A}_{ij} = \frac{\partial L}{\partial A_{ij}}
$$

In code, $\bar{A}$ is `dA`. Each layer's `backward` receives the adjoint of its output and returns the adjoint of its input.

**Chain rule.** If a layer computes $Y$ from $\theta$, the loss depends on $\theta$ only through the entries of $Y$:

$$
\frac{\partial L}{\partial \theta} = \sum_{n,j} \bar{Y}_{nj}\,\frac{\partial Y_{nj}}{\partial \theta}
$$

**Kronecker delta.** $\delta_{ij} = 1$ if $i = j$, and $0$ otherwise. It *sifts* sums: $\sum_j \delta_{ij}\, a_j = a_i$.

**Indicator.** $\mathbb{I}[\cdot]$ is $1$ if the condition holds, and $0$ otherwise.

**Element-wise product.** $A \odot B$ multiplies matching entries (the Hadamard product).

**Shared parameters.** A parameter used at every position (a weight matrix, bias, or gain) gets the *sum* of the adjoints from every position where it is used. An activation exists separately at each position, so its adjoint has no such sum.

**Sum over routes.** If a variable reaches the loss along several routes, its adjoint is the *sum* of the adjoints along each route.

---

## 2. Linear layer

### Forward

$$
Y = XW + \mathbf{1}\mathbf{b}^\top
\qquad\Longleftrightarrow\qquad
Y_{nj} = \sum_{i=1}^{d_{\text{in}}} X_{ni}\,W_{ij} + b_j
$$

| symbol | meaning | shape |
|---|---|---|
| $X$ | input | $N \times d_{\text{in}}$ |
| $W$ | weights | $d_{\text{in}} \times d_{\text{out}}$ |
| $\mathbf{b}$ | bias | $d_{\text{out}}$ |
| $\mathbf{1}$ | vector of ones (adds $\mathbf{b}^\top$ to every row) | $N$ |
| $Y$ | output | $N \times d_{\text{out}}$ |

**Given** $\bar{Y}$. **Find** $\bar{W}$, $\bar{\mathbf{b}}$ (stored) and $\bar{X}$ (returned).

### Weights

$W_{ab}$ appears in $Y_{nj}$ only through the $i = a$ term, and only when $j = b$:

$$
\frac{\partial Y_{nj}}{\partial W_{ab}} = X_{na}\,\delta_{jb}
$$

$$
\bar{W}_{ab}
= \sum_{n,j} \bar{Y}_{nj}\, X_{na}\,\delta_{jb}
= \sum_{n} X_{na}\,\bar{Y}_{nb}
= \left(X^\top \bar{Y}\right)_{ab}
$$

$$
\bar{W} = X^\top \bar{Y}
\qquad (d_{\text{in}} \times N)(N \times d_{\text{out}}) = d_{\text{in}} \times d_{\text{out}}
$$

### Bias

$$
\frac{\partial Y_{nj}}{\partial b_c} = \delta_{jc}
\qquad\Rightarrow\qquad
\bar{b}_c = \sum_{n,j} \bar{Y}_{nj}\,\delta_{jc} = \sum_{n} \bar{Y}_{nc}
$$

$$
\bar{\mathbf{b}} = \bar{Y}^\top \mathbf{1}
\qquad (d_{\text{out}} \times N)(N) = d_{\text{out}}
$$

The bias is shared by every position, so its adjoint sums over all $N$ rows.

### Input

Output row $m$ depends only on input row $m$:

$$
\frac{\partial Y_{mj}}{\partial X_{na}} = \delta_{mn}\,W_{aj}
$$

$$
\bar{X}_{na}
= \sum_{m,j} \bar{Y}_{mj}\,\delta_{mn}\,W_{aj}
= \sum_{j} \bar{Y}_{nj}\,W_{aj}
= \left(\bar{Y} W^\top\right)_{na}
$$

$$
\bar{X} = \bar{Y} W^\top
\qquad (N \times d_{\text{out}})(d_{\text{out}} \times d_{\text{in}}) = N \times d_{\text{in}}
$$

Each output's error is sent back along the same weights it was computed with.

### Summary

$$
\bar{W} = X^\top \bar{Y},
\qquad
\bar{\mathbf{b}} = \bar{Y}^\top \mathbf{1},
\qquad
\bar{X} = \bar{Y} W^\top
$$

### In code (`Linear.backward`)

| maths | code |
|---|---|
| $X$ and $\bar{Y}$ as $N \times d$ matrices | `X_rows = self.X.reshape(N, d_in)`, `dY_rows = dY.reshape(N, d_out)` |
| $\bar{W} = X^\top \bar{Y}$ | `self.dW = X_rows.T @ dY_rows` |
| $\bar{\mathbf{b}} = \bar{Y}^\top \mathbf{1}$ | `self.db = dY_rows.sum(axis=0)` |
| $\bar{X} = \bar{Y} W^\top$ | `return dY @ self.W.T` |

---

## 3. Cross-entropy with softmax

### Forward

For position $n$ with logits $\mathbf{z}^{(n)} \in \mathbb{R}^{V}$ and correct next token $y_n$:

$$
p^{(n)}_k = \frac{\exp\big(z^{(n)}_k\big)}{\sum_{j=1}^{V}\exp\big(z^{(n)}_j\big)},
\qquad
L = -\frac{1}{N}\sum_{n=1}^{N}\log p^{(n)}_{y_n}
$$

This is the mean negative log-likelihood of a categorical (Multinoulli) distribution over the next token.

| symbol | meaning | shape |
|---|---|---|
| $Z$ | logits, row $n$ is $\mathbf{z}^{(n)\top}$ | $N \times V$ |
| $P$ | probabilities, row $n$ is $\mathbf{p}^{(n)\top}$ | $N \times V$ |
| $Y$ | one-hot targets, row $n$ is $\mathbf{e}_{y_n}^\top$ | $N \times V$ |
| $L$ | loss | scalar |

**Given** nothing (this is the top of the chain: $\partial L / \partial L = 1$). **Find** $\bar{Z}$.

### One position

Writing the loss for one position with the softmax expanded:

$$
\ell = -\log p_y = -z_y + \log\sum_{j=1}^{V}\exp(z_j)
$$

Differentiating with respect to $z_k$, using $\partial z_j / \partial z_k = \delta_{jk}$:

$$
\frac{\partial \ell}{\partial z_k}
= -\delta_{yk} + \frac{\sum_j \exp(z_j)\,\delta_{jk}}{\sum_j \exp(z_j)}
= -\delta_{yk} + p_k
$$

$$
\nabla_{\mathbf{z}}\,\ell = \left(\mathbf{p} - \mathbf{e}_y\right)^\top
$$

The predicted probabilities minus the one-hot target.

### The batch

Each $\ell_n$ depends only on its own logits, and the mean contributes $\frac{1}{N}$:

$$
\bar{Z} = \frac{1}{N}\left(P - Y\right)
\qquad N \times V
$$

Each row of $\bar{Z}$ sums to $0$, since $\sum_k p_k = 1$: cross-entropy never moves all the logits up or down together.

The forward pass is computed through log-softmax to avoid overflow and $\log 0$ (see `numerical-stability.md`). This doesn't change the gradient.

### In code (`CrossEntropyWithSoftmax.backward`)

| maths | code |
|---|---|
| $P$ | `self.P`, saved in `forward` |
| $Y$ (one-hot) | `Y_onehot`, built with `np.put_along_axis(..., self.targets[..., None], 1.0, axis=-1)` |
| $\bar{Z} = \frac{1}{N}(P - Y)$ | `return (self.P - Y_onehot) / N` |

---

## 4. Softmax

Used row by row on the attention scores.

### Forward

For one row $\mathbf{z} \in \mathbb{R}^{n}$:

$$
p_i = \frac{\exp(z_i)}{S},
\qquad
S = \sum_{j=1}^{n}\exp(z_j)
$$

**Given** $\bar{\mathbf{p}}$. **Find** $\bar{\mathbf{z}}$.

### The Jacobian

By the quotient rule, with $\partial \exp(z_i) / \partial z_k = \exp(z_i)\,\delta_{ik}$ and $\partial S / \partial z_k = \exp(z_k)$:

$$
\frac{\partial p_i}{\partial z_k}
= \frac{\exp(z_i)\,\delta_{ik}\,S - \exp(z_i)\exp(z_k)}{S^2}
= p_i\left(\delta_{ik} - p_k\right)
$$

$$
\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top
\qquad n \times n
$$

### The adjoint

$$
\bar{z}_k
= \sum_{i} \bar{p}_i\,p_i\left(\delta_{ik} - p_k\right)
= p_k\,\bar{p}_k - p_k\sum_{i} p_i\,\bar{p}_i
= p_k\left(\bar{p}_k - \mathbf{p}^\top\bar{\mathbf{p}}\right)
$$

$$
\bar{\mathbf{z}} = \mathbf{p} \odot \left(\bar{\mathbf{p}} - \left(\mathbf{p}^\top\bar{\mathbf{p}}\right)\mathbf{1}\right)
$$

This gives the same result as multiplying by the $n \times n$ Jacobian, without ever building it.

### Many rows at once

With one softmax per row of $P$, let $\operatorname{rowsum}(M) = M\mathbf{1}$ hold each row's sum:

$$
\bar{Z} = P \odot \left(\bar{P} - \operatorname{rowsum}\left(P \odot \bar{P}\right)\mathbf{1}^\top\right)
$$

$\operatorname{rowsum}(P \odot \bar{P})\,\mathbf{1}^\top$ copies each row's value of $\mathbf{p}^\top\bar{\mathbf{p}}$ across that row.

### In code (`Softmax.backward`)

| maths | code |
|---|---|
| $P$ | `self.P`, saved in `forward` |
| $\operatorname{rowsum}(P \odot \bar{P})$ | `p_dot_dY = (self.P * dY).sum(axis=-1, keepdims=True)` |
| $P \odot (\bar{P} - \ldots\mathbf{1}^\top)$ | `return self.P * (dY - p_dot_dY)` (the `(..., 1)` shape broadcasts across each row) |

---

## 5. ReLU and the feed-forward network

### ReLU

$$
Y_i = \max(0, X_i)
\qquad\Rightarrow\qquad
\frac{\partial Y_i}{\partial X_k} = \mathbb{I}[X_i > 0]\,\delta_{ik}
$$

(At exactly $0$ the derivative is undefined, and the convention is to use $0$.) The Jacobian is diagonal, so:

$$
\bar{X}_k = \sum_{i}\bar{Y}_i\,\mathbb{I}[X_i > 0]\,\delta_{ik} = \bar{Y}_k\,\mathbb{I}[X_k > 0]
\qquad\Rightarrow\qquad
\bar{X} = \bar{Y} \odot \mathbb{I}[X > 0]
$$

The adjoint passes through where the input was positive, and is blocked where it wasn't.

### Feed-forward network: forward

With $d_{\text{ff}} = 4d$:

$$
H_1 = XW_1 + \mathbf{1}\mathbf{b}_1^\top,
\qquad
H_2 = \operatorname{ReLU}(H_1),
\qquad
Y = H_2W_2 + \mathbf{1}\mathbf{b}_2^\top
$$

| symbol | shape |
|---|---|
| $X$, $Y$ | $N \times d$ |
| $W_1$, $\mathbf{b}_1$ | $d \times d_{\text{ff}}$, $d_{\text{ff}}$ |
| $H_1$, $H_2$ | $N \times d_{\text{ff}}$ |
| $W_2$, $\mathbf{b}_2$ | $d_{\text{ff}} \times d$, $d$ |

### Feed-forward network: backward

The network is a chain $X \to H_1 \to H_2 \to Y$, so the backward pass applies the Linear and ReLU results in reverse:

$$
\begin{aligned}
\bar{H}_2 &= \bar{Y}\,W_2^\top, &\quad \bar{W}_2 &= H_2^\top\bar{Y}, &\quad \bar{\mathbf{b}}_2 &= \bar{Y}^\top\mathbf{1} \\
\bar{H}_1 &= \bar{H}_2 \odot \mathbb{I}[H_1 > 0] \\
\bar{X} &= \bar{H}_1\,W_1^\top, &\quad \bar{W}_1 &= X^\top\bar{H}_1, &\quad \bar{\mathbf{b}}_1 &= \bar{H}_1^\top\mathbf{1}
\end{aligned}
$$

Combined:

$$
\bar{X} = \Big(\big(\bar{Y}\,W_2^\top\big) \odot \mathbb{I}[H_1 > 0]\Big)\,W_1^\top
$$

### In code

| maths | code |
|---|---|
| $\bar{X} = \bar{Y} \odot \mathbb{I}[X > 0]$ | `ReLU.backward`: `return dY * (self.X > 0)` |
| $\bar{H}_2$, $\bar{W}_2$, $\bar{\mathbf{b}}_2$ | `dH2 = self.fc2.backward(dY)` |
| $\bar{H}_1$ | `dH1 = self.relu.backward(dH2)` |
| $\bar{X}$, $\bar{W}_1$, $\bar{\mathbf{b}}_1$ | `return self.fc1.backward(dH1)` |

---

## 6. Layer normalisation

### Forward

Each position's vector $\mathbf{x} \in \mathbb{R}^{d}$ is normalised on its own:

$$
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i,
\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2,
\qquad
s = \sqrt{\sigma^2 + \epsilon}
$$

$$
\hat{x}_i = \frac{x_i - \mu}{s},
\qquad
\mathbf{y} = \boldsymbol{\gamma} \odot \hat{\mathbf{x}} + \boldsymbol{\beta}
$$

| symbol | meaning | shape |
|---|---|---|
| $\mathbf{x}$, $\hat{\mathbf{x}}$, $\mathbf{y}$ | input, normalised input, output (one position) | $d$ |
| $\mu$, $\sigma^2$, $s$ | that vector's mean, variance, and $\sqrt{\sigma^2 + \epsilon}$ | scalars |
| $\boldsymbol{\gamma}$, $\boldsymbol{\beta}$ | gain and bias, shared by every position | $d$ |

**Given** $\bar{\mathbf{y}}$ at each position. **Find** $\bar{\boldsymbol{\gamma}}$, $\bar{\boldsymbol{\beta}}$ (stored) and $\bar{\mathbf{x}}$ (returned).

### Gain and bias

$y_i = \gamma_i\hat{x}_i + \beta_i$ is element-wise, so $\partial y_i / \partial\gamma_k = \hat{x}_i\,\delta_{ik}$ and $\partial y_i / \partial\beta_k = \delta_{ik}$. Both parameters are shared by every position, so their adjoints sum over positions:

$$
\bar{\boldsymbol{\gamma}} = \sum_{n=1}^{N}\bar{\mathbf{y}}^{(n)} \odot \hat{\mathbf{x}}^{(n)},
\qquad
\bar{\boldsymbol{\beta}} = \sum_{n=1}^{N}\bar{\mathbf{y}}^{(n)}
$$

### Back through the gain

$$
\mathbf{g} := \bar{\hat{\mathbf{x}}} = \boldsymbol{\gamma} \odot \bar{\mathbf{y}}
$$

### The Jacobian of the normalisation

$\mu$ and $s$ depend on every $x_k$, so each $\hat{x}_i$ depends on every $x_k$, through three routes: directly, through the mean, and through the standard deviation.

$$
\frac{\partial \mu}{\partial x_k} = \frac{1}{d},
\qquad
\frac{\partial (x_i - \mu)}{\partial x_k} = \delta_{ik} - \frac{1}{d}
$$

$$
\frac{\partial \sigma^2}{\partial x_k}
= \frac{1}{d}\sum_{i} 2(x_i - \mu)\left(\delta_{ik} - \frac{1}{d}\right)
= \frac{2}{d}(x_k - \mu)
$$

(the second part vanishes because $\sum_i (x_i - \mu) = 0$), so

$$
\frac{\partial s}{\partial x_k} = \frac{1}{2s}\cdot\frac{2}{d}(x_k - \mu) = \frac{\hat{x}_k}{d}
$$

By the product rule on $\hat{x}_i = (x_i - \mu)\,s^{-1}$:

$$
\frac{\partial \hat{x}_i}{\partial x_k}
= \frac{1}{s}\left(\delta_{ik} - \frac{1}{d}\right) - \frac{x_i - \mu}{s^2}\cdot\frac{\hat{x}_k}{d}
= \frac{1}{s}\left(\delta_{ik} - \frac{1}{d} - \frac{1}{d}\,\hat{x}_i\,\hat{x}_k\right)
$$

$$
\frac{\partial \hat{\mathbf{x}}}{\partial \mathbf{x}} = \frac{1}{s}\left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top\right)
$$

The three terms are the direct route, the route through the mean, and the route through the standard deviation.

### The input adjoint

$$
\bar{x}_k = \sum_{i} g_i\,\frac{\partial \hat{x}_i}{\partial x_k}
= \frac{1}{s}\left(g_k - \frac{1}{d}\sum_{i} g_i - \hat{x}_k\cdot\frac{1}{d}\sum_{i} g_i\,\hat{x}_i\right)
$$

$$
\bar{\mathbf{x}} = \frac{1}{s}\Big(\mathbf{g} - \operatorname{mean}(\mathbf{g})\,\mathbf{1} - \hat{\mathbf{x}}\,\operatorname{mean}(\mathbf{g} \odot \hat{\mathbf{x}})\Big)
$$

Each position uses only its own $\mathbf{g}$, $\hat{\mathbf{x}}$ and $s$: the input isn't shared, so there's no sum over positions.

**Check.** LayerNorm's output doesn't change if the same constant is added to every $x_i$, so the entries of $\bar{\mathbf{x}}$ must sum to zero. Since $\sum_k \hat{x}_k = 0$:

$$
\sum_{k}\bar{x}_k = \frac{1}{s}\left(\sum_k g_k - \sum_k g_k - 0\right) = 0
$$

### In code (`LayerNorm.backward`)

| maths | code |
|---|---|
| $s$, $\hat{\mathbf{x}}$ | `self.std`, `self.X_hat`, saved in `forward` |
| $\bar{\boldsymbol{\gamma}}$, $\bar{\boldsymbol{\beta}}$ | `(dY * self.X_hat).reshape(N, d).sum(axis=0)`, `dY.reshape(N, d).sum(axis=0)` |
| $\mathbf{g} = \boldsymbol{\gamma} \odot \bar{\mathbf{y}}$ | `dX_hat = self.gamma * dY` |
| $\operatorname{mean}(\mathbf{g})$ | `mean_dX_hat = dX_hat.mean(axis=-1, keepdims=True)` |
| $\operatorname{mean}(\mathbf{g} \odot \hat{\mathbf{x}})$ | `mean_dX_hat_X_hat = (dX_hat * self.X_hat).mean(axis=-1, keepdims=True)` |
| $\bar{\mathbf{x}}$ | `return (dX_hat - mean_dX_hat - self.X_hat * mean_dX_hat_X_hat) / self.std` |

---

## 7. Causal multi-head attention

### Forward

| step | operation | shape (code) |
|---|---|---|
| 1 | $Q_{\text{full}} = XW^Q + \mathbf{1}\mathbf{b}_Q^\top$, and likewise $K_{\text{full}}$, $V_{\text{full}}$ | $(B, T, d)$ |
| 2 | split into heads: $Q$, $K$, $V$ | $(B, h, T, d_k)$ |
| 3 | scores: $S = QK^\top / \sqrt{d_k}$ | $(B, h, T, T)$ |
| 4 | causal mask: entries above the diagonal set to $-\infty$ | $(B, h, T, T)$ |
| 5 | attention weights: $A = \operatorname{softmax}(S)$, row by row | $(B, h, T, T)$ |
| 6 | weighted values: $O = AV$ | $(B, h, T, d_k)$ |
| 7 | merge heads: $O_{\text{merged}}$ | $(B, T, d)$ |
| 8 | output projection: $Y = O_{\text{merged}}W^O + \mathbf{1}\mathbf{b}_O^\top$ | $(B, T, d)$ |

Steps 3 to 6 act separately on each sequence and head, so they are derived for one sequence and one head, where $Q, K, V, O \in \mathbb{R}^{T \times d_k}$ and $S, A \in \mathbb{R}^{T \times T}$.

**Given** $\bar{Y}$. **Find** the parameter adjoints of the four projections (stored) and $\bar{X}$ (returned).

### The matrix product rule

Two steps multiply two activations, so both factors need adjoints. For $C = MN$, with $C_{ij} = \sum_k M_{ik}N_{kj}$:

$$
\frac{\partial C_{ij}}{\partial M_{ab}} = \delta_{ia}\,N_{bj}
\qquad\Rightarrow\qquad
\bar{M}_{ab} = \sum_{j}\bar{C}_{aj}\,N_{bj} = \left(\bar{C}N^\top\right)_{ab}
$$

$$
\frac{\partial C_{ij}}{\partial N_{ab}} = M_{ia}\,\delta_{jb}
\qquad\Rightarrow\qquad
\bar{N}_{ab} = \sum_{i}M_{ia}\,\bar{C}_{ib} = \left(M^\top\bar{C}\right)_{ab}
$$

$$
\bar{M} = \bar{C}N^\top,
\qquad
\bar{N} = M^\top\bar{C}
$$

### Backward, step by step (reverse order)

**Step 8, output projection.** A Linear layer:

$$
\bar{O}_{\text{merged}} = \bar{Y}(W^O)^\top
$$

**Step 7, merge heads.** Merging only rearranges entries, so its adjoint is the reverse rearrangement:

$$
\bar{O} = \operatorname{split}\big(\bar{O}_{\text{merged}}\big)
$$

**Step 6, $O = AV$.** The product rule with $M = A$, $N = V$:

$$
\bar{A} = \bar{O}\,V^\top \quad (T \times T),
\qquad
\bar{V} = A^\top\bar{O} \quad (T \times d_k)
$$

**Step 5, softmax.** Row by row (Section 4):

$$
\bar{S} = A \odot \left(\bar{A} - \operatorname{rowsum}\left(A \odot \bar{A}\right)\mathbf{1}^\top\right)
$$

**Step 4, mask.** Masked scores were replaced by a constant, so their adjoint is $0$. This is already true: masked positions have $A_{ij} = 0$, so step 5 gives $\bar{S}_{ij} = 0$ there.

**Step 3, $S = QK^\top / \sqrt{d_k}$.** With $S_{ij} = \frac{1}{\sqrt{d_k}}\sum_m Q_{im}K_{jm}$:

$$
\frac{\partial S_{ij}}{\partial Q_{ab}} = \frac{1}{\sqrt{d_k}}\,\delta_{ia}\,K_{jb}
\qquad\Rightarrow\qquad
\bar{Q} = \frac{1}{\sqrt{d_k}}\,\bar{S}\,K
$$

$$
\frac{\partial S_{ij}}{\partial K_{ab}} = \frac{1}{\sqrt{d_k}}\,Q_{ib}\,\delta_{ja}
\qquad\Rightarrow\qquad
\bar{K} = \frac{1}{\sqrt{d_k}}\,\bar{S}^\top Q
$$

Both have shape $T \times d_k$.

**Step 2, split heads.** The reverse rearrangement, merge:

$$
\bar{Q}_{\text{full}} = \operatorname{merge}(\bar{Q}),
\quad
\bar{K}_{\text{full}} = \operatorname{merge}(\bar{K}),
\quad
\bar{V}_{\text{full}} = \operatorname{merge}(\bar{V})
$$

**Step 1, projections.** $X$ feeds all three projections, so its adjoint is the sum over the three routes:

$$
\bar{X} = \bar{Q}_{\text{full}}(W^Q)^\top + \bar{K}_{\text{full}}(W^K)^\top + \bar{V}_{\text{full}}(W^V)^\top
$$

$Q$, $K$ and $V$ are activations, separate for each sequence and head, so their adjoints have no sum over the batch. The sums over positions happen inside each projection's $\bar{W} = X^\top\bar{Y}$, because those weights are shared.

### In code (`MultiHeadAttention.backward`)

| step | maths | code |
|---|---|---|
| 8 | $\bar{O}_{\text{merged}}$ | `dO_merged = self.W_o.backward(dY)` |
| 7 | $\bar{O}$ | `dO = self.split_heads(dO_merged)` |
| 6 | $\bar{A} = \bar{O}V^\top$, $\bar{V} = A^\top\bar{O}$ | `dA = dO @ self.V.mT`, `dV = self.A.mT @ dO` |
| 5 | $\bar{S}$ | `dS = self.softmax.backward(dA)` |
| 3 | $\bar{Q}$, $\bar{K}$ | `dQ = dS @ self.K / np.sqrt(self.d_k)`, `dK = dS.mT @ self.Q / np.sqrt(self.d_k)` |
| 2 | merge | `self.merge_heads(dQ)`, and likewise for `dK`, `dV` |
| 1 | $\bar{X}$ | `self.W_q.backward(...) + self.W_k.backward(...) + self.W_v.backward(...)` |

`.mT` transposes the last two axes only, so it is the per-head transpose for every sequence and head at once.

---

## 8. Residual connections and the transformer block

### A residual connection

$$
S = X + F(X)
$$

where $F$ is a sub-layer (attention or the feed-forward network). $S$ is a sum, so $\partial S / \partial F = I$ and $\bar{F} = \bar{S}$. $X$ reaches $S$ along two routes, directly and through $F$:

$$
\frac{\partial S}{\partial X} = I + \frac{\partial F}{\partial X}
\qquad\Rightarrow\qquad
\bar{X} = \underbrace{\bar{S}}_{\text{direct}} + \underbrace{F.\text{backward}\big(\bar{S}\big)}_{\text{through } F}
$$

The direct term passes the adjoint back unchanged, which gives the gradient a clear path through deep stacks of blocks.

### The post-norm block

Forward:

$$
S_1 = X + \operatorname{MultiHead}(X),
\quad
X_1 = \operatorname{LN}_1(S_1),
\quad
S_2 = X_1 + \operatorname{FFN}(X_1),
\quad
Y = \operatorname{LN}_2(S_2)
$$

Backward, in reverse, applying the residual rule at each sum:

$$
\begin{aligned}
\bar{S}_2 &= \operatorname{LN}_2.\text{backward}(\bar{Y}) \\
\bar{X}_1 &= \bar{S}_2 + \operatorname{FFN}.\text{backward}(\bar{S}_2) \\
\bar{S}_1 &= \operatorname{LN}_1.\text{backward}(\bar{X}_1) \\
\bar{X} &= \bar{S}_1 + \operatorname{MultiHead}.\text{backward}(\bar{S}_1)
\end{aligned}
$$

### In code (`TransformerBlock.backward`)

```python
dS2 = self.ln2.backward(dY)
dX1 = dS2 + self.ffn.backward(dS2)
dS1 = self.ln1.backward(dX1)
return dS1 + self.attn.backward(dS1)
```

---

## 9. Token and positional embeddings

### Token embeddings

**Forward.** Each position looks up the row of the embedding matrix for its token $y_{bt}$:

$$
E_{btk} = (W_E)_{y_{bt},\,k},
\qquad
W_E \in \mathbb{R}^{V \times d}
$$

**Backward.** Entry $(W_E)_{vc}$ is copied into $E_{btk}$ exactly when the token there is $v$ and the feature matches:

$$
\frac{\partial E_{btk}}{\partial (W_E)_{vc}} = \delta_{y_{bt},\,v}\;\delta_{kc}
$$

$$
(\bar{W}_E)_{vc} = \sum_{b,t,k}\bar{E}_{btk}\,\delta_{y_{bt},\,v}\,\delta_{kc} = \sum_{(b,t)\,:\,y_{bt} = v}\bar{E}_{btc}
$$

Row $v$ of $\bar{W}_E$ is the sum of the adjoints at every position where token $v$ appears.

**As a Linear layer.** With $U \in \{0, 1\}^{N \times V}$ the one-hot token IDs, the lookup is $E = UW_E$, so the Linear result gives the same thing:

$$
\bar{W}_E = U^\top\bar{E}
$$

No input adjoint is returned: the input is integer token IDs, and this is the first layer.

### Positional embeddings

**Forward.** The same position vector is added to every sequence:

$$
Y_{btk} = X_{btk} + (W_P)_{tk},
\qquad
W_P \in \mathbb{R}^{C \times d}, \quad t = 1, \ldots, T \le C
$$

**Input.** Each $X_{btk}$ appears in exactly one output entry, with coefficient 1:

$$
\bar{X} = \bar{Y}
$$

**Parameters.** $(W_P)_{ac}$ appears in $Y_{bac}$ for every sequence $b$, and not at all when $a > T$:

$$
\frac{\partial Y_{btk}}{\partial (W_P)_{ac}} = \delta_{ta}\,\delta_{kc}
\qquad\Rightarrow\qquad
\bar{W}_P[a, :] =
\begin{cases}
\displaystyle\sum_{b=1}^{B}\bar{Y}[b, a, :] & a \le T \\[6pt]
\mathbf{0} & a > T
\end{cases}
$$

Each position's vector is shared across the batch, so its adjoint sums over $b$ but not over $t$.

### In code

| maths | code |
|---|---|
| $(\bar{W}_E)_v = \sum_{y_{bt} = v}\bar{E}_{bt}$ | `np.add.at(self.dW, ids_rows, dY_rows)` (adds, so repeated tokens accumulate) |
| $\bar{X} = \bar{Y}$ | `PositionalEncodings.backward`: `return dY` |
| $\bar{W}_P$, rows $1..T$ | `self.dW[:T] = dY.sum(axis=0)` |
| $\bar{W}_P$, rows beyond $T$ | `self.dW = np.zeros_like(self.W)` |

---

## 10. The full backward pass

Forward, with $\ell$ the number of blocks:

$$
X_0 = \operatorname{PosEnc}\big(\operatorname{Embed}(\text{ids})\big),
\qquad
X_i = \operatorname{Block}_i(X_{i-1}),
\qquad
Z = \operatorname{LMHead}(X_\ell),
\qquad
L = \operatorname{CE}(Z)
$$

Backward, starting from the loss:

$$
\begin{aligned}
\bar{Z} &= \tfrac{1}{N}\left(P - Y\right) \\
\bar{X}_\ell &= \operatorname{LMHead}.\text{backward}(\bar{Z}) \\
\bar{X}_{i-1} &= \operatorname{Block}_i.\text{backward}(\bar{X}_i), \qquad i = \ell, \ldots, 1 \\
\bar{E} &= \operatorname{PosEnc}.\text{backward}(\bar{X}_0) \\
&\operatorname{Embed}.\text{backward}(\bar{E})
\end{aligned}
$$

Every call stores its own parameter adjoints along the way, and `params_and_grads()` collects them for the optimiser.

### In code

```python
dZ = loss_fn.backward()                    # Z-bar = (P - Y) / N
dX = model.lm_head.backward(dZ)
for block in reversed(model.blocks):       # last block first
    dX = block.backward(dX)
dX = model.positional_encoding.backward(dX)
model.token_embedding.backward(dX)
```
