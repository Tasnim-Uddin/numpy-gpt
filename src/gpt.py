"""
A GPT (decoder-only transformer) in NumPy: forward and backward passes.

Notation (shapes in docstrings and comments):
    B           batch size (number of sequences)
    T           sequence length (tokens per sequence), T <= context_length
    N           total positions in the batch, N = B * T
    d_model     model dimension (size of each token's vector)
    h           number of attention heads (num_heads)
    d_k         per-head dimension, d_model // h
    V           vocabulary size (vocab_size)

Maths in comments:
    @           matrix product
    (*)         element-wise (Hadamard) product
    ^T          transpose (of the last two axes for batched arrays, i.e. .mT)

Layer interface (every layer follows this):
    forward(X)    computes the output Y and saves whatever backward will need
    backward(dY)  dY is the adjoint of the output, dL/dY.
                  Returns dX, the adjoint of the input (for the previous layer).
                  Stores parameter adjoints (dW, db, dgamma, dbeta) next to their
                  parameters, for the update step.
    An adjoint always has the same shape as its variable: each entry is the
    derivative of the loss L with respect to that entry.
"""

import numpy as np


class Softmax:
    """
    Softmax along the last axis (each row sums to 1):
    p_i = exp(x_i) / sum_j exp(x_j)
    """

    def forward(self, X):
        """
        X:       (..., n) scores, e.g. attention scores
        returns: (..., n) probabilities P
        """
        # Subtract each row's max so the largest exponent is e^0 = 1 (no overflow).
        # Doesn't change the result: the shift cancels in the division.
        shifted = X - X.max(axis=-1, keepdims=True)
        exps = np.exp(shifted)

        # Divide each row by its own total so it sums to 1; saved for backward
        self.P = exps / exps.sum(axis=-1, keepdims=True)
        return self.P

    def backward(self, dY):
        """
        dY:      (..., n) adjoint of the output P
        returns: (..., n) adjoint of the input, per row: dX = p (*) (dY - (p^T dY) 1)
                 (the same as multiplying by the Jacobian diag(p) - p p^T, without building it)
        """
        # p^T dY for each row: probability-weighted average of the incoming adjoint -> (..., 1)
        p_dot_dY = (self.P * dY).sum(axis=-1, keepdims=True)

        # Subtract it from every entry of its row, then multiply element-wise by p
        return self.P * (dY - p_dot_dY)


class CrossEntropyWithSoftmax:
    """
    Softmax + cross-entropy (mean negative log-likelihood), computed stably via log-softmax:
    L = -(1/N) sum_n log p^(n)[y_n]
    """

    def forward(self, logits, targets):
        """
        logits:  (B, T, V) raw scores Z
        targets: (B, T) correct next-token IDs y
        returns: scalar loss L
        """
        # Shift by each row's max so exp() can't overflow (doesn't change softmax)
        shifted = logits - logits.max(axis=-1, keepdims=True)

        # log p[k] = shifted[k] - log(sum over j of exp(shifted[j]))
        # Computed directly, so log(0) can never happen
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))   # (B, T, V)

        # Saved for backward
        self.P = np.exp(log_probs)   # (B, T, V) softmax probabilities
        self.targets = targets       # (B, T)    correct token IDs

        # Log-prob of the correct token at each position -> (B, T, 1)
        picked = np.take_along_axis(log_probs, targets[..., None], axis=-1)

        # Loss = -log p[target], averaged over all N positions
        return -picked.mean()

    def backward(self):
        """
        Top of the chain, so there is no upstream adjoint (dL/dL = 1).
        returns: (B, T, V) dZ = (P - Y_onehot) / N
        """
        B, T, V = self.P.shape
        N = B * T

        # One-hot targets: a 1 at each position's correct token, 0 elsewhere
        Y_onehot = np.zeros_like(self.P)                                       # (B, T, V)
        np.put_along_axis(Y_onehot, self.targets[..., None], 1.0, axis=-1)

        # Prediction minus target, divided by N because the loss is a mean
        return (self.P - Y_onehot) / N


class Z_LossWithSoftmax:
    """TODO: auxiliary z-loss, coef * mean(logsumexp(logits)**2). Training-only."""
    pass


class Linear:
    """
    Fully connected layer: Y = X W + 1 b^T
    Each output feature is a learned weighted sum of all input features, plus a bias.
    """

    def __init__(self, in_features, out_features):
        # W[i, j] = how much input feature i contributes to output feature j.
        # Small random values; training learns the real ones.
        self.W = np.random.randn(in_features, out_features) * 0.02   # (d_in, d_out)

        # One bias per output feature, starting at zero
        self.b = np.zeros(out_features)                               # (d_out,)

    def forward(self, X):
        """
        X:       (B, T, d_in)  input, one vector per position
        returns: (B, T, d_out) Y = X W + b
        """
        self.X = X   # saved for backward: dW = X^T dY

        # @ multiplies along the last axis (d_in); B and T pass through.
        # b broadcasts across every position (the 1 b^T term).
        return X @ self.W + self.b

    def backward(self, dY):
        """
        dY:      (B, T, d_out) adjoint of the output
        returns: (B, T, d_in)  dX = dY W^T
        stores:  self.dW = X^T dY  (d_in, d_out)
                 self.db = dY^T 1  (d_out,)
        """
        B, T, d_in = self.X.shape
        d_out = dY.shape[-1]
        N = B * T

        # Stack every position of every sequence as one row (the N x d matrices of the maths)
        X_rows = self.X.reshape(N, d_in)     # (N, d_in)
        dY_rows = dY.reshape(N, d_out)       # (N, d_out)

        # W is shared by every position: the matrix product sums input x error over all N rows
        self.dW = X_rows.T @ dY_rows         # (d_in, d_out), same shape as W

        # b is shared by every position: sum the adjoint over all N rows
        self.db = dY_rows.sum(axis=0)        # (d_out,), same shape as b

        # Send each output's error back along the same weights
        return dY @ self.W.T                 # (B, T, d_in), same shape as X


class ReLU:
    """Element-wise max(0, x): negatives become 0, positives pass through."""

    def forward(self, X):
        """
        X:       (...) any shape
        returns: (...) Y = max(0, X), same shape
        """
        self.X = X   # saved for backward: the mask 1[X > 0]

        # Non-linear step: stops stacked Linear layers collapsing into one
        return np.maximum(0, X)

    def backward(self, dY):
        """
        dY:      (...) adjoint of the output
        returns: (...) dX = dY (*) 1[X > 0]
        """
        # (self.X > 0) is True/False, used as 1/0:
        # pass the adjoint where the input was positive, block it elsewhere
        return dY * (self.X > 0)


class FeedForwardNetwork:
    """
    Per-position MLP: expand -> ReLU -> shrink. Processes what attention gathered.
    H1 = X W1 + b1,   H2 = ReLU(H1),   Y = H2 W2 + b2
    """

    def __init__(self, d_model):
        # fc = "fully connected" (a Linear layer)
        self.fc1 = Linear(d_model, 4 * d_model)   # expand: 4x room for pattern detectors
        self.relu = ReLU()                        # detectors either fire or go to 0
        self.fc2 = Linear(4 * d_model, d_model)   # shrink back so it fits the block

    def forward(self, X):
        """
        X:       (B, T, d_model) token vectors after the attention sub-layer
        returns: (B, T, d_model) processed vectors
        """
        H1 = self.fc1.forward(X)       # (B, T, 4*d_model) hidden pre-activations
        H2 = self.relu.forward(H1)     # (B, T, 4*d_model) keep only the detectors that fired
        return self.fc2.forward(H2)    # (B, T, d_model)   combine into an update

    def backward(self, dY):
        """
        dY:      (B, T, d_model) adjoint of the output
        returns: (B, T, d_model) adjoint of the input
        stores:  dW, db inside fc1 and fc2
        """
        # The forward steps in reverse: fc2, relu, fc1
        dH2 = self.fc2.backward(dY)     # dH2 = dY W2^T            (B, T, 4*d_model)
        dH1 = self.relu.backward(dH2)   # dH1 = dH2 (*) 1[H1 > 0]  (B, T, 4*d_model)
        return self.fc1.backward(dH1)   # dX  = dH1 W1^T           (B, T, d_model)


class LayerNorm:
    """
    Normalise each position's vector to mean 0, variance 1, then scale and shift:
    x_hat = (x - mean) / s,  s = sqrt(var + eps),  y = gamma (*) x_hat + beta
    """

    def __init__(self, d_model, eps=1e-5):
        self.eps = eps                    # tiny constant to avoid dividing by 0
        self.gamma = np.ones(d_model)     # learned gain (scale), starts as "no change"
        self.beta = np.zeros(d_model)     # learned bias (shift), starts as "no change"

    def forward(self, X):
        """
        X:       (B, T, d_model)
        returns: (B, T, d_model)
        """
        # Mean and variance of each position's own vector -> (B, T, 1)
        mean = X.mean(axis=-1, keepdims=True)
        var = X.var(axis=-1, keepdims=True)

        # Saved for backward
        self.std = np.sqrt(var + self.eps)      # s:     (B, T, 1)
        self.X_hat = (X - mean) / self.std      # x_hat: (B, T, d_model), mean 0, variance about 1

        # Learned per-feature scale and shift
        return self.gamma * self.X_hat + self.beta

    def backward(self, dY):
        """
        dY:      (B, T, d_model) adjoint of the output
        returns: (B, T, d_model) dX = (dX_hat - mean(dX_hat) - x_hat * mean(dX_hat (*) x_hat)) / s
        stores:  self.dgamma, self.dbeta  (d_model,) each
        """
        B, T, d = dY.shape
        N = B * T

        # gamma and beta are shared by every position: sum over all N rows
        self.dgamma = (dY * self.X_hat).reshape(N, d).sum(axis=0)   # sum_n dY (*) x_hat
        self.dbeta = dY.reshape(N, d).sum(axis=0)                   # sum_n dY

        # Adjoint of x_hat (back through the gain)
        dX_hat = self.gamma * dY                                                   # (B, T, d_model)

        # Back through the normalisation, per position (each has its own mean and s)
        mean_dX_hat = dX_hat.mean(axis=-1, keepdims=True)                          # (B, T, 1) route through the mean
        mean_dX_hat_X_hat = (dX_hat * self.X_hat).mean(axis=-1, keepdims=True)     # (B, T, 1) route through s
        return (dX_hat - mean_dX_hat - self.X_hat * mean_dX_hat_X_hat) / self.std  # (B, T, d_model)


class MultiHeadAttention:
    """
    Causal multi-head self-attention.
    head_i       = softmax(Q_i K_i^T / sqrt(d_k) + M) V_i
    MultiHead(X) = Concat(head_1, ..., head_h) W^O
    """

    def __init__(self, d_model, num_heads, context_length):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # per-head query/key/value size

        # One big projection each; columns are grouped by head (head i = cols i*d_k to (i+1)*d_k)
        self.W_q = Linear(d_model, d_model)   # W^Q
        self.W_k = Linear(d_model, d_model)   # W^K
        self.W_v = Linear(d_model, d_model)   # W^V
        self.W_o = Linear(d_model, d_model)   # W^O: mixes the heads' outputs back together

        self.softmax = Softmax()

        # Causal mask M, built once at max length: True above the diagonal = future
        self.mask = np.triu(np.ones((context_length, context_length), dtype=bool), k=1)

    def split_heads(self, X):
        """(B, T, d_model) -> (B, h, T, d_k)"""
        B, T, _ = X.shape
        # Split the feature axis into heads, then move heads next to batch
        return X.reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def merge_heads(self, X):
        """(B, h, T, d_k) -> (B, T, d_model), i.e. Concat(head_1, ..., head_h)"""
        B, h, T, d_k = X.shape
        # Move heads back next to features, then join them side by side
        return X.transpose(0, 2, 1, 3).reshape(B, T, h * d_k)

    def forward(self, X):
        """
        X:       (B, T, d_model) token vectors
        returns: (B, T, d_model) context-aware token vectors
        """
        T = X.shape[1]   # current sequence length (can be <= context_length)

        # 1-2. Project, then split into heads -> (B, h, T, d_k) each; saved for backward
        self.Q = self.split_heads(self.W_q.forward(X))
        self.K = self.split_heads(self.W_k.forward(X))
        self.V = self.split_heads(self.W_v.forward(X))

        # 3. Scaled scores, one T x T matrix per head -> (B, h, T, T)
        S = self.Q @ self.K.mT / np.sqrt(self.d_k)

        # 4. Causal mask: (T, T) broadcasts across batch and heads
        S = np.where(self.mask[:T, :T], -np.inf, S)

        # 5. Attention weights, each row sums to 1 -> (B, h, T, T); saved for backward
        self.A = self.softmax.forward(S)

        # 6. Weighted sum of values per head -> (B, h, T, d_k)
        O = self.A @ self.V

        # 7-8. Concatenate heads -> (B, T, d_model), then mix with W^O
        return self.W_o.forward(self.merge_heads(O))

    def backward(self, dY):
        """
        dY:      (B, T, d_model) adjoint of the output
        returns: (B, T, d_model) adjoint of the input
        stores:  dW, db inside W_q, W_k, W_v, W_o
        """
        # 8. Output projection (Linear)
        dO_merged = self.W_o.backward(dY)                  # (B, T, d_model)

        # 7. Merge was a rearrangement, so its adjoint is the reverse: split
        dO = self.split_heads(dO_merged)                   # (B, h, T, d_k)

        # 6. O = A V (matrix product rule): dA = dO V^T,  dV = A^T dO
        dA = dO @ self.V.mT                                # (B, h, T, T)
        dV = self.A.mT @ dO                                # (B, h, T, d_k)

        # 5. Softmax, row by row
        dS = self.softmax.backward(dA)                     # (B, h, T, T)

        # 4. Causal mask: nothing to do, dS is already 0 wherever A = 0 (masked)

        # 3. S = Q K^T / sqrt(d_k) (matrix product rule): dQ = dS K / sqrt(d_k),  dK = dS^T Q / sqrt(d_k)
        dQ = dS @ self.K / np.sqrt(self.d_k)               # (B, h, T, d_k)
        dK = dS.mT @ self.Q / np.sqrt(self.d_k)            # (B, h, T, d_k)

        # 2. Split was a rearrangement, so its adjoint is the reverse: merge
        dQ_merged = self.merge_heads(dQ)                   # (B, T, d_model)
        dK_merged = self.merge_heads(dK)                   # (B, T, d_model)
        dV_merged = self.merge_heads(dV)                   # (B, T, d_model)

        # 1. Projections (Linear): X fed all three, so its adjoint is the sum of the three routes
        return (self.W_q.backward(dQ_merged)
                + self.W_k.backward(dK_merged)
                + self.W_v.backward(dV_merged))            # (B, T, d_model)


class EmbeddingLayer:
    """
    Token embeddings via a lookup table:
    E[b, t] = W_E[token_ids[b, t]]   (equivalently E = U W_E, with U the one-hot token IDs)
    """

    def __init__(self, vocab_size, d_model):
        # Token embedding matrix W_E: row v is the embedding for token v.
        # Small random values (std 0.02) so activations start small.
        self.W = np.random.randn(vocab_size, d_model) * 0.02   # (V, d_model)

    def forward(self, token_ids):
        """
        token_ids: (B, T) integer token IDs
        returns:   (B, T, d_model) embeddings E
        """
        self.token_ids = token_ids   # saved for backward: which row each position used

        # Indexing with an array swaps each ID for its row
        return self.W[token_ids]

    def backward(self, dY):
        """
        dY:      (B, T, d_model) adjoint of the output embeddings
        returns: nothing (the input is integer token IDs, and this is the first layer)
        stores:  self.dW (V, d_model): row v = sum of dY over every position where token v appeared
        """
        B, T, d = dY.shape
        N = B * T

        # One row per position, as in the maths
        ids_rows = self.token_ids.reshape(N)     # (N,)       token at each position
        dY_rows = dY.reshape(N, d)               # (N, d_model) adjoint at each position

        # dW = U^T dY: add each position's adjoint into its token's row.
        # np.add.at ADDS for repeated IDs (dW[ids] += ... would keep only one of them)
        self.dW = np.zeros_like(self.W)          # tokens not in the batch keep 0
        np.add.at(self.dW, ids_rows, dY_rows)


class PositionalEncodings:
    """
    Learned positional embeddings (GPT-1/GPT-2 style): adds one learned vector per position.
    Y[b, t] = X[b, t] + W_P[t]   (the same position vector for every sequence b)
    """

    def __init__(self, context_length, d_model):
        # Positional embedding matrix W_P: row t is the vector for "position t".
        # Small random values (std 0.02), learned during training like W_E.
        self.W = np.random.randn(context_length, d_model) * 0.02   # (context_length, d_model)

    def forward(self, X):
        """
        X:       (B, T, d_model) token embeddings, T <= context_length
        returns: (B, T, d_model) token embeddings + position vectors
        """
        T = X.shape[1]

        # First T rows, (T, d_model); broadcasts so every sequence gets the same position vectors
        return X + self.W[:T]

    def backward(self, dY):
        """
        dY:      (B, T, d_model) adjoint of the output
        returns: (B, T, d_model) dX = dY (the addition passes the adjoint straight through)
        stores:  self.dW (context_length, d_model): rows 0..T-1 = sum of dY over the batch, rest 0
        """
        B, T, d = dY.shape

        # Rows T and beyond weren't used in this forward pass, so their adjoint is 0
        self.dW = np.zeros_like(self.W)

        # Each position's vector is shared by all B sequences: sum over the batch (not over positions)
        self.dW[:T] = dY.sum(axis=0)             # (T, d_model)

        return dY


class TransformerBlock:
    """
    One post-norm transformer block (original Transformer / GPT-1 style).
    S1 = X  + MultiHead(X),   X1 = LayerNorm_1(S1)
    S2 = X1 + FFN(X1),        Y  = LayerNorm_2(S2)
    """

    def __init__(self, d_model, num_heads, context_length):
        self.attn = MultiHeadAttention(d_model, num_heads, context_length)  # gather info from earlier tokens
        self.ln1 = LayerNorm(d_model)                                       # normalise after attention + residual
        self.ffn = FeedForwardNetwork(d_model)                              # process each token's gathered info
        self.ln2 = LayerNorm(d_model)                                       # normalise after FFN + residual

    def forward(self, X):
        """
        X:       (B, T, d_model)
        returns: (B, T, d_model)
        """
        # Attention sub-layer: residual add, then normalise
        S1 = X + self.attn.forward(X)        # residual sum
        X1 = self.ln1.forward(S1)

        # Feed-forward sub-layer: residual add, then normalise
        S2 = X1 + self.ffn.forward(X1)       # residual sum
        return self.ln2.forward(S2)

    def backward(self, dY):
        """
        dY:      (B, T, d_model) adjoint of the block's output
        returns: (B, T, d_model) adjoint of the block's input
        stores:  parameter adjoints inside attn, ln1, ffn, ln2
        """
        # Feed-forward sub-layer, reversed
        dS2 = self.ln2.backward(dY)                  # through LayerNorm_2
        dX1 = dS2 + self.ffn.backward(dS2)           # residual: direct route + route through the FFN

        # Attention sub-layer, reversed
        dS1 = self.ln1.backward(dX1)                 # through LayerNorm_1
        return dS1 + self.attn.backward(dS1)         # residual: direct route + route through attention


class GPT:
    """Decoder-only transformer: token IDs -> next-token logits."""

    def __init__(self, vocab_size, context_length, d_model, num_heads, num_layers):
        self.context_length = context_length   # max tokens per sequence

        # Inputs: what each token is (W_E) + where it is (W_P)
        self.token_embedding = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncodings(context_length, d_model)

        # num_layers separate transformer blocks, each with its own weights
        self.blocks = [
            TransformerBlock(d_model, num_heads, context_length)
            for _ in range(num_layers)
        ]

        # LM head: each token's d_model vector -> one score per vocabulary token
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids):
        """
        token_ids: (B, T) integer token IDs, T <= context_length
        returns:   (B, T, V) logits Z (next-token scores at every position)
        """
        T = token_ids.shape[1]
        assert T <= self.context_length, "sequence is longer than context_length"

        # 1. Token embeddings, then add positional embeddings -> (B, T, d_model)
        X = self.token_embedding.forward(token_ids)
        X = self.positional_encoding.forward(X)

        # 2. Each transformer block in turn -> (B, T, d_model)
        for block in self.blocks:
            X = block.forward(X)

        # 3. LM head: scores for every vocabulary token -> (B, T, V)
        return self.lm_head.forward(X)

    def backward(self, dZ):
        """
        dZ:      (B, T, V) adjoint of the logits (from the loss's backward)
        returns: nothing (the input is integer token IDs)
        stores:  every layer's parameter adjoints, inside that layer
        """
        # 3. LM head -> (B, T, d_model)
        dX = self.lm_head.backward(dZ)

        # 2. Blocks in reverse order, last block first -> (B, T, d_model)
        for block in reversed(self.blocks):
            dX = block.backward(dX)

        # 1. Positional embeddings (stores dW_P), then token embeddings (stores dW_E)
        dX = self.positional_encoding.backward(dX)
        self.token_embedding.backward(dX)