# numpy-gpt

A GPT (decoder-only transformer) built from scratch in NumPy, with **every forward and backward pass derived and implemented by hand**: no PyTorch, no autograd.

Trained on Tiny Shakespeare at the character level, it learns to generate Shakespeare-style text.

## Highlights

- **Hand-derived backpropagation** for every layer: cross-entropy, linear, ReLU, layer norm, softmax, causal multi-head attention, residual connections, and token and positional embeddings
- **Gradient checking**: every parameter's analytic gradient is verified against central-difference estimates
- **Optimisers written from scratch**: SGD and Adam (with bias correction)
- **Complete pipeline**: character-level tokenizer, mini-batch training with validation loss, checkpointing, and temperature-controlled text generation

## Architecture

A GPT-1-style decoder-only transformer:

```
token IDs
  → token embedding + learned positional embedding
  → N × [ causal multi-head self-attention → add & LayerNorm
          → feed-forward (Linear → ReLU → Linear) → add & LayerNorm ]   (post-norm)
  → LM head (Linear) → logits over the vocabulary
```

Attention is scaled dot-product attention with a causal mask:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

## Project structure

```
numpy-gpt/
├── src/
│   ├── gpt.py              # model: every layer's forward and backward pass
│   ├── data.py             # character tokenizer, train/val split, random batches
│   ├── optim.py            # SGD and Adam
│   ├── train.py            # training loop, evaluation, checkpointing
│   ├── generate.py         # autoregressive text generation
│   └── test_gradients.py   # central-difference gradient check
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+ and NumPy 2.0+ (developed on Python 3.15).

```bash
git clone https://github.com/Tasnim-Uddin/numpy-gpt.git
cd numpy-gpt

# Virtual environment (using uv; python -m venv works too)
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Dataset: Tiny Shakespeare (~1 MB)
mkdir -p data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

All commands run from the project root.

**Check the gradients** (confirms every backward pass is correct):

```bash
python src/test_gradients.py
```

**Train** (saves the weights to `checkpoints/`):

```bash
python src/train.py
```

**Generate text** from the trained model:

```bash
python src/generate.py
```

Settings such as model size, learning rate, prompt and temperature are constants at the top of `train.py` and `generate.py`.

## Default configuration

| setting | value |
|---|---|
| vocabulary | 65 characters |
| context length | 64 |
| model dimension | 64 |
| attention heads | 4 |
| transformer blocks | 4 |
| parameters | 212,417 |
| batch size | 16 |
| optimiser | Adam, learning rate 1e-3 |
| training steps | 5,000 |

## Results

<!-- Fill in from your training run, for example: -->
<!-- Final loss after 5,000 steps: train X.XX, validation X.XX (starting from ln 65 ≈ 4.17). -->

Sample output (temperature 0.8, prompt `ROMEO:`):

```
<!-- paste a sample from python src/generate.py -->
```

## How the backward pass works

Every layer follows the same interface:

- `forward(X)` computes the output and saves whatever the backward pass needs
- `backward(dY)` receives the adjoint of its output ($\partial L / \partial Y$), returns the adjoint of its input, and stores the gradients of its own parameters

`GPT.backward` runs the layers in reverse order (reverse-mode automatic differentiation, done by hand). A few of the derived results:

| layer | input adjoint | parameter adjoints |
|---|---|---|
| cross-entropy | $\bar{Z} = \frac{1}{N}(P - Y)$ | — |
| linear, $Y = XW + b$ | $\bar{X} = \bar{Y} W^\top$ | $\bar{W} = X^\top \bar{Y}$, $\bar{b} = \bar{Y}^\top \mathbf{1}$ |
| softmax (per row) | $\bar{z} = p \odot (\bar{p} - (p^\top \bar{p})\,\mathbf{1})$ | — |
| residual, $S = X + F(X)$ | $\bar{X} = \bar{S} + F\text{.backward}(\bar{S})$ | — |

## Possible next steps

- Pre-norm blocks with a final LayerNorm (GPT-2 style)
- RMSNorm, RoPE and SwiGLU (modern LLM components)
- Byte-pair encoding (BPE) tokenizer
- AdamW (decoupled weight decay)
- A PyTorch version, also used to cross-check these gradients

## References

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- Radford et al. (2018), [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
- Ba, Kiros & Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450)
- Kingma & Ba (2014), [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)
- Tiny Shakespeare dataset from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
- Inspired by the video *How I Built GPT from Scratch in Numpy* ([YouTube](https://www.youtube.com/watch?v=3LWHYlmyqZ0))