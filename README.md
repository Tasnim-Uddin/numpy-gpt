# numpy-gpt

A GPT (decoder-only transformer) built from scratch in NumPy, with **every forward and backward pass derived and implemented by hand**: no PyTorch, no autograd.

Trained on Tiny Shakespeare at the character level, it learns to generate Shakespeare-style text.

## Sample output

After training (212,417 parameters, 5,000 steps, about 5 minutes on a laptop CPU), prompted with `ROMEO:` at temperature 0.8:

```
ROMEO:
Shall on wife this hands husband the treambles
Therefore that I forth sucked your lordsharit,
at you'ld gone that short her first of pintes,
To holy take for the lord! what by thou gentleman my
wifether contage if all this as man?

JULIET:
Do my muntery, holest the man
To stuorshiph the man lights to and now close
That is the have I to follow,
But, and her from a new lieve--and do told.
Comfort it we holf purtivious mark, pray prison broother,
Blour of inframent anle of the fear of it.
```

Starting from random weights, the model has learned the structure of a play (speaker names in capitals followed by a colon, then lines of verse), real character names, mostly real English words, and Shakespearean forms like *thou*. It still invents words and has no meaning across sentences: it builds text one character at a time and only sees the last 64 characters, as expected for a model this small.

### Training

| step | train loss | validation loss |
|---|---|---|
| 0 | 4.19 | 4.19 |
| 1,000 | 2.00 | 2.04 |
| 2,500 | 1.72 | 1.88 |
| 5,000 | 1.56 | 1.75 |

The loss starts at ln 65 ≈ 4.17 (random guessing among 65 characters) and ends with a validation loss of 1.75, a perplexity of e^1.75 ≈ 5.7: on average the model is as uncertain as choosing between about 6 characters instead of 65. The widening gap between training and validation loss shows the first signs of overfitting.

## Highlights

- **Hand-derived backpropagation** for every layer: cross-entropy, linear, ReLU, layer norm, softmax, causal multi-head attention, residual connections, and token and positional embeddings
- **Gradient checking**: every parameter's analytic gradient is verified against central-difference estimates
- **Optimisers written from scratch**: SGD and Adam (with bias correction)
- **Complete pipeline**: character-level tokenizer, mini-batch training with validation loss, checkpointing, and temperature-controlled text generation
- **Documentation**: the architecture, every derivation, and the numerical details, explained in [`docs/`](docs/README.md)

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

More detail, with diagrams and the shape at every step: [docs/forward-pass.md](docs/forward-pass.md).

## Project structure

```
numpy-gpt/
├── docs/                   # explanations and derivations (start with docs/README.md)
├── src/
│   ├── gpt.py              # model: every layer's forward and backward pass
│   ├── data.py             # character tokenizer, train/val split, random batches
│   ├── optim.py            # SGD and Adam
│   ├── train.py            # training loop, evaluation, checkpointing
│   ├── generate.py         # autoregressive text generation
│   └── test_gradients.py   # central-difference gradient check
├── data/                   # Tiny Shakespeare (downloaded; not in the repo)
├── checkpoints/            # trained weights (created by train.py; not in the repo)
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

**Train** (saves the weights to `checkpoints/`; takes about 5 minutes on a laptop CPU):

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

## Documentation

Detailed explanations live in [`docs/`](docs/README.md):

| document | covers |
|---|---|
| [Forward pass](docs/forward-pass.md) | architecture, data flow and shapes, with diagrams |
| [Backward pass](docs/backward-pass.md) | every layer's gradients, derived by hand |
| [Numerical stability](docs/numerical-stability.md) | avoiding overflow, `log(0)` and division by zero |
| [Gradient checking](docs/gradient-checking.md) | verifying gradients with central differences |
| [Optimisers](docs/optimisers.md) | mini-batch SGD and Adam |
| [Data and generation](docs/data-and-generation.md) | tokenizer, batches, training loop, sampling and temperature |

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

Full derivations for every layer: [docs/backward-pass.md](docs/backward-pass.md).

## Possible next steps

- Pre-norm blocks with a final LayerNorm (GPT-2 style)
- RMSNorm, RoPE and SwiGLU (modern LLM components)
- Dropout and AdamW (decoupled weight decay) to reduce overfitting
- A learning-rate schedule (warmup and cosine decay)
- Z-loss (an auxiliary penalty on log Z) and QK-norm, for training stability
- Byte-pair encoding (BPE) tokenizer
- A PyTorch version, also used to cross-check these gradients

## References

- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
- Radford et al. (2018), [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (GPT-1)
- Ba, Kiros & Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450)
- Kingma & Ba (2014), [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980)
- Tiny Shakespeare dataset from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn)
- Inspired by the video *How I Built GPT from Scratch in Numpy* ([YouTube](https://www.youtube.com/watch?v=3LWHYlmyqZ0))