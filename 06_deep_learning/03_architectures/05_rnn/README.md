<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Recurrent%20Neural%20Networks%20RNN&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/rnn-architecture.svg" width="100%">

*Caption: RNNs process sequences by maintaining a hidden state hₜ that is updated at each time step. The same weights are shared across all time steps. LSTM and GRU add gating mechanisms to handle long-range dependencies.*

---

## 📂 Overview

RNNs are designed for sequential data (text, time series, audio) where the order matters. They maintain a hidden state that acts as "memory" of previous inputs.

---

## 📐 Mathematical Foundations

### 1. Vanilla RNN Equations

**Forward Pass:**

```math
h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)
y_t = W_{hy} h_t + b_y
```

Where:
- $x\_t \in \mathbb{R}^d$ = input at time $t$
- $h\_t \in \mathbb{R}^n$ = hidden state at time $t$
- $y\_t \in \mathbb{R}^k$ = output at time $t$
- $W\_{xh} \in \mathbb{R}^{n \times d}$ = input-to-hidden weights
- $W\_{hh} \in \mathbb{R}^{n \times n}$ = hidden-to-hidden weights
- $W\_{hy} \in \mathbb{R}^{k \times n}$ = hidden-to-output weights

### 2. Backpropagation Through Time (BPTT)

**Loss:**

```math
L = \sum_{t=1}^{T} L_t(y_t, \hat{y}_t)
```

**Gradient w.r.t. $W\_{hh}$:**

```math
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial L_t}{\partial h_t} \cdot \frac{\partial h_t}{\partial h_k} \cdot \frac{\partial h_k}{\partial W_{hh}}
```

**Chain rule for hidden states:**

```math
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^{t} W_{hh}^\top \cdot \text{diag}(\tanh'(z_i))
```

### 3. Vanishing/Exploding Gradient Problem

**Theorem:** For vanilla RNN, gradients satisfy:

```math
\left\| \frac{\partial h_t}{\partial h_k} \right\| \leq (\sigma_{\max}(W_{hh}))^{t-k} \cdot \gamma^{t-k}
```

Where $\gamma = \max\_i |\tanh'(z\_i)| \leq 1$ and $\sigma\_{\max}$ is the largest singular value.

**Proof:**
```
∂h_t/∂h_{t-1} = W_hh^T · diag(tanh'(z_t))

Taking norms:
‖∂h_t/∂h_{t-1}‖ ≤ ‖W_hh‖ · ‖diag(tanh')‖ ≤ σ_max(W_hh) · 1

Over t-k steps:
‖∂h_t/∂h_k‖ ≤ (σ_max(W_hh))^(t-k)

If σ_max < 1: gradients vanish exponentially
If σ_max > 1: gradients explode exponentially
```

---

## 🔬 LSTM: Long Short-Term Memory

### Complete LSTM Equations

**Forget Gate** (what to discard):

```math
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
```

**Input Gate** (what to add):

```math
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
```

**Candidate Cell State:**

```math
\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)
```

**Cell State Update:**

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

**Output Gate:**

```math
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
```

**Hidden State:**

```math
h_t = o_t \odot \tanh(C_t)
```

Where $\odot$ denotes element-wise multiplication.

### Why LSTM Solves Vanishing Gradients

**Key Insight:** Cell state provides a "gradient highway":

```math
\frac{\partial C_t}{\partial C_{t-1}} = f_t
```

**Proof that gradients can flow:**
```
∂C_t/∂C_k = ∏_{i=k+1}^{t} f_i

If forget gates f_i ≈ 1:
  ∂C_t/∂C_k ≈ 1  (gradients preserved!)

Compare to vanilla RNN:
  ∂h_t/∂h_k = ∏(W_hh · tanh') → vanishes or explodes

LSTM cell state has additive updates, not multiplicative!
```

### Parameter Count

For LSTM with input size $d$ and hidden size $n$:

```math
\text{Parameters} = 4 \times (n \times (d + n) + n) = 4n(d + n + 1)
```

The factor of 4 comes from: forget, input, candidate, output gates.

---

## 🔬 GRU: Gated Recurrent Unit

### GRU Equations (Simplified LSTM)

**Reset Gate:**

```math
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r)
```

**Update Gate:**

```math
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z)
```

**Candidate Hidden:**

```math
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h)
```

**Hidden State Update:**

```math
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
```

### GRU vs LSTM Comparison

| Aspect | LSTM | GRU |
|--------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| States | 2 (cell + hidden) | 1 (hidden only) |
| Parameters | $4n(d+n+1)$ | $3n(d+n+1)$ |
| Performance | Slightly better for long sequences | Faster, competitive |

---

## 📊 Theoretical Analysis

### Gradient Clipping for Exploding Gradients

**Algorithm:**

```math
\hat{g} = \begin{cases}
g & \text{if } \|g\| \leq \tau \\
\frac{\tau g}{\|g\|} & \text{if } \|g\| > \tau
\end{cases}
```

Where $\tau$ is the threshold (typically 1.0 or 5.0).

### Orthogonal Initialization

**Goal:** Keep $\|W\_{hh}\|\_2 = 1$ to prevent vanishing/exploding at initialization.

**Method:** Initialize $W\_{hh}$ as orthogonal matrix:

```math
W_{hh}^\top W_{hh} = I \implies \sigma_{\max}(W_{hh}) = 1
```

### Hidden State Dynamics

**Theorem:** For sufficiently long sequences, vanilla RNN hidden states converge to:

```math
\lim_{t \to \infty} h_t = (I - W_{hh})^{-1} W_{xh} x_\infty
```

If $\|W\_{hh}\| < 1$ (contractive mapping).

**Implication:** RNN "forgets" early inputs → need LSTM/GRU for long-range dependencies.

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import math

class VanillaRNNCell(nn.Module):
    """Manual RNN cell implementation for understanding"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Weights
        self.W_xh = nn.Linear(input_size, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Orthogonal initialization for W_hh
        nn.init.orthogonal_(self.W_hh.weight)
    
    def forward(self, x, h_prev):
        # h_t = tanh(W_xh · x_t + W_hh · h_{t-1})
        h_next = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        return h_next

class LSTMCell(nn.Module):
    """Manual LSTM cell implementation"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Combined weights for efficiency: 4 gates at once
        self.W_x = nn.Linear(input_size, 4 * hidden_size)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
    
    def forward(self, x, states):
        h_prev, c_prev = states
        
        # Compute all gates at once
        gates = self.W_x(x) + self.W_h(h_prev)
        
        # Split into individual gates
        i, f, g, o = gates.chunk(4, dim=-1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        g = torch.tanh(g)     # Candidate cell
        o = torch.sigmoid(o)  # Output gate
        
        # Update cell and hidden states
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class SequenceModel(nn.Module):
    """LSTM for sequence modeling"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
            bidirectional=True
        )
        # 2x hidden for bidirectional
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embed = self.embedding(x)  # (batch, seq_len, embed_size)
        
        # LSTM
        output, (h_n, c_n) = self.lstm(embed)
        # output: (batch, seq_len, hidden*2)
        # h_n: (num_layers*2, batch, hidden)
        
        # Use final hidden state (concat forward and backward)
        h_forward = h_n[-2]  # Last layer, forward
        h_backward = h_n[-1]  # Last layer, backward
        h_final = torch.cat([h_forward, h_backward], dim=-1)
        
        return self.fc(h_final)

# Gradient clipping example
def train_step(model, optimizer, x, y, clip_value=1.0):
    optimizer.zero_grad()
    output = model(x)
    loss = nn.CrossEntropyLoss()(output, y)
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
    optimizer.step()
    return loss.item()

# Usage
model = SequenceModel(
    vocab_size=10000,
    embed_size=256,
    hidden_size=512,
    num_layers=2,
    num_classes=10
)

# Check parameters
lstm_params = sum(p.numel() for p in model.lstm.parameters())
print(f"LSTM parameters: {lstm_params:,}")
# For 2-layer bidirectional: 2 * 2 * 4 * hidden * (input + hidden + 1)
```

---

## 🔗 Where This Topic Is Used

| Application | RNN Type |
|-------------|---------|
| **Language Modeling** | LSTM (before Transformers) |
| **Speech Recognition** | CTC with LSTM |
| **Time Series** | Sequence prediction |
| **Machine Translation** | Seq2Seq (before Attention) |
| **Music Generation** | LSTM for melodies |
| **Handwriting Recognition** | Bidirectional LSTM |

---

## 📊 RNN vs Transformers

| Aspect | RNN/LSTM | Transformer |
|--------|----------|-------------|
| Parallelization | Sequential (slow) | Parallel (fast) |
| Long-range | LSTM helps but still struggles | Attention excels |
| Memory | O(1) per step | O(n²) attention matrix |
| Inductive bias | Strong temporal | Positional encoding |
| Training | BPTT, gradient issues | Standard backprop |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | LSTM Paper | [Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) |
| 📄 | GRU Paper | [arXiv](https://arxiv.org/abs/1406.1078) |
| 📄 | Vanishing Gradients | [Paper](http://www.cs.toronto.edu/~hinton/absps/vanishing.pdf) |
| 🎥 | Colah's Blog: LSTM | [Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| 🎥 | Karpathy's RNN blog | [Blog](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) |
| 🇨🇳 | LSTM详解 | [知乎](https://zhuanlan.zhihu.com/p/32085405) |
| 🇨🇳 | RNN与LSTM原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88785898) |
| 🇨🇳 | 循环神经网络 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |

---

⬅️ [Back: MoE](../04_moe/README.md) | ➡️ [Next: Transformer](../06_transformer/README.md)

---

⬅️ [Back: Architectures](../../README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
