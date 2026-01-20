<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Long%20Short-Term%20Memory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Overview

Long Short-Term Memory (LSTM) networks solve the vanishing gradient problem in RNNs through a gating mechanism. The key innovation is the **cell state**, which acts as a "conveyor belt" allowing information to flow unchanged across many timesteps.

---

## 📐 LSTM Architecture: Complete Mathematical Formulation

### The Vanishing Gradient Problem (Why LSTMs Exist)

**Standard RNN:**
```
hₜ = tanh(Wₕₕ hₜ₋₁ + Wₓₕ xₜ + b)

Gradient through time:
∂hₜ/∂hₜ₋₁ = diag(1 - tanh²(·)) · Wₕₕ

For T timesteps:
∂hₜ/∂h₀ = Π_{k=1}^{T} ∂hₖ/∂hₖ₋₁

Since |tanh'(x)| ≤ 1 and typically < 1:
||∂hₜ/∂h₀|| → 0 as T → ∞ (vanishing)
or
||∂hₜ/∂h₀|| → ∞ as T → ∞ (exploding)
```

**LSTM Solution:** Introduce gates that control information flow.

---

### LSTM Equations (Complete)

**Input:** $x_t \in \mathbb{R}^d$, previous hidden state $h_{t-1} \in \mathbb{R}^h$, previous cell state $c_{t-1} \in \mathbb{R}^h$

**1. Forget Gate:** What information to discard from cell state

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

```
Intuition: f_t ∈ (0, 1)ʰ controls what fraction of old information to keep
- f_t ≈ 1: Keep everything
- f_t ≈ 0: Forget everything
```

**2. Input Gate:** What new information to store

$$
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)
\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)
$$

```
Intuition:
- i_t ∈ (0, 1)ʰ: How much of new info to write
- c̃_t ∈ (-1, 1)ʰ: New candidate values
```

**3. Cell State Update:** The key to long-term memory

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

```
Intuition: Linear combination of old and new information
- f_t ⊙ c_{t-1}: Retained old information
- i_t ⊙ c̃_t: New information to add

Key insight: This is nearly a linear operation!
Gradient flows easily through c_t → c_{t-1}
```

**4. Output Gate:** What to output from cell state

$$
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)
h_t = o_t \odot \tanh(c_t)
$$

```
Intuition:
- o_t ∈ (0, 1)ʰ: What parts of cell state to expose
- h_t: Filtered version of cell state
```

---

## 📊 Parameter Dimensions

| Parameter | Dimension | Count |
|-----------|-----------|-------|
| $W_f, W_i, W_c, W_o$ | $h \times (h + d)$ | 4 matrices |
| $b_f, b_i, b_c, b_o$ | $h$ | 4 vectors |
| **Total** | | $4h(h+d) + 4h = 4h(h+d+1)$ |

**Example:** For $h = 512$, $d = 256$:
- Total parameters per layer: $4 \times 512 \times (512 + 256 + 1) \approx 1.57M$

---

## 🔬 Gradient Flow Analysis

### Why LSTMs Solve Vanishing Gradients

**Cell State Gradient:**
```
∂cₜ/∂cₜ₋₁ = diag(fₜ) + terms involving ∂fₜ/∂cₜ₋₁

If we ignore the second term (small in practice):
∂cₜ/∂cₜ₋₁ ≈ diag(fₜ)

For T timesteps:
∂cₜ/∂c₀ ≈ Π_{k=1}^{T} diag(fₖ)

Key insight: If fₖ ≈ 1, gradient flows unchanged!
Unlike tanh which always shrinks gradients.
```

**Formal Proof:**
```
∂L/∂c₀ = ∂L/∂cₜ · ∂cₜ/∂cₜ₋₁ · ... · ∂c₁/∂c₀

For each step:
∂cₖ/∂cₖ₋₁ = fₖ + (stuff)

If forget gates are close to 1:
∂L/∂c₀ ≈ ∂L/∂cₜ · (1 × 1 × ... × 1) = ∂L/∂cₜ

The gradient reaches c₀ almost unchanged!
```

### Constant Error Carousel

The cell state update can be written as:
```
cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

If fₜ = 1 and iₜ = 0:
cₜ = cₜ₋₁ (information preserved perfectly)

This is the "constant error carousel" - 
error signals can flow through time without decay.
```

---

## 📐 LSTM Variants

### 1. Peephole Connections

Add cell state to gate computations:

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + W_{cf} \cdot c_{t-1} + b_f)
i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + W_{ci} \cdot c_{t-1} + b_i)
o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + W_{co} \cdot c_t + b_o)
$$

**Benefit:** Gates can observe cell state directly.

### 2. GRU (Gated Recurrent Unit)

Simplified version with 2 gates instead of 3:

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t])
$$ (update gate)
$$

r_t = \sigma(W_r \cdot [h_{t-1}, x_t])
$$ (reset gate)

$$
\tilde{h}_t = \tanh(W \cdot [r_t \odot h_{t-1}, x_t])
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$

**Key difference:** No separate cell state; fewer parameters.

### 3. Coupled Forget-Input Gates

Force $i_t = 1 - f_t$:

$$
c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t
$$

**Benefit:** Reduces parameters, enforces conservation of information.

---

## 🎯 Backpropagation Through Time (BPTT) for LSTM

### Forward Pass Summary

```python
for t in range(T):

    # Concatenate inputs
    z_t = concat(h_{t-1}, x_t)  # Shape: (h + d)
    
    # Gates (all use sigmoid)
    f_t = sigmoid(W_f @ z_t + b_f)
    i_t = sigmoid(W_i @ z_t + b_i)
    o_t = sigmoid(W_o @ z_t + b_o)
    
    # Candidate cell state
    c_tilde = tanh(W_c @ z_t + b_c)
    
    # Cell state update
    c_t = f_t * c_{t-1} + i_t * c_tilde
    
    # Hidden state
    h_t = o_t * tanh(c_t)
```

### Backward Pass (Gradient Derivation)

```
Given: ∂L/∂hₜ, ∂L/∂cₜ (from next timestep or output layer)

Step 1: Output gate gradients
∂L/∂oₜ = ∂L/∂hₜ ⊙ tanh(cₜ)
∂L/∂(Wₒzₜ) = ∂L/∂oₜ ⊙ σ'(Wₒzₜ) = ∂L/∂oₜ ⊙ oₜ ⊙ (1 - oₜ)

Step 2: Cell state gradients
∂L/∂cₜ += ∂L/∂hₜ ⊙ oₜ ⊙ (1 - tanh²(cₜ))

Step 3: Forget gate gradients
∂L/∂fₜ = ∂L/∂cₜ ⊙ cₜ₋₁
∂L/∂(Wfzₜ) = ∂L/∂fₜ ⊙ fₜ ⊙ (1 - fₜ)

Step 4: Input gate gradients
∂L/∂iₜ = ∂L/∂cₜ ⊙ c̃ₜ
∂L/∂c̃ₜ = ∂L/∂cₜ ⊙ iₜ

Step 5: Propagate to previous timestep
∂L/∂cₜ₋₁ = ∂L/∂cₜ ⊙ fₜ  ← KEY: multiplied by fₜ ≈ 1
∂L/∂hₜ₋₁ = W'∂L/∂(gates)  ← Standard backprop through weights
```

---

## 💻 Complete Implementation

### NumPy Implementation (From Scratch)

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(s):
    return s * (1 - s)

def tanh_derivative(t):
    return 1 - t ** 2

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        
        # Combined weights for efficiency: [W_f, W_i, W_c, W_o]
        self.W = np.random.randn(4 * hidden_dim, input_dim + hidden_dim) * scale
        self.b = np.zeros((4 * hidden_dim,))
        
        # For gradient computation
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        """
        Forward pass for single timestep
        
        Args:
            x: (batch, input_dim)
            h_prev: (batch, hidden_dim)
            c_prev: (batch, hidden_dim)
        
        Returns:
            h_next: (batch, hidden_dim)
            c_next: (batch, hidden_dim)
        """
        H = self.hidden_dim
        
        # Concatenate input and previous hidden state
        z = np.concatenate([h_prev, x], axis=1)  # (batch, H + D)
        
        # Compute all gates at once
        gates = z @ self.W.T + self.b  # (batch, 4H)
        
        # Split into individual gates
        f = sigmoid(gates[:, 0:H])           # Forget gate
        i = sigmoid(gates[:, H:2*H])         # Input gate
        c_tilde = np.tanh(gates[:, 2*H:3*H]) # Candidate
        o = sigmoid(gates[:, 3*H:4*H])       # Output gate
        
        # Cell state update
        c_next = f * c_prev + i * c_tilde
        
        # Hidden state
        h_next = o * np.tanh(c_next)
        
        # Cache for backward pass
        self.cache = (x, h_prev, c_prev, z, f, i, c_tilde, o, c_next, h_next)
        
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        """
        Backward pass for single timestep
        
        Args:
            dh_next: gradient of loss w.r.t. h_next
            dc_next: gradient of loss w.r.t. c_next
        
        Returns:
            dh_prev, dc_prev, dW, db
        """
        x, h_prev, c_prev, z, f, i, c_tilde, o, c_next, h_next = self.cache
        H = self.hidden_dim
        
        # Gradient of hidden state
        tanh_c = np.tanh(c_next)
        
        # Output gate gradient
        do = dh_next * tanh_c
        do_input = do * sigmoid_derivative(o)
        
        # Cell state gradient (accumulate from h and direct c gradient)
        dc = dc_next + dh_next * o * tanh_derivative(tanh_c)
        
        # Forget gate gradient
        df = dc * c_prev
        df_input = df * sigmoid_derivative(f)
        
        # Input gate gradient
        di = dc * c_tilde
        di_input = di * sigmoid_derivative(i)
        
        # Candidate gradient
        dc_tilde = dc * i
        dc_tilde_input = dc_tilde * tanh_derivative(c_tilde)
        
        # Concatenate gate gradients
        dgates = np.concatenate([df_input, di_input, dc_tilde_input, do_input], axis=1)
        
        # Weight and bias gradients
        dW = dgates.T @ z
        db = np.sum(dgates, axis=0)
        
        # Input gradients
        dz = dgates @ self.W
        dh_prev = dz[:, :H]
        dx = dz[:, H:]
        
        # Cell state gradient to previous timestep
        dc_prev = dc * f
        
        return dh_prev, dc_prev, dW, db, dx

class LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.cell = LSTMCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        
        # Output projection
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b_out = np.zeros((output_dim,))
    
    def forward(self, X):
        """
        Forward pass for sequence
        
        Args:
            X: (batch, seq_len, input_dim)
        
        Returns:
            outputs: (batch, seq_len, output_dim)
        """
        batch_size, seq_len, _ = X.shape
        
        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))
        
        self.hidden_states = []
        self.cell_states = []
        outputs = []
        
        for t in range(seq_len):
            h, c = self.cell.forward(X[:, t, :], h, c)
            self.hidden_states.append(h)
            self.cell_states.append(c)
            
            # Output projection
            out = h @ self.W_out.T + self.b_out
            outputs.append(out)
        
        return np.stack(outputs, axis=1)

# Example usage
batch_size = 32
seq_len = 10
input_dim = 50
hidden_dim = 128
output_dim = 10

lstm = LSTM(input_dim, hidden_dim, output_dim)
X = np.random.randn(batch_size, seq_len, input_dim)
outputs = lstm.forward(X)
print(f"Output shape: {outputs.shape}")  # (32, 10, 10)
```

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class LSTMFromScratch(nn.Module):
    """
    LSTM implementation from scratch in PyTorch
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Forget gate
        self.W_f = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Input gate
        self.W_i = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Candidate
        self.W_c = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # Output gate
        self.W_o = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Initialize forget gate bias to 1 (helps gradient flow)
        nn.init.ones_(self.W_f.bias)
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            hidden: tuple of (h_0, c_0)
        
        Returns:
            output: (batch, seq_len, hidden_dim)
            (h_n, c_n): final hidden states
        """
        batch_size, seq_len, _ = x.shape
        
        if hidden is None:
            h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h, c = hidden
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            combined = torch.cat([h, x_t], dim=1)
            
            # Gates
            f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
            i_t = torch.sigmoid(self.W_i(combined))  # Input gate
            c_tilde = torch.tanh(self.W_c(combined)) # Candidate
            o_t = torch.sigmoid(self.W_o(combined))  # Output gate
            
            # Cell state update
            c = f_t * c + i_t * c_tilde
            
            # Hidden state update
            h = o_t * torch.tanh(c)
            
            outputs.append(h)
        
        return torch.stack(outputs, dim=1), (h, c)

class BiLSTM(nn.Module):
    """
    Bidirectional LSTM
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            lengths: actual sequence lengths for packing
        
        Returns:
            output: (batch, seq_len, 2 * hidden_dim)
        """
        if lengths is not None:

            # Pack padded sequence
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output, (h_n, c_n) = self.lstm(x_packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        else:
            output, (h_n, c_n) = self.lstm(x)
        
        return output, (h_n, c_n)

# Language Model Example
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights (optional, but common)
        if embed_dim == hidden_dim:
            self.fc.weight = self.embedding.weight
    
    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, seq_len) token indices
            hidden: initial hidden state
        
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        embed = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embed, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

# Test
model = LSTMFromScratch(input_dim=256, hidden_dim=512)
x = torch.randn(32, 20, 256)  # (batch, seq, features)
output, (h_n, c_n) = model(x)
print(f"Output: {output.shape}, h_n: {h_n.shape}, c_n: {c_n.shape}")
```

---

## 🔗 Comparison with Other Architectures

| Feature | RNN | LSTM | GRU | Transformer |
|---------|-----|------|-----|-------------|
| **Parameters** | $h^2$ | $4h^2$ | $3h^2$ | $4h^2 + 2h^2/n_{heads}$ |
| **Vanishing gradient** | Severe | Solved | Solved | N/A |
| **Parallelization** | Sequential | Sequential | Sequential | Fully parallel |
| **Long-range** | Poor | Good | Good | Excellent |
| **Memory** | $O(h)$ | $O(h)$ | $O(h)$ | $O(n^2)$ |

---

## 📚 Key Insights

| Insight | Explanation |
|---------|-------------|
| **Forget gate bias = 1** | Initialize $b_f = 1$ so $f_t \approx 1$ initially, allowing gradient flow |
| **Cell state is nearly linear** | $c_t = f_t \odot c_{t-1} + ...$ allows error to flow unchanged |
| **Peephole not always better** | Original LSTM often works just as well |
| **GRU is often sufficient** | Fewer parameters, similar performance |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Original LSTM Paper | Hochreiter & Schmidhuber, 1997 |
| 📄 | LSTM: A Search Space Odyssey | Greff et al., 2015 |
| 🎥 | Colah's Blog: Understanding LSTM | [colah.github.io](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| 📄 | GRU Paper | Cho et al., 2014 |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [RNN](../README.md) | [Architectures](../../README.md) | [Seq2Seq](../../10_seq2seq/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
