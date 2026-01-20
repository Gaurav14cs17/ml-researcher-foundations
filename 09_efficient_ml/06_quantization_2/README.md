<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=120&section=header&text=Lecture%206%20Quantization%20Part%20II&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-09-1ABC9C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

# Lecture 6: Quantization (Part II)

[← Back to Course](../) | [← Previous](../05_quantization_1/) | [Next: NAS I →](../07_neural_architecture_search_1/)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09_efficient_ml/06_quantization_2/demo.ipynb) ← **Try the code!**

---

## 📺 Video Lecture

| Resource | Link |
|----------|------|
| 🎥 **Lecture Video** | [Watch on YouTube](https://www.youtube.com/watch?v=xYDadmguObs&list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=6) |
| 📊 **Slides** | [MIT Course Page](https://hanlab.mit.edu/courses/2023-fall-65940) |
| ⏱️ **Duration** | ~90 minutes |

### 📝 Video Summary

This lecture covers **advanced quantization techniques**, especially for LLMs:

- **Quantization-Aware Training (QAT)**: Training with simulated quantization
- **Straight-Through Estimator (STE)**: Backpropagating through non-differentiable rounding
- **LLM quantization challenges**: Activation outliers that break standard methods
- **SmoothQuant**: Migrating quantization difficulty from activations to weights
- **GPTQ**: Second-order weight quantization for one-shot LLM compression
- **AWQ**: Activation-aware weight quantization preserving important weights

> 💡 *"QLoRA enables training a 65B parameter model on a single GPU—a game-changer for accessibility."* — Prof. Song Han

---

![Overview](overview.png)

## Quantization-Aware Training (QAT)

Train with simulated quantization to improve accuracy:

```python
def fake_quantize(x, scale, zero_point, bits=8):
    """Simulate quantization during training"""
    q_min, q_max = 0, 2**bits - 1
    
    # Forward: quantize
    x_q = torch.clamp(
        torch.round(x / scale + zero_point),
        q_min, q_max
    )
    
    # "Dequantize" back to float for gradient flow
    x_deq = (x_q - zero_point) * scale
    
    return x_deq  # Gradients flow through as if no quantization
```

---

## Straight-Through Estimator (STE)

Rounding is not differentiable. STE approximates gradients:

```
Forward:  y = round(x)
Backward: ∂L/∂x = ∂L/∂y  (pretend round = identity)
```

---

## LLM Quantization Challenges

LLMs have **activation outliers** that break standard quantization:

```
Normal activations: [-1, 0.5, 0.8, 1.2, -0.3]
LLM activations:    [-1, 0.5, 50.0, 1.2, -0.3]  # Outlier!
```

### Solutions

1. **LLM.int8()** - Mixed precision for outlier channels
2. **SmoothQuant** - Migrate difficulty from activations to weights
3. **GPTQ** - Second-order weight quantization
4. **AWQ** - Activation-aware weight quantization

---

## SmoothQuant

Key insight: **Weights are easy to quantize, activations are hard.**

Solution: Balance the difficulty:

```python
# Before: hard activations, easy weights
X @ W

# After SmoothQuant: balanced
(X / s) @ (W * s)

# s is chosen to equalize difficulty
s = sqrt(max(|X|) / max(|W|))
```

---

## GPTQ (4-bit LLM Quantization)

Quantize one column at a time, fixing errors in remaining weights:

```
For each column j:
    1. Quantize column j
    2. Compute quantization error
    3. Distribute error to columns j+1, j+2, ... 
       (using Hessian information)
```

**Result:** GPT-3 175B → 4-bit with minimal accuracy loss!

---

## 📐 Mathematical Foundations & Proofs

### Straight-Through Estimator (STE)

**Problem:** The round function has zero gradient almost everywhere:

```math
\frac{d \text{round}(x)}{dx} = 0 \text{ (a.e.)}
```

**Solution:** STE approximates the gradient as identity:

```math
\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial \text{round}(x)}
```

**Formal justification:**

Define the quantization function with STE:

```math
y = x + \text{sg}(\text{round}(x) - x)
```

where \( \text{sg} \) is stop-gradient operator.

Forward: \( y = \text{round}(x) \)
Backward: \( \frac{\partial y}{\partial x} = 1 \)

**Why it works:** In expectation, the gradient direction is preserved even though magnitude is approximated.

---

### Quantization-Aware Training Loss

QAT minimizes:

```math
\mathcal{L}_{QAT} = \mathcal{L}_{task}(Q(W), X, Y) + \lambda \mathcal{L}_{reg}
```

where \( Q(W) \) denotes quantized weights.

The network learns to produce weights that are quantization-friendly:

```math
W^* = \arg\min_W \mathcal{L}_{task}(Q(W), X, Y)
```

---

### SmoothQuant Mathematical Formulation

**Original computation:**

```math
Y = XW
```

**Smoothed computation:**

```math
Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W) = \hat{X} \hat{W}
```

**Optimal smoothing factor:**

```math
s_j = \frac{\max_i |X_{ij}|^\alpha}{\max_k |W_{jk}|^{1-\alpha}}
```

where \( \alpha \in [0, 1] \) controls the migration strength.

**Proof of equivalence:**

```math
\hat{X}\hat{W} = X \cdot \text{diag}(s)^{-1} \cdot \text{diag}(s) \cdot W = X \cdot I \cdot W = XW
```

**Effect on quantization difficulty:**

Before smoothing:
- Activation range: \( [0, 100] \) (hard to quantize, outliers)
- Weight range: \( [-1, 1] \) (easy)

After smoothing with \( s = 10 \):
- Activation range: \( [0, 10] \) (easier)
- Weight range: \( [-10, 10] \) (still manageable)

Both are now within similar ranges → balanced quantization.

---

### GPTQ: Optimal Brain Quantization

**Problem:** Quantize weight matrix \( W \) to minimize output error:

```math
\arg\min_{\hat{W}} \|WX - \hat{W}X\|_F^2
```

**Hessian approximation:**

```math
H = 2X X^T
```

**Per-column quantization with error compensation:**

For column \( j \):
1. Quantize: \( \hat{w}_j = Q(w_j) \)
2. Error: \( \delta_j = w_j - \hat{w}_j \)
3. Compensate remaining columns:

```math
w_{j+1:} \leftarrow w_{j+1:} - \delta_j \cdot \frac{H^{-1}_{j+1:,j}}{H^{-1}_{jj}}
```

**Derivation of update rule:**

The optimal update minimizes:

```math
\min_{\delta_{j+1:}} \|(\delta_j e_j + \delta_{j+1:}^T) X\|_F^2
```

Taking the gradient and setting to zero:

```math
\delta_{j+1:} = -\delta_j \cdot (X_{j+1:} X_{j+1:}^T)^{-1} X_{j+1:} X_j^T
```

Using block matrix inversion:

```math
\delta_{j+1:} = -\delta_j \cdot \frac{H^{-1}_{j+1:,j}}{H^{-1}_{jj}}
```

---

### AWQ: Activation-Aware Weight Quantization

**Key observation:** Not all weights are equally important. Weights that process large activations matter more.

**Importance metric:**

```math
\text{Importance}(w_j) = \mathbb{E}[|X_j|] \cdot |w_j|
```

**Salient weight protection:** Scale important weights before quantization:

```math
\hat{W}_j = s_j \cdot W_j, \quad \hat{X}_j = X_j / s_j
```

For important channels, use larger \( s_j \) → more quantization levels for important weights.

**Optimal scaling:**

```math
s_j^* = \arg\min_{s_j} \|Q(s_j W_j) - s_j W_j\|_2
```

---

### Mixed-Precision Quantization

**Problem:** Find optimal bit-width per layer:

```math
\min_{b_1, ..., b_L} \mathcal{L}(Q_{b_1}(W_1), ..., Q_{b_L}(W_L))
\text{s.t.} \sum_l \text{Size}(Q_{b_l}(W_l)) \leq B
```

**Sensitivity-based allocation:**

Layer sensitivity:

```math
S_l = \frac{\partial \mathcal{L}}{\partial b_l} \approx \mathcal{L}(b_l = 4) - \mathcal{L}(b_l = 8)
```

Allocate more bits to sensitive layers:

```math
b_l = 8 \text{ if } S_l > \tau \text{ else } 4
```

---

### QLoRA: Quantized Low-Rank Adaptation

**Core idea:** Combine 4-bit quantization with LoRA:

```math
Y = X \cdot Q_4(W_0) + X \cdot BA
```

where:
- \( Q_4(W_0) \) = 4-bit quantized pretrained weights (frozen)
- \( B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d} \) = trainable LoRA adapters (FP16)

**Memory savings:**

| Component | Size |
|-----------|------|
| Base model (FP16) | 2 × params |
| QLoRA base (4-bit) | 0.5 × params |
| LoRA adapters | 2 × r × d × layers |

For 7B model with r=64: 3.5GB (4-bit) + ~100MB (LoRA) ≈ 3.6GB total!

---

## 🧮 Key Derivations

### Quantization Error Propagation

For network \( f = f_L \circ ... \circ f_1 \):

```math
\|\hat{f}(x) - f(x)\| \leq \sum_{l=1}^L \|\nabla f_{l+1:L}\| \cdot \|\epsilon_l\|
```

where \( \epsilon_l \) is the quantization error at layer \( l \).

**Implication:** Later layers amplify earlier errors → quantize carefully!

---

### Hessian Computation in GPTQ

The Hessian \( H = XX^T \) requires \( O(d^2 \cdot n) \) to compute and \( O(d^2) \) to store.

**Efficient computation:**
```python
H = torch.zeros(d, d)
for batch in calibration_data:
    H += batch.T @ batch  # Running sum
H /= n_samples
```

**Efficient inversion:**
Use Cholesky decomposition: \( H = LL^T \), then solve triangular systems.

---

### LLM.int8() Mixed Precision

**Rule:** Use FP16 for outlier dimensions, INT8 for rest.

```math
Y = X_{outlier} W_{outlier}^{FP16} + X_{normal} W_{normal}^{INT8}
```

Outlier detection: dimension \( j \) is outlier if \( \max_i |X_{ij}| > 6 \)

Typical: 0.1-1% of dimensions are outliers.

---

## 💻 Code Examples

### Quantization-Aware Training (QAT)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FakeQuantize(torch.autograd.Function):
    """
    Fake quantization with straight-through estimator
    """
    @staticmethod
    def forward(ctx, x, scale, zero_point, num_bits=8):
        qmin, qmax = 0, 2**num_bits - 1
        x_q = torch.clamp(torch.round(x / scale) + zero_point, qmin, qmax)
        x_dq = (x_q - zero_point) * scale
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: gradient passes through unchanged
        return grad_output, None, None, None

class QATLinear(nn.Module):
    """Linear layer with quantization-aware training"""
    def __init__(self, in_features, out_features, bits=8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bits = bits
        
        # Learnable quantization parameters
        self.weight_scale = nn.Parameter(torch.ones(1))
        self.weight_zero = nn.Parameter(torch.zeros(1))
        self.act_scale = nn.Parameter(torch.ones(1))
        self.act_zero = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        # Quantize weights
        w_q = FakeQuantize.apply(
            self.linear.weight, self.weight_scale, self.weight_zero, self.bits
        )
        
        # Quantize activations
        x_q = FakeQuantize.apply(x, self.act_scale, self.act_zero, self.bits)
        
        return F.linear(x_q, w_q, self.linear.bias)

# SmoothQuant implementation
def smooth_quant(model, calibration_data, alpha=0.5):
    """
    Apply SmoothQuant to balance activation/weight quantization
    """
    act_scales = {}
    
    # Collect activation statistics
    def hook_fn(name):
        def hook(module, input, output):
            x = input[0]
            act_scales[name] = x.abs().max(dim=0)[0]
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run calibration
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    
    for h in hooks:
        h.remove()
    
    # Apply smoothing
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in act_scales:
            act_scale = act_scales[name]
            weight_scale = module.weight.abs().max(dim=0)[0]
            
            # Compute smoothing factor
            s = (act_scale ** alpha) / (weight_scale ** (1 - alpha))
            s = s.clamp(min=1e-5)
            
            # Apply: divide activations, multiply weights
            module.weight.data *= s.unsqueeze(0)
    
    return model
```

### GPTQ-style Quantization

```python
def gptq_quantize_layer(W, X, bits=4, group_size=128):
    """
    GPTQ: Accurate Post-Training Quantization for LLMs
    
    W: weight matrix (out_features, in_features)
    X: calibration data (n_samples, in_features)
    """
    out_features, in_features = W.shape
    
    # Compute Hessian
    H = X.T @ X / X.shape[0]
    H += 1e-4 * torch.eye(H.shape[0])  # Regularization
    
    # Cholesky for efficient inverse
    H_inv = torch.linalg.inv(H)
    
    W_q = torch.zeros_like(W)
    
    for i in range(0, in_features, group_size):
        j = min(i + group_size, in_features)
        
        # Get group
        W_group = W[:, i:j].clone()
        H_inv_group = H_inv[i:j, i:j]
        
        for col in range(j - i):
            w = W_group[:, col]
            d = H_inv_group[col, col]
            
            # Quantize column
            scale = w.abs().max() / (2**(bits-1) - 1)
            w_q = torch.round(w / scale).clamp(-2**(bits-1), 2**(bits-1)-1)
            W_q[:, i+col] = w_q * scale
            
            # Compute error and compensate
            err = (w - W_q[:, i+col])
            if col + 1 < j - i:
                W_group[:, col+1:] -= err.unsqueeze(1) * H_inv_group[col, col+1:] / d
    
    return W_q

# AWQ: Activation-aware Weight Quantization
def awq_quantize(W, X, bits=4):
    """
    AWQ: Protect salient weights based on activation magnitude
    """
    # Compute activation importance
    act_importance = X.abs().mean(dim=0)
    
    # Scale weights by importance before quantization
    # Salient channels get larger scale → less relative error
    scale = act_importance / act_importance.mean()
    W_scaled = W * scale.unsqueeze(0)
    
    # Quantize
    qmax = 2**(bits-1) - 1
    weight_scale = W_scaled.abs().max(dim=1, keepdim=True)[0] / qmax
    W_q = torch.round(W_scaled / weight_scale).clamp(-qmax-1, qmax)
    
    # Unscale
    W_q = (W_q * weight_scale) / scale.unsqueeze(0)
    
    return W_q
```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| QAT | Mobile CNNs, Edge deployment |
| SmoothQuant | LLM INT8 inference |
| GPTQ/AWQ | LLM INT4 inference |
| QLoRA | LLM fine-tuning on consumer GPUs |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [← Quantization I](../05_quantization_1/README.md) | [Efficient ML](../README.md) | [NAS I →](../07_neural_architecture_search_1/README.md) |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | LLM.int8() | [arXiv](https://arxiv.org/abs/2208.07339) |
| 📄 | SmoothQuant | [arXiv](https://arxiv.org/abs/2211.10438) |
| 📄 | GPTQ | [arXiv](https://arxiv.org/abs/2210.17323) |
| 📄 | AWQ | [arXiv](https://arxiv.org/abs/2306.00978) |
| 📄 | QLoRA | [arXiv](https://arxiv.org/abs/2305.14314) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=1ABC9C&height=80&section=footer" width="100%"/>
</p>
