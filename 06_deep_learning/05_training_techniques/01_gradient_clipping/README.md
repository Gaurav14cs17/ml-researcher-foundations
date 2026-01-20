<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Gradient%20Clipping&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### The Exploding Gradient Problem

For RNNs/deep networks, gradients can grow exponentially:

```math
\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{t=1}^{T} \frac{\partial h_t}{\partial h_{t-1}} \frac{\partial h_1}{\partial \theta}

```

If $\left\|\frac{\partial h_t}{\partial h_{t-1}}\right\| > 1$, gradient grows as $O(c^T)$.

**Symptoms:**
- Loss becomes NaN

- Weights diverge

- Training becomes unstable

---

## 📐 Clipping Methods

### 1. Clip by Value (Element-wise)

```math
\tilde{g}_i = \text{clip}(g_i, -\tau, \tau) = \begin{cases}
\tau & \text{if } g_i > \tau \\
-\tau & \text{if } g_i < -\tau \\
g_i & \text{otherwise}
\end{cases}

```

**Properties:**
- Simple and fast

- Changes gradient direction

- Can distort optimization

### 2. Clip by Norm (Global) ⭐ Most Common

```math
\tilde{g} = \begin{cases}
g & \text{if } \|g\| \leq \tau \\
\frac{\tau g}{\|g\|} & \text{if } \|g\| > \tau
\end{cases}

```

Equivalently:

```math
\tilde{g} = \min\left(1, \frac{\tau}{\|g\|}\right) g

```

**Properties:**
- Preserves gradient direction

- Only scales magnitude

- Preferred for most applications

### 3. Clip by Norm (Per-layer)

Apply clip by norm to each layer's gradient separately:

```math
\tilde{g}_l = \min\left(1, \frac{\tau}{\|g_l\|}\right) g_l

```

**Use case:** When layers have very different gradient scales.

---

## 📐 Theoretical Analysis

### Gradient Norm Bound

With clipping threshold $\tau$:

```math
\|\tilde{g}\| \leq \tau

```

### Effect on Learning Rate

Effective learning rate with clipping:

```math
\eta_{eff} = \eta \cdot \min\left(1, \frac{\tau}{\|g\|}\right)

```

When $\|g\| > \tau$: learning rate is adaptively reduced.

### Convergence with Clipping

For convex $f$ with $L$-Lipschitz gradient:

**Without clipping:** SGD converges with rate $O(1/\sqrt{T})$

**With clipping:** Still converges, but may be slower for rare large gradients.

---

## 📊 When to Use

| Model Type | Gradient Clipping | Typical Threshold |
|------------|-------------------|-------------------|
| **RNN/LSTM** | Essential | 1.0 - 5.0 |
| **Transformers** | Recommended | 1.0 |
| **CNNs** | Rarely needed | - |
| **GANs** | Often helpful | 1.0 |
| **LLMs** | Standard | 1.0 |

---

## 💻 Implementation

```python
import torch
import torch.nn as nn

# ============ PyTorch Built-in ============

# Clip by global norm (most common)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# Returns the total norm before clipping (useful for monitoring)
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Gradient norm: {total_norm:.4f}")

# Clip by value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=0.5)

# ============ Manual Implementation ============

def clip_grad_norm(parameters, max_norm, norm_type=2):
    """
    Clip gradient norm manually
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    # Compute total norm
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    
    # Compute clipping coefficient
    clip_coef = max_norm / (total_norm + 1e-6)
    
    # Clip if needed
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    
    return total_norm

def clip_grad_value(parameters, clip_value):
    """
    Clip gradient values manually
    """
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(-clip_value, clip_value)

# ============ Training Loop Example ============

def train_step(model, batch, optimizer, clip_norm=1.0):
    optimizer.zero_grad()
    
    loss = model(batch)
    loss.backward()
    
    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    
    # Skip update if gradients are still too large (rare edge case)
    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        print("Skipping step due to NaN/Inf gradient")
        return None
    
    optimizer.step()
    
    return loss.item(), grad_norm.item()

# ============ Gradient Norm Monitoring ============

def compute_grad_norm(model, norm_type=2):
    """Compute gradient norm without clipping"""
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
    return total_norm ** (1. / norm_type)

# Track gradient norms during training
grad_norms = []
for batch in dataloader:
    loss.backward()
    grad_norm = compute_grad_norm(model)
    grad_norms.append(grad_norm)
    
    # Clip after measuring
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

# Plot to check for explosion
import matplotlib.pyplot as plt
plt.plot(grad_norms)
plt.xlabel('Step')
plt.ylabel('Gradient Norm')
plt.yscale('log')
plt.show()

```

---

## 🔬 Adaptive Gradient Clipping

### AdaClip (Adaptive Clipping)

Adjust threshold based on gradient history:

```math
\tau_t = \alpha \cdot \text{EMA}(\|g\|)

```

Where EMA is exponential moving average.

### Gradient Norm Penalty (Alternative)

Instead of hard clipping, add penalty:

```math
\mathcal{L}_{total} = \mathcal{L} + \lambda \max(0, \|g\| - \tau)

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | On the difficulty of training RNNs | [arXiv](https://arxiv.org/abs/1211.5063) |
| 📄 | Understanding Gradient Clipping | [arXiv](https://arxiv.org/abs/1905.11881) |
| 📖 | PyTorch clip_grad_norm_ | [Docs](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) |
| 🇨🇳 | 梯度裁剪详解 | [知乎](https://zhuanlan.zhihu.com/p/43454694) |

---

## 🔗 When to Use

| Model | Clipping |
|-------|----------|
| **RNN/LSTM** | Essential |
| **Transformers/LLMs** | Common (max_norm=1.0) |
| **CNNs** | Usually not needed |
| **GANs** | Often helpful |
| **Very Deep Networks** | Recommended |

---

➡️ [Next: Initialization](../02_initialization/README.md)

---

⬅️ [Back: Training Techniques](../../README.md)

---

➡️ [Next: Initialization](../02_initialization/README.md)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
