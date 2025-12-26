<!-- Animated Header -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=F38181&center=true&vCenter=true&width=800&lines=🔢+Numerical+Computation;Floating+Point+and+Stability" alt="Numerical Computation" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=8,9,10&height=180&section=header&text=Numerical%20Computation&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=IEEE%20754%20•%20Stability%20•%20Mixed%20Precision%20•%20NaN%20Debugging&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06_of_06-F38181?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Topics-5_Concepts-4ECDC4?style=for-the-badge&logo=buffer&logoColor=white" alt="Topics"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-6C63FF?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-00D4AA?style=for-the-badge&logo=calendar&logoColor=white" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01_mathematical_thinking/README.md) → [Proof Techniques](../02_proof_techniques/README.md) → [Set Theory](../03_set_theory/README.md) → [Logic](../04_logic/README.md) → [Asymptotic Analysis](../05_asymptotic_analysis/README.md) → Numerical Computation

---

## 📌 TL;DR

Understanding floating-point math **prevents mysterious training failures**:

- **IEEE 754 Format** — How FP32, FP16, BF16 represent numbers
- **Numerical Issues** — Overflow, underflow, catastrophic cancellation
- **Stable Algorithms** — Log-sum-exp trick, stable softmax
- **Mixed Precision** — Train 2× faster with FP16/BF16

> [!CAUTION]
> **Common cause of `loss = NaN`:**
> ```python
> # ❌ BAD: Overflow in exp()
> softmax = np.exp(x) / np.sum(np.exp(x))
> 
> # ✅ GOOD: Subtract max first
> softmax = np.exp(x - x.max()) / np.sum(np.exp(x - x.max()))
> ```

---

## 📚 What You'll Learn

- [ ] Understand IEEE 754 floating-point representation
- [ ] Identify and fix numerical instability issues
- [ ] Implement stable softmax and log-sum-exp
- [ ] Use mixed precision training effectively
- [ ] Debug NaN/Inf errors in training

---

## 📑 Table of Contents

1. [IEEE 754 Representation](#1-ieee-754-floating-point)
2. [Common Numerical Issues](#2-common-numerical-issues)
3. [Stable Algorithms](#3-stable-algorithms)
4. [Mixed Precision Training](#4-mixed-precision-training)
5. [Debugging NaN/Inf](#5-debugging-naninf)
6. [Key Numbers](#-key-numbers-to-remember)
7. [Code Implementations](#-code-implementations)
8. [Resources](#-resources)

---

## 1. IEEE 754 Floating Point

### 📖 Representation

```
IEEE 754 Float32 (32 bits):
┌───┬──────────────┬─────────────────────────────┐
│ S │   Exponent   │          Mantissa           │
│ 1 │    8 bits    │          23 bits            │
└───┴──────────────┴─────────────────────────────┘

Value = (-1)^S × (1 + M/2²³) × 2^(E-127)

Where:
  S = Sign bit (0 = positive, 1 = negative)
  E = Exponent (8 bits, bias = 127)
  M = Mantissa/Significand (23 bits, implicit leading 1)
```

### 📊 Format Comparison

| Format | Bits | Exponent | Mantissa | Range | Precision | ML Use |
|:------:|:----:|:--------:|:--------:|:-----:|:---------:|:-------|
| **FP32** | 32 | 8 | 23 | ±3.4×10³⁸ | ~7 digits | Default training |
| **FP16** | 16 | 5 | 10 | ±65,504 | ~3 digits | Mixed precision |
| **BF16** | 16 | 8 | 7 | ±3.4×10³⁸ | ~2 digits | TPU/A100+ |
| **FP8** | 8 | 4/5 | 3/2 | varies | ~1 digit | H100 training |

### 📐 Key Insight: BF16 vs FP16

```
FP16:  [1 sign] [5 exp] [10 mantissa]  → Range: ±65,504
BF16:  [1 sign] [8 exp] [7 mantissa]   → Range: ±3.4×10³⁸

BF16 has FP32's range but less precision.
Better for training (gradients can be large).
FP16 often needs loss scaling to avoid underflow.
```

---

## 2. Common Numerical Issues

### 🔥 Issue 1: Overflow in Softmax

```python
# ❌ BAD: exp(1000) = inf
x = np.array([1000, 1001, 1002])
softmax = np.exp(x) / np.sum(np.exp(x))  # [nan, nan, nan]

# ✅ GOOD: Subtract max (numerically stable)
x_shifted = x - np.max(x)  # [0, 1, 2]
softmax = np.exp(x_shifted) / np.sum(np.exp(x_shifted))  # Works!
```

**Why it works:** softmax(x - c) = softmax(x) for any constant c.

### 🔥 Issue 2: Underflow in Log-Sum-Exp

```python
# ❌ BAD: exp(-1000) = 0, log(0) = -inf
x = np.array([-1000, -1001, -1002])
logsumexp = np.log(np.sum(np.exp(x)))  # -inf (wrong!)

# ✅ GOOD: Log-sum-exp trick
c = np.max(x)
logsumexp = c + np.log(np.sum(np.exp(x - c)))  # -999.59 (correct!)
```

### 🔥 Issue 3: Catastrophic Cancellation

```python
# ❌ BAD: Subtracting nearly equal numbers
a = 1.0000001
b = 1.0000000
diff = a - b  # Should be 1e-7, but loses precision!

# Example: Variance calculation
x = np.array([1e8, 1e8 + 1, 1e8 + 2])
var_bad = np.mean(x**2) - np.mean(x)**2  # 0 or negative! (WRONG)
var_good = np.var(x)  # 0.667 (correct, uses stable algorithm)
```

---

## 3. Stable Algorithms

### 📐 Stable Softmax

```python
def stable_softmax(x):
    """Numerically stable softmax."""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### 📐 Stable Log-Sum-Exp

```python
def stable_logsumexp(x):
    """Numerically stable log-sum-exp."""
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))
```

### 📐 Stable Cross-Entropy

```python
def stable_cross_entropy(logits, labels):
    """Stable cross-entropy from logits (not probabilities)."""
    # Don't do: -log(softmax(logits))
    # Do: Use log_softmax directly
    log_probs = logits - stable_logsumexp(logits)
    return -np.sum(labels * log_probs)
```

### 📐 Proof: Softmax Shift Invariance

**Theorem:** softmax(x - c) = softmax(x) for any scalar c.

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | softmax(x - c)ᵢ = exp(xᵢ - c) / Σⱼ exp(xⱼ - c) | Definition |
| 2 | = exp(xᵢ)·exp(-c) / Σⱼ exp(xⱼ)·exp(-c) | Exponent rule |
| 3 | = exp(xᵢ)·exp(-c) / (exp(-c)·Σⱼ exp(xⱼ)) | Factor out |
| 4 | = exp(xᵢ) / Σⱼ exp(xⱼ) | Cancel exp(-c) |
| 5 | = softmax(x)ᵢ | ∎ |

---

## 4. Mixed Precision Training

### 📖 Why Mixed Precision?

```
┌─────────────────────────────────────────────────────────────┐
│                   MIXED PRECISION TRAINING                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Forward Pass (FP16)  →  Loss (FP32)  →  Backward (FP16)  │
│         ↓                                      ↓           │
│   Faster compute                        Gradients (FP16)   │
│   Less memory                                  ↓           │
│                                         Scale gradients    │
│                                                  ↓         │
│                                         Update (FP32)      │
│                                         Master weights     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 💻 PyTorch Implementation

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # Forward in FP16
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # Backward with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 📊 Benefits

| Metric | FP32 | Mixed (FP16) | Improvement |
|:------:|:----:|:------------:|:-----------:|
| Memory | 100% | ~50% | 2× less |
| Speed | 1× | ~2× | 2× faster |
| Accuracy | Baseline | Same | No loss |

---

## 5. Debugging NaN/Inf

### 🔍 Common Causes

| Symptom | Likely Cause | Fix |
|:--------|:-------------|:----|
| `loss = NaN` after a few epochs | Learning rate too high | Reduce LR |
| `loss = NaN` immediately | Bad initialization | Check init |
| `loss = inf` | Overflow in exp() | Use stable softmax |
| Gradients = 0 | Underflow | Use loss scaling |

### 💻 Debugging Code

```python
def debug_nan(model, x, y):
    """Find where NaN/Inf appears."""
    
    # Check input
    if torch.isnan(x).any() or torch.isinf(x).any():
        print("NaN/Inf in INPUT!")
        return
    
    # Hook to check activations
    def hook(name):
        def fn(module, input, output):
            if torch.isnan(output).any():
                print(f"NaN in {name}")
            if torch.isinf(output).any():
                print(f"Inf in {name}")
        return fn
    
    # Register hooks
    handles = []
    for name, module in model.named_modules():
        handles.append(module.register_forward_hook(hook(name)))
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    # Clean up
    for h in handles:
        h.remove()
```

---

## 📊 Key Numbers to Remember

| Constant | FP32 | FP16 | Notes |
|:---------|:----:|:----:|:------|
| **Machine ε** | 1.19×10⁻⁷ | 9.77×10⁻⁴ | Smallest x: 1+x ≠ 1 |
| **Max value** | 3.4×10³⁸ | 65,504 | Before overflow |
| **Min positive** | 1.18×10⁻³⁸ | 6.10×10⁻⁵ | Before underflow |

> [!WARNING]
> **FP16 max is only 65,504!** Logits > 11 will overflow in exp()!

---

## 💻 Code Implementations

```python
import numpy as np
import torch

# =============================================================================
# STABLE IMPLEMENTATIONS
# =============================================================================

def stable_softmax_numpy(x, axis=-1):
    """Numerically stable softmax in NumPy."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def stable_logsumexp_numpy(x, axis=-1):
    """Numerically stable log-sum-exp in NumPy."""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(x_max, axis) + np.log(np.sum(np.exp(x - x_max), axis=axis))

def stable_cross_entropy_numpy(logits, targets):
    """Stable cross-entropy loss."""
    log_softmax = logits - stable_logsumexp_numpy(logits, axis=-1, keepdims=True)
    return -np.sum(targets * log_softmax) / len(targets)

# =============================================================================
# DEMONSTRATIONS
# =============================================================================

def demo_overflow():
    """Demonstrate overflow and fix."""
    print("=== Overflow Demo ===")
    x = np.array([1000., 1001., 1002.])
    
    print(f"Input: {x}")
    
    # Bad
    try:
        bad = np.exp(x) / np.sum(np.exp(x))
        print(f"Naive softmax: {bad}")
    except:
        print("Naive softmax: OVERFLOW")
    
    # Good
    good = stable_softmax_numpy(x)
    print(f"Stable softmax: {good}")

def demo_underflow():
    """Demonstrate underflow and fix."""
    print("\n=== Underflow Demo ===")
    x = np.array([-1000., -1001., -1002.])
    
    print(f"Input: {x}")
    
    # Bad
    bad = np.log(np.sum(np.exp(x)))
    print(f"Naive logsumexp: {bad}")
    
    # Good
    good = stable_logsumexp_numpy(x)
    print(f"Stable logsumexp: {good}")

def demo_cancellation():
    """Demonstrate catastrophic cancellation."""
    print("\n=== Cancellation Demo ===")
    x = np.array([1e8, 1e8 + 1, 1e8 + 2])
    
    print(f"Input: {x}")
    
    # Bad: E[X²] - E[X]²
    mean_sq = np.mean(x**2)
    sq_mean = np.mean(x)**2
    var_bad = mean_sq - sq_mean
    print(f"Naive variance: {var_bad}")
    
    # Good: Two-pass
    var_good = np.var(x)
    print(f"Stable variance: {var_good}")

# Run demos
demo_overflow()
demo_underflow()
demo_cancellation()
```

---

## 📚 Resources

| Type | Title | Link |
|:-----|:------|:-----|
| 📄 Paper | What Every Computer Scientist Should Know About Floating-Point | [ACM](https://dl.acm.org/doi/10.1145/103162.103163) |
| 📄 Paper | Mixed Precision Training | [arXiv](https://arxiv.org/abs/1710.03740) |
| 🎥 Video | Computerphile: Floating Point | [YouTube](https://www.youtube.com/watch?v=PZRI1IfStY0) |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[⏱️ Asymptotic Analysis](../05_asymptotic_analysis/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 6 of 6**<br>
**🔢 Numerical Computation**

</td>
<td align="right" width="33%">

➡️ **Next**<br>
[🏠 Section Home](../README.md)

</td>
</tr>
</table>

### Continue Learning

| Direction | Destination |
|:---------:|-------------|
| 🏠 Section Home | [01: Mathematical Foundations](../README.md) |
| ➡️ Next Section | [02: Mathematics (Linear Algebra, Calculus)](../../02_mathematics/README.md) |
| 📋 Full Course | [Course Home](../../README.md) |

---

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=24&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=🎉+Congratulations!;You've+completed+the+Foundations+series!" alt="Completion" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=8,9,10&height=100&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=F38181&center=true&vCenter=true&width=600&lines=Made+with+❤️+by+Gaurav+Goswami" alt="Footer" />
</p>
