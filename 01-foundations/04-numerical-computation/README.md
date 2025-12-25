<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=04 Numerical Computation&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔢 Numerical Computation

> **Understanding numerical stability and precision in ML**

---

## 🎯 Visual Overview

<img src="./images/numerical-stability-complete.svg" width="100%">

*Caption: Numerical computation deals with floating-point arithmetic, numerical stability, and precision issues that affect ML training and inference.*

---

## 📐 Mathematical Foundations

### Floating-Point Representation
```
IEEE 754 Single Precision (32-bit):
(-1)^s × 1.mantissa × 2^(exponent-127)

• Sign: 1 bit
• Exponent: 8 bits (range: 2^-126 to 2^127)
• Mantissa: 23 bits (precision: ~7 decimal digits)

Double Precision (64-bit):
• Exponent: 11 bits
• Mantissa: 52 bits (precision: ~16 decimal digits)
```

### Machine Epsilon
```
ε_machine = smallest ε where 1 + ε ≠ 1

float32:  ε ≈ 1.19 × 10^-7
float64:  ε ≈ 2.22 × 10^-16
float16:  ε ≈ 9.77 × 10^-4

Implication: Numbers closer than ε are indistinguishable
```

### Numerical Stability
```
Stable Algorithm:
Small input perturbations → Small output perturbations

Condition Number:
κ(A) = ||A|| · ||A^-1||

• κ ≈ 1: Well-conditioned (stable)
• κ >> 1: Ill-conditioned (unstable)
```

---

## 🎯 Key Concepts

| Concept | Description | ML Impact |
|---------|-------------|-----------|
| **Overflow** | Number too large | Exploding gradients |
| **Underflow** | Number too small (→0) | Vanishing gradients |
| **Catastrophic Cancellation** | Subtracting similar numbers | Loss of precision |
| **Numerical Stability** | Algorithm sensitivity | Training reliability |

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Machine epsilon
print(f"float32 epsilon: {np.finfo(np.float32).eps}")
print(f"float64 epsilon: {np.finfo(np.float64).eps}")

# Overflow example
x = torch.tensor([1000.0], dtype=torch.float32)
print(f"exp(1000): {torch.exp(x)}")  # inf!

# Log-sum-exp trick for stability
def log_sum_exp_stable(x):
    """Numerically stable log-sum-exp"""
    c = x.max()
    return c + torch.log(torch.sum(torch.exp(x - c)))

logits = torch.tensor([1000.0, 1001.0, 1002.0])
print(f"Stable LSE: {log_sum_exp_stable(logits)}")

# Softmax stability
def softmax_stable(x):
    """Numerically stable softmax"""
    x_max = x.max(dim=-1, keepdim=True).values
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# Condition number
A = np.array([[1, 2], [2, 4.0001]])
cond = np.linalg.cond(A)
print(f"Condition number: {cond:.2f}")
```

---

## 🌍 ML Applications

| Application | Numerical Issue | Solution |
|-------------|-----------------|----------|
| **Softmax** | exp overflow | Subtract max before exp |
| **Log Likelihood** | log(0) = -inf | Add small epsilon |
| **Cross Entropy** | log(p) when p→0 | Clamp probabilities |
| **Gradient Descent** | Vanishing/exploding | Gradient clipping, normalization |
| **Matrix Inversion** | Ill-conditioned | Regularization, pseudo-inverse |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Numerical Recipes | [Book](http://numerical.recipes/) |
| 📖 | Deep Learning Book Ch. 4 | [Book](https://www.deeplearningbook.org/contents/numerical.html) |
| 🎥 | Floating Point (Computerphile) | [YouTube](https://www.youtube.com/watch?v=PZRI1IfStY0) |
| 🇨🇳 | 数值计算基础 | [知乎](https://zhuanlan.zhihu.com/p/32855481) |
| 🇨🇳 | 深度学习数值稳定性 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88622888) |

---

## 🔗 Where This Topic Is Used

| Application | How Numerical Computation Is Used |
|-------------|----------------------------------|
| **Training Neural Networks** | Stable gradients, loss computation |
| **Mixed Precision Training** | FP16/BF16 for speed, FP32 for accumulation |
| **Softmax/Cross-Entropy** | Log-sum-exp trick for stability |
| **Attention Mechanism** | Scaled dot-product prevents overflow |
| **Normalization Layers** | Numerical stability in mean/variance |

---

⬅️ [Back: 03-Set Theory](../03-set-theory/) | ➡️ [Next: 05-Asymptotic Analysis](../05-asymptotic-analysis/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

