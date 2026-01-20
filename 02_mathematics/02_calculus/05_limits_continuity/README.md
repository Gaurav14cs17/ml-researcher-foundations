<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Limits%20and%20Continuity&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/limits.svg" width="100%">

*Caption: lim_{x→a} f(x) = L means f(x) gets arbitrarily close to L as x→a. Continuous: lim = f(a). Required for derivatives to exist. Ensures loss landscapes are smooth enough to optimize.*

---

## 📂 Overview

Limits formalize "approaching" behavior. Continuity ensures no jumps or breaks. Most activation functions are continuous (ReLU has a corner but is continuous). These properties enable gradient-based optimization.

---

## 📐 Mathematical Definitions

### Limit Definition (ε-δ)

```
lim_{x→a} f(x) = L

Means: ∀ε > 0, ∃δ > 0 such that
       0 < |x - a| < δ ⟹ |f(x) - L| < ε

```

### Limit Properties

```
lim [f(x) + g(x)] = lim f(x) + lim g(x)
lim [f(x) · g(x)] = lim f(x) · lim g(x)
lim [f(x)/g(x)] = lim f(x) / lim g(x)  (if lim g(x) ≠ 0)

```

### Continuity

```
f is continuous at a if:

1. f(a) is defined

2. lim_{x→a} f(x) exists

3. lim_{x→a} f(x) = f(a)

Continuous functions: polynomials, exp, log, sin, cos

```

### Importance in ML

```
• ReLU: max(0, x) is continuous but not differentiable at 0
• Sigmoid: σ(x) = 1/(1+e^{-x}) is smooth (infinitely differentiable)
• Softmax: exp(xᵢ)/Σexp(xⱼ) is smooth
• Loss landscapes need continuity for optimization

```

---

## 💻 Code Examples

```python
import numpy as np
import torch

# Numerical limit: lim_{x→0} sin(x)/x = 1
x = np.array([0.1, 0.01, 0.001, 0.0001])
print(np.sin(x) / x)  # → 1.0

# Check continuity via numerical gradient
def is_approximately_continuous(f, x, epsilon=1e-5):
    """Check if |f(x+ε) - f(x)| → 0 as ε → 0"""
    left = f(torch.tensor(x - epsilon))
    right = f(torch.tensor(x + epsilon))
    center = f(torch.tensor(x))
    return abs(left - center) < 0.1 and abs(right - center) < 0.1

# ReLU: continuous but not differentiable at 0
relu = torch.nn.ReLU()
print(is_approximately_continuous(relu, 0.0))  # True

# Activation functions and their properties
activations = {
    'ReLU': lambda x: torch.maximum(x, torch.tensor(0.)),
    'Sigmoid': torch.sigmoid,
    'GELU': torch.nn.functional.gelu,
    'Softplus': torch.nn.functional.softplus  # Smooth ReLU
}

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Rudin: Principles of Analysis | [Book](https://www.mheducation.com/highered/product/principles-mathematical-analysis-rudin/M9780070542358.html) |
| 🎥 | 3Blue1Brown: Calculus | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 📖 | Khan Academy Calculus | [Website](https://www.khanacademy.org/math/calculus-1) |
| 🇨🇳 | 极限与连续 | [知乎](https://zhuanlan.zhihu.com/p/25383715) |
| 🇨🇳 | 微积分基础 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 高等数学 | [B站](https://www.bilibili.com/video/BV1864y1T7Ks) |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Integration](../04_integration/README.md) | [Calculus](../README.md) | [Taylor Series](../06_taylor/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
