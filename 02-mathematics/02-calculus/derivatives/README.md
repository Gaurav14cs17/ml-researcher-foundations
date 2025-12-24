<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Derivatives&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Derivatives

> **The foundation of gradient-based optimization**

---

## 🎯 Visual Overview

<img src="./images/derivatives.svg" width="100%">

*Caption: A derivative f'(x) measures the instantaneous rate of change - the slope of the tangent line at point x. The chain rule is critical for backpropagation: it tells us how to propagate gradients through composed functions.*

---

## 📂 Overview

Derivatives are the mathematical foundation of machine learning optimization. Understanding them is essential for gradient descent, backpropagation, and neural network training.

---

## 📐 Definition

```
Derivative as limit:
f'(x) = lim[h→0] (f(x+h) - f(x)) / h

Interpretation:
- Rate of change at point x
- Slope of tangent line
- Sensitivity of output to input
```

---

## 🔑 Essential Rules

| Rule | Formula |
|------|---------|
| **Power** | d/dx xⁿ = nxⁿ⁻¹ |
| **Chain** | d/dx f(g(x)) = f'(g(x))·g'(x) |
| **Product** | d/dx (fg) = f'g + fg' |
| **Sum** | d/dx (f+g) = f' + g' |

---

## 🌍 ML Applications

| Function | Derivative | Where Used |
|----------|------------|------------|
| **ReLU** | 0 or 1 | CNNs |
| **Sigmoid** | σ(x)(1-σ(x)) | Binary output |
| **Softmax** | complex | Classification |
| **Cross-entropy** | p - y | Loss gradient |

---

## 💻 Code

```python
import torch

# Automatic differentiation in PyTorch
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x + 1  # f(x) = x² + 3x + 1

y.backward()
print(f"f(2) = {y.item()}")       # f(2) = 11
print(f"f'(2) = {x.grad.item()}")  # f'(2) = 2x + 3 = 7

# Numerical gradient (for verification)
def numerical_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

f = lambda x: x**2 + 3*x + 1
print(f"Numerical: {numerical_grad(f, 2.0)}")  # ≈ 7.0
```

---

⬅️ [Back: Calculus](../)

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Chain Rule](../chain-rule/) | ➡️ [Next: Gradients](../gradients/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
