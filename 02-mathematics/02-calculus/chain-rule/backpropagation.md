<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Backpropagation&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Backpropagation

> **The algorithm that makes deep learning work**

---

## 📐 Core Idea

```
Forward pass: Compute output and loss
x → h₁ → h₂ → ... → ŷ → L

Backward pass: Compute gradients via chain rule
∂L/∂x ← ∂L/∂h₁ ← ∂L/∂h₂ ← ... ← ∂L/∂ŷ ← 1

Key insight: Reuse intermediate derivatives!
```

---

## 🔑 Chain Rule Application

```
∂L/∂W₁ = ∂L/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁

Each layer passes gradient backward:
δₗ = ∂L/∂hₗ (gradient w.r.t. layer output)
δₗ₋₁ = δₗ · ∂hₗ/∂hₗ₋₁ (chain rule)
```

---

## 💻 Code Example

```python
import torch

# Simple 2-layer network
x = torch.randn(10, requires_grad=True)
W1 = torch.randn(10, 5, requires_grad=True)
W2 = torch.randn(5, 1, requires_grad=True)

# Forward pass
h = torch.relu(x @ W1)
y = h @ W2
loss = y.sum()

# Backward pass (automatic!)
loss.backward()

# Gradients computed
print(W1.grad.shape)  # (10, 5)
print(W2.grad.shape)  # (5, 1)
```

---

## 📊 Complexity

```
Forward: O(n) operations
Backward: O(n) operations (same order!)

Memory: O(n) to store activations for backward pass
```

---

<- [Back](./README.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
