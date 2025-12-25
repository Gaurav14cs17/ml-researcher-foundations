<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Chain%20Rule&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/backprop-graph.svg" width="100%">

*Caption: A computational graph showing how the chain rule computes gradients. Forward pass (blue) computes values; backward pass (red) propagates gradients using the chain rule. This is exactly how PyTorch and TensorFlow compute gradients automatically.*

---

## 📂 Topics in This Folder

| File | Topic | Application |
|------|-------|-------------|
| [backpropagation.md](./backpropagation.md) | 🔥 Reverse-mode autodiff | Training NNs |

---

## 📐 Mathematical Foundations

### Single Variable Chain Rule
```
If y = f(g(x)):
dy/dx = f'(g(x)) · g'(x)

Example: y = sin(x²)
dy/dx = cos(x²) · 2x
```

### Multivariate Chain Rule
```
If z = f(u, v), u = u(x, y), v = v(x, y):
∂z/∂x = (∂z/∂u)(∂u/∂x) + (∂z/∂v)(∂v/∂x)
```

### Matrix Form (Jacobian)
```
If f: ℝⁿ → ℝᵐ, g: ℝᵏ → ℝⁿ:
J_{f∘g} = J_f · J_g

This is why backpropagation multiplies Jacobians!
```

---

## 🎯 The Core Idea

```
Single variable:
If y = f(g(x)), then dy/dx = dy/du · du/dx
                              ↑       ↑
                           outer   inner

Multivariable:
If L = L(y₁, y₂, ..., yₘ) and each yᵢ = yᵢ(x)
Then:
∂L/∂x = Σᵢ (∂L/∂yᵢ) · (∂yᵢ/∂x)
        -------------------------
        Sum over all paths from L to x
```

---

## 🔥 Backpropagation is Just Chain Rule!

```
Neural Network:
x → [W₁] → h₁ → [W₂] → h₂ → [W₃] → ŷ → L

∂L/∂W₁ = ∂L/∂ŷ · ∂ŷ/∂h₂ · ∂h₂/∂h₁ · ∂h₁/∂W₁
         -----   ------   ------   ------
         From L   Layer 3  Layer 2  Layer 1
         
Forward:  x → h₁ → h₂ → ŷ → L
Backward: ∂L/∂ŷ ← ∂L/∂h₂ ← ∂L/∂h₁ ← ∂L/∂W₁
```

---

## 💻 Code Example

```python
import torch

# Computational graph
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2        # y = x²
z = y ** 3        # z = y³ = x⁶

z.backward()      # Compute ∂z/∂x via chain rule

# Manual: dz/dx = dz/dy · dy/dx = 3y² · 2x = 3(x²)² · 2x = 6x⁵
print(x.grad)     # tensor([192.]) = 6 * 2⁵
```

---

## 📊 Forward vs Backward Mode

```
For function f: ℝⁿ → ℝᵐ

Forward mode (tangent):
- Compute ∂fᵢ/∂xⱼ one column at a time
- Cost: O(n) passes for full Jacobian
- Good when: n < m (few inputs)

Backward mode (adjoint):
- Compute ∂fᵢ/∂xⱼ one row at a time  
- Cost: O(m) passes for full Jacobian
- Good when: m < n (few outputs)

Deep learning: L: ℝ^(millions) → ℝ (one loss!)
→ Backward mode (backprop) is perfect!
→ One backward pass computes ALL gradients!
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | Automatic Differentiation Survey | [arXiv](https://arxiv.org/abs/1502.05767) |
| 🎥 | Backprop Explained | [3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U) |
| 📖 | Deep Learning Book Ch 6 | Goodfellow |

---

## 🔗 Where This Topic Is Used

| Topic | How Chain Rule Is Used |
|-------|----------------------|
| **Backpropagation** | Chain rule computes all gradients |
| **PyTorch autograd** | Implements reverse-mode chain rule |
| **Transformer Training** | Gradient through attention layers |
| **CNN Training** | Gradient through conv layers |
| **RNN / LSTM** | Backprop through time (BPTT) |
| **Diffusion Training** | Gradient through denoising steps |
| **Neural ODE** | Adjoint method = continuous chain rule |
| **Differentiable Rendering** | Gradient through rendering |
| **Physics-Informed NN** | Gradient of physics constraints |

### Used In These Computations

| Computation | Chain Rule Application |
|-------------|----------------------|
| `loss.backward()` | Millions of chain rule applications |
| Gradient clipping | Modify chain rule outputs |
| Second derivatives | Chain rule of chain rule |
| Jacobian-vector product | Efficient chain rule |

### Prerequisite For

```
Chain Rule --> Backpropagation
          --> Automatic differentiation
          --> Understanding gradient flow
          --> Implementing custom layers
          --> Debugging vanishing/exploding gradients
```

---

⬅️ [Back: Calculus](../)

---

➡️ [Next: Derivatives](../derivatives/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
