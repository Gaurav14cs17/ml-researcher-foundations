<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Jacobian%20Matrix&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Definition

```
For f: ℝⁿ → ℝᵐ

J = ∂f/∂x ∈ ℝᵐˣⁿ

J[i,j] = ∂fᵢ/∂xⱼ

"Row i = gradient of fᵢ"
```

---

## 🎯 Intuition

```
For small Δx:
f(x + Δx) ≈ f(x) + J·Δx

"Jacobian is the best linear approximation"
```

---

## 📊 Special Cases

| Type | Input | Output | Derivative |
|------|-------|--------|------------|
| Scalar function | 1 | 1 | Derivative (1×1) |
| Gradient | n | 1 | Gradient (1×n) |
| Jacobian | n | m | Jacobian (m×n) |

---

## 💻 Code

```python
import torch

# Function f: R³ → R²
def f(x):
    return torch.stack([
        x[0] * x[1],
        x[1] * x[2]
    ])

x = torch.tensor([1., 2., 3.], requires_grad=True)

# Compute Jacobian
jacobian = torch.autograd.functional.jacobian(f, x)
print(jacobian.shape)  # (2, 3)
# [[∂f₁/∂x₁, ∂f₁/∂x₂, ∂f₁/∂x₃],
#  [∂f₂/∂x₁, ∂f₂/∂x₂, ∂f₂/∂x₃]]
```

---

## 🌍 In Deep Learning

| Context | Jacobian |
|---------|----------|
| Layer output | ∂output/∂input |
| Change of variables | For flow-based models |
| Lipschitz constant | Spectral norm of Jacobian |

---

---

⬅️ [Back: Hessian](./hessian.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
