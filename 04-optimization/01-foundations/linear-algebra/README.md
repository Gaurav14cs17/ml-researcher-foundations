<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Linear Algebra for Optimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 📐 Linear Algebra for Optimization

> **Matrix operations essential for optimization**

---

## 📐 Key Concepts

```
Gradient: ∇f(x) = [∂f/∂x₁, ..., ∂f/∂xₙ]ᵀ

Hessian: H = [∂²f/∂xᵢ∂xⱼ]

Positive Definite: xᵀHx > 0 for all x ≠ 0
→ Local minimum if ∇f = 0 and H ≻ 0

Condition Number: κ(H) = λₘₐₓ/λₘᵢₙ
→ Large κ = slow convergence
```

---

## 💻 Code Example

```python
import torch

def compute_hessian(f, x):
    """Compute Hessian matrix"""
    return torch.autograd.functional.hessian(f, x)

# Check if positive definite
def is_positive_definite(H):
    eigenvalues = torch.linalg.eigvalsh(H)
    return (eigenvalues > 0).all()
```

---

⬅️ [Back: Foundations](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

