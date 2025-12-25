<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Linear%20Algebra%20for%20Optimizatio&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
