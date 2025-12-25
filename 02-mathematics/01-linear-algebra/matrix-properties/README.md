<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Matrix%20Properties&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/matrix-properties.svg" width="100%">

*Caption: Key properties: rank (dimension of column space), determinant (volume scaling), trace (sum of eigenvalues), conditioning (numerical stability). Positive definite: x'Ax > 0 for all x≠0.*

---

## 📂 Overview

Matrix properties determine algorithm behavior. Rank tells us about solvability, conditioning affects numerical stability, and positive definiteness ensures convexity of quadratic forms.

---

## 📐 Mathematical Definitions

### Rank
```
rank(A) = dim(column space) = dim(row space)

Properties:
• rank(A) ≤ min(m, n) for A ∈ ℝᵐˣⁿ
• Full rank: rank(A) = min(m, n)
• rank(AB) ≤ min(rank(A), rank(B))
• Low-rank ⟹ redundant information (LoRA exploits this!)
```

### Determinant
```
det(A) = product of eigenvalues

Properties:
• det(AB) = det(A) · det(B)
• det(A⁻¹) = 1/det(A)
• det(A) = 0 ⟺ A is singular
• |det(A)| = volume scaling factor
```

### Trace
```
tr(A) = Σᵢ Aᵢᵢ = sum of eigenvalues

Properties:
• tr(AB) = tr(BA)  (cyclic)
• tr(A + B) = tr(A) + tr(B)
• tr(A) = tr(Aᵀ)
```

### Positive Definite
```
A is positive definite ⟺ xᵀAx > 0 ∀x ≠ 0
                       ⟺ all eigenvalues > 0
                       ⟺ A = LLᵀ (Cholesky)

Used in:
• Covariance matrices (always PSD)
• Hessians at minima (positive definite)
• Kernel matrices (PSD required)
```

### Condition Number
```
κ(A) = ||A|| · ||A⁻¹|| = σₘₐₓ/σₘᵢₙ

• κ ≈ 1: well-conditioned
• κ >> 1: ill-conditioned (numerically unstable)
• κ = ∞: singular
```

---

## 💻 Code Examples

```python
import numpy as np
from scipy.linalg import cholesky

A = np.array([[4, 2], [2, 3]])

# Basic properties
print(f"Rank: {np.linalg.matrix_rank(A)}")
print(f"Determinant: {np.linalg.det(A)}")
print(f"Trace: {np.trace(A)}")

# Condition number
print(f"Condition: {np.linalg.cond(A)}")

# Check positive definite
eigenvalues = np.linalg.eigvalsh(A)
is_pd = all(eigenvalues > 0)
print(f"Positive definite: {is_pd}")

# Cholesky decomposition (only works for PD)
L = cholesky(A, lower=True)  # A = LLᵀ
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Strang Linear Algebra | [Book](https://math.mit.edu/~gs/linearalgebra/) |
| 🎥 | 3Blue1Brown | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) |
| 📖 | NumPy linalg | [Docs](https://numpy.org/doc/stable/reference/routines.linalg.html) |
| 🇨🇳 | 矩阵性质详解 | [知乎](https://zhuanlan.zhihu.com/p/29846048) |
| 🇨🇳 | 正定矩阵 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88785898) |
| 🇨🇳 | 线性代数课程 | [B站](https://www.bilibili.com/video/BV1ys411472E) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: matrix-properties](../)

---

⬅️ [Back: Eigen](../eigen/) | ➡️ [Next: Transformations](../transformations/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
