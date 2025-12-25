<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Singular%20Value%20Decomposition&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/svd-complete.svg" width="100%">

*Caption: SVD decomposes any matrix A = UΣV^T into orthogonal matrices and singular values.*

---

## 📐 Key Formulas

```
A = UΣV^T

Where:
• U: m×m orthogonal (left singular vectors)
• Σ: m×n diagonal (singular values σ₁ ≥ σ₂ ≥ ... ≥ 0)
• V: n×n orthogonal (right singular vectors)

Low-rank approximation (rank k):
A_k = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢ^T

Best rank-k approximation (Eckart-Young theorem)
```

---

## 💻 Code Examples

```python
import numpy as np

A = np.random.randn(100, 50)

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=False)

# Low-rank approximation
k = 10
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Approximation error
error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
print(f"Relative error with rank-{k}: {error:.4f}")
```

---

## 🌍 Applications

| Application | How SVD Is Used |
|-------------|-----------------|
| **PCA** | SVD of centered data |
| **LoRA** | Low-rank weight updates |
| **Compression** | Keep top-k singular values |
| **Recommenders** | Matrix factorization |
| **Pseudoinverse** | A⁺ = VΣ⁺U^T |

---

➡️ See [Decompositions](../decompositions/) for more details

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
