<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Eigenvalues%20and%20Eigenvectors&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
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
Av = λv

Where:
• A is n×n matrix
• v is eigenvector (v ≠ 0)
• λ is eigenvalue

"A scales v by λ, without changing direction"
```

---

## 🔑 Finding Eigenvalues

```
det(A - λI) = 0  (characteristic polynomial)

For 2×2:
λ² - tr(A)λ + det(A) = 0

For n×n: Use numerical methods (QR algorithm)
```

---

## 📊 Properties

| Property | Value |
|----------|-------|
| Sum of eigenvalues | tr(A) |
| Product of eigenvalues | det(A) |
| Eigenvalues of Aᵏ | λᵏ |
| Eigenvalues of A⁻¹ | 1/λ |

---

## 💻 Code

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)   # [5., 2.]
print("Eigenvectors:\n", eigenvectors)

# Verify: Av = λv
v = eigenvectors[:, 0]
lam = eigenvalues[0]
print(A @ v)      # Should equal λv
print(lam * v)    # Same!
```

---

## 🌍 ML Applications

| Application | How Used |
|-------------|----------|
| PCA | Eigenvectors of covariance |
| PageRank | Dominant eigenvector |
| Graph clustering | Laplacian eigenvalues |
| Stability analysis | Eigenvalues of Hessian |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
