# Eigenvalues and Eigenvectors

> **The fundamental objects of linear algebra**

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


