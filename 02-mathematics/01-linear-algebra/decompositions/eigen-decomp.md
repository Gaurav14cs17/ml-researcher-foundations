<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Eigendecomposition&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Eigendecomposition

> **Diagonalizing a matrix**

---

## 📐 Definition

```
A = VΛV⁻¹

Where:
• V: Matrix of eigenvectors (columns)
• Λ: Diagonal matrix of eigenvalues
• V⁻¹: Inverse of V

For symmetric A: A = VΛVᵀ (orthogonal V)
```

---

## 🔑 Properties

| Property | Value |
|----------|-------|
| Existence | Requires n linearly independent eigenvectors |
| Symmetric matrices | Always diagonalizable |
| Powers | Aⁿ = VΛⁿV⁻¹ |
| Functions | f(A) = Vf(Λ)V⁻¹ |

---

## 🌍 Applications

| Application | Use |
|-------------|-----|
| PCA | Eigenvectors of XᵀX |
| Spectral clustering | Eigenvectors of Laplacian |
| Matrix powers | Aⁿ efficiently |
| ODE solutions | e^{At} |

---

## 💻 Code

```python
import numpy as np

# Symmetric matrix
A = np.array([[4, 2], [2, 3]])

# Eigendecomposition
eigenvalues, V = np.linalg.eig(A)
# For symmetric: np.linalg.eigh (more stable)

# Verify: A = V Λ V^(-1)
Lambda = np.diag(eigenvalues)
A_reconstructed = V @ Lambda @ np.linalg.inv(V)
print(np.allclose(A, A_reconstructed))  # True

# Matrix power using eigendecomposition
A_squared = V @ np.diag(eigenvalues**2) @ np.linalg.inv(V)
print(np.allclose(A @ A, A_squared))  # True
```

---

## ⚠️ Limitations

```
Not all matrices are diagonalizable:
• Defective matrices (not enough eigenvectors)

For those, use:
• Jordan normal form
• Schur decomposition
• SVD (always exists)
```

---

---

➡️ [Next: Qr](./qr.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
