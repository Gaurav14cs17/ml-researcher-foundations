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
