# QR Decomposition

> **Orthogonal × Upper triangular**

---

## 📐 Definition

```
A = QR

Where:
• A: m × n matrix
• Q: m × m orthogonal (QᵀQ = I)
• R: m × n upper triangular
```

---

## 🌍 Applications

| Application | How |
|-------------|-----|
| Least squares | Solve Rx = Qᵀb |
| Eigenvalues | QR algorithm |
| Numerical stability | More stable than normal equations |

---

## 💻 Code

```python
import numpy as np

A = np.random.randn(5, 3)
Q, R = np.linalg.qr(A)

# Verify
print(Q @ R)  # ≈ A
print(Q.T @ Q)  # ≈ I

# Solve least squares: Ax = b
b = np.random.randn(5)
x = np.linalg.solve(R, Q.T @ b)
```

---

---

⬅️ [Back: Eigen Decomp](./eigen-decomp.md) | ➡️ [Next: Svd](./svd.md)
