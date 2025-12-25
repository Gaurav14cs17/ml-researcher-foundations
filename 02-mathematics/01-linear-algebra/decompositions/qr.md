<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=QR%20Decomposition&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

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

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
