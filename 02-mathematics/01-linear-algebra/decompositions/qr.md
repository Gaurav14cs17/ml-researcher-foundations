<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=QR%20Decomposition&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
