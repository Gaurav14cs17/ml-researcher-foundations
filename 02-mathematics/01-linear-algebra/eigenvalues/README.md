<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Eigenvalues&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔢 Eigenvalues

> **Scalar values that characterize linear transformations**

---

## 🎯 Visual Overview

<img src="./images/eigenvalues-complete.svg" width="100%">

*Caption: Eigenvalues λ and eigenvectors v satisfy Av = λv. They reveal the fundamental properties of matrices.*

---

## 📐 Key Concepts

```
Eigenvalue Equation:
Av = λv

Finding eigenvalues:
det(A - λI) = 0 (characteristic polynomial)

Properties:
• tr(A) = Σλᵢ (trace = sum of eigenvalues)
• det(A) = Πλᵢ (determinant = product)
• A is invertible ⟺ all λᵢ ≠ 0
```

---

## 💻 Code Examples

```python
import numpy as np

A = np.array([[4, 2], [1, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors:\n{eigenvectors}")

# Verify: Av = λv
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"Av = {A @ v}, λv = {lam * v}")
```

---

➡️ See [Eigen](../eigen/) for complete theory

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

