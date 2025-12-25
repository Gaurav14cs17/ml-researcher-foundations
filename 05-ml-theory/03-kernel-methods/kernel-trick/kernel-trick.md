<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=The%20Kernel%20Trick&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/kernel-trick.svg" width="100%">

*Caption: The kernel trick allows computing dot products in high (even infinite) dimensional feature spaces without ever explicitly computing the transformation φ(x). This enables linear methods to learn non-linear boundaries.*

---

## 📐 Core Idea

```
Instead of:
1. Map x → φ(x) into high-dimensional space
2. Compute φ(x)ᵀφ(y)

Use kernel:
k(x, y) = φ(x)ᵀφ(y)

"Compute inner product without explicit mapping"
```

---

## 📊 Common Kernels

| Kernel | Formula | Feature Space |
|--------|---------|---------------|
| Linear | xᵀy | Same |
| Polynomial | (xᵀy + c)^d | O(n^d) |
| RBF/Gaussian | exp(-γ\|\|x-y\|\|²) | Infinite! |
| Laplacian | exp(-γ\|\|x-y\|\|₁) | Infinite |

---

## 🔑 Why It Works

```
Mercer's theorem:
k(x, y) is a valid kernel iff:
K matrix is positive semi-definite for any data

φ(x) exists implicitly!
```

---

## 💻 Code

```python
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

# Explicit RBF kernel
def rbf_kernel_manual(X, Y, gamma=1.0):
    """K[i,j] = exp(-gamma * ||x_i - y_j||^2)"""
    dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
            np.sum(Y**2, axis=1) - 2 * X @ Y.T
    return np.exp(-gamma * dists)

# Using sklearn
X = np.random.randn(100, 10)
K = rbf_kernel(X, gamma=0.1)
print(K.shape)  # (100, 100)
```

---

## 🌍 Applications

| Application | How Kernel Used |
|-------------|-----------------|
| SVM | k(x, support_vectors) |
| Kernel PCA | Eigendecomposition of K |
| Gaussian Processes | Covariance function |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
