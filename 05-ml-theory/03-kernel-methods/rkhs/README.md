<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Rkhs&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Reproducing Kernel Hilbert Space (RKHS)

> **The mathematical foundation of kernel methods**

---

## 🎯 Visual Overview

<img src="./images/rkhs.svg" width="100%">

*Caption: RKHS is a Hilbert space with the reproducing property: f(x) = ⟨f, k(·,x)⟩. Every kernel defines a unique RKHS. The representer theorem shows solutions are linear combinations of kernel evaluations at data points.*

---

## 📂 Overview

RKHS provides the mathematical foundation for kernel methods. It connects function spaces, kernels, and optimization in a elegant theoretical framework.

---

## 📐 Mathematical Definitions

### RKHS Definition
```
H is an RKHS with kernel k if:
1. H is a Hilbert space of functions f: X → ℝ
2. k(·, x) ∈ H for all x
3. Reproducing property: f(x) = ⟨f, k(·,x)⟩_H

⟨f, g⟩_H = inner product in H
```

### Kernel and Feature Map
```
k(x, x') = ⟨φ(x), φ(x')⟩

φ: X → H is the feature map
Kernel computes inner product in feature space

Moore-Aronszajn: Every PSD kernel defines unique RKHS
```

### Representer Theorem
```
For optimization:
min_{f∈H} [Σᵢ L(yᵢ, f(xᵢ)) + λ||f||²_H]

Solution has form:
f*(x) = Σᵢ αᵢ k(xᵢ, x)

Only need to store n coefficients αᵢ!
```

### RKHS Norm
```
||f||²_H controls smoothness

For RBF kernel:
Small ||f||_H → smooth functions
Large ||f||_H → can be wiggly

Regularization: λ||f||²_H penalizes non-smooth f
```

### Kernel Ridge Regression
```
α* = (K + λI)⁻¹y

Where K_ij = k(xᵢ, xⱼ)

Predictions:
f*(x) = Σᵢ α*ᵢ k(xᵢ, x) = k_x^T (K + λI)⁻¹ y
```

---

## 💻 Code Examples

```python
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF

# Kernel Ridge Regression
krr = KernelRidge(kernel='rbf', alpha=1.0, gamma=0.1)
krr.fit(X_train, y_train)
y_pred = krr.predict(X_test)

# Manual implementation
def rbf_kernel(X1, X2, gamma=1.0):
    """k(x,x') = exp(-γ||x-x'||²)"""
    dists = np.sum(X1**2, axis=1, keepdims=True) \
          + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
    return np.exp(-gamma * dists)

def kernel_ridge(X, y, X_test, kernel_fn, lambda_reg=1.0):
    """f*(x) = k_x^T (K + λI)^{-1} y"""
    K = kernel_fn(X, X)
    K_test = kernel_fn(X_test, X)
    alpha = np.linalg.solve(K + lambda_reg * np.eye(len(K)), y)
    return K_test @ alpha

# RKHS norm approximation
def rkhs_norm_squared(alpha, K):
    """||f||²_H = α^T K α for f = Σαᵢk(·,xᵢ)"""
    return alpha @ K @ alpha

# Kernel SVM
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C=1.0)  # Uses dual formulation in RKHS
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Scholkopf: Learning with Kernels | [Book](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) |
| 📖 | Bishop PRML Ch. 6 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | Representer Theorem | [Paper](https://link.springer.com/chapter/10.1007/3-540-44581-1_27) |
| 🇨🇳 | RKHS详解 | [知乎](https://zhuanlan.zhihu.com/p/29527729) |
| 🇨🇳 | 核方法理论 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 核函数与SVM | [B站](https://www.bilibili.com/video/BV164411b7dx) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: rkhs](../)

---

⬅️ [Back: Kernels](../kernels/) | ➡️ [Next: Svm](../svm/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
