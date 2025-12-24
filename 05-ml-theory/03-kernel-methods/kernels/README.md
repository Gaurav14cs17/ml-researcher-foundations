<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Kernels&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Kernel Functions

> **Similarity measures in feature space**

---

## 🎯 Visual Overview

<img src="./images/kernels.svg" width="100%">

*Caption: A kernel k(x,x') computes inner products in feature space without explicitly computing features: k(x,x') = ⟨φ(x), φ(x')⟩. Common kernels: Linear, Polynomial, RBF (Gaussian).*

---

## 📂 Overview

Kernels enable working in high-dimensional feature spaces efficiently. They're the foundation of SVMs, Gaussian processes, and many classical ML methods.

---

## 📐 Mathematical Definition

### The Kernel Trick

```
Kernel k(x, x') = ⟨φ(x), φ(x')⟩

Instead of:
1. Map x → φ(x) in high-D space
2. Compute inner product

Just compute:
k(x, x') directly (often cheaper!)

Example: φ(x) might be ∞-dimensional, but k is finite!
```

### Mercer's Condition

```
A function k(x, x') is a valid kernel if and only if
for any set of points {x₁, ..., xₙ}, the kernel matrix K is
positive semi-definite:

K_{ij} = k(xᵢ, xⱼ)

∀v: v^T K v ≥ 0
```

---

## 📊 Common Kernels

| Kernel | Formula | Feature Space |
|--------|---------|---------------|
| **Linear** | k(x,x') = x^T x' | Original space |
| **Polynomial** | k(x,x') = (x^T x' + c)^d | Polynomial features |
| **RBF/Gaussian** | k(x,x') = exp(-γ\|\|x-x'\|\|²) | Infinite-dimensional |
| **Laplacian** | k(x,x') = exp(-γ\|\|x-x'\|\|₁) | Infinite-dimensional |
| **Sigmoid** | k(x,x') = tanh(αx^T x' + c) | Neural network-like |

### RBF Kernel (Most Popular)

```
k(x, x') = exp(-γ ||x - x'||²)
         = exp(-||x - x'||² / (2σ²))

Where:
    γ = 1/(2σ²) controls width
    Large γ: Points must be very close to be similar
    Small γ: Points can be far apart and still similar

As γ → ∞: k(x,x') → δ(x-x')  (identity kernel)
As γ → 0: k(x,x') → 1        (all points similar)
```

### Polynomial Kernel

```
k(x, x') = (x^T x' + c)^d

For d=2, c=1:
k([x₁,x₂], [y₁,y₂]) = (x₁y₁ + x₂y₂ + 1)²
                    = 1 + 2x₁y₁ + 2x₂y₂ + 2x₁x₂y₁y₂ + x₁²y₁² + x₂²y₂²

Equivalent to feature map:
φ([x₁,x₂]) = [1, √2x₁, √2x₂, √2x₁x₂, x₁², x₂²]
```

---

## 🔑 Key Properties

| Property | Description | Formula |
|----------|-------------|---------|
| **Symmetry** | k(x,x') = k(x',x) | Order doesn't matter |
| **PSD** | Kernel matrix is positive semi-definite | v^TKv ≥ 0 |
| **Closure under sum** | k₁ + k₂ is a kernel | Combine kernels |
| **Closure under product** | k₁ × k₂ is a kernel | Combine kernels |
| **Closure under scaling** | ck is a kernel (c > 0) | Scale kernels |

### Kernel Composition

```
Given valid kernels k₁, k₂:

Addition:     k(x,x') = k₁(x,x') + k₂(x,x')  ✓ valid
Product:      k(x,x') = k₁(x,x') × k₂(x,x')  ✓ valid
Scaling:      k(x,x') = c × k₁(x,x')         ✓ valid (c > 0)
Exponent:     k(x,x') = exp(k₁(x,x'))        ✓ valid
Polynomial:   k(x,x') = (k₁(x,x') + c)^d     ✓ valid (c ≥ 0)
```

---

## 💻 Code Examples

### Implementing Kernels

```python
import numpy as np

def linear_kernel(x1, x2):
    """Linear kernel: k(x,x') = x^T x'"""
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3, coef0=1):
    """Polynomial kernel: k(x,x') = (x^T x' + c)^d"""
    return (np.dot(x1, x2) + coef0) ** degree

def rbf_kernel(x1, x2, gamma=1.0):
    """RBF (Gaussian) kernel: k(x,x') = exp(-γ||x-x'||²)"""
    diff = x1 - x2
    return np.exp(-gamma * np.dot(diff, diff))

def kernel_matrix(X, kernel_func, **kwargs):
    """Compute full kernel matrix"""
    n = len(X)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = kernel_func(X[i], X[j], **kwargs)
    return K


# Example
X = np.random.randn(5, 2)
K_linear = kernel_matrix(X, linear_kernel)
K_rbf = kernel_matrix(X, rbf_kernel, gamma=0.5)
print("Linear kernel matrix:\n", K_linear)
print("RBF kernel matrix:\n", K_rbf)
```

### Using Scikit-Learn

```python
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel

# SVM with different kernels
svm_linear = SVC(kernel='linear')
svm_rbf = SVC(kernel='rbf', gamma=0.5)
svm_poly = SVC(kernel='poly', degree=3)

# Custom kernel
def my_kernel(X, Y):
    return rbf_kernel(X, Y, gamma=0.5) + 0.1 * polynomial_kernel(X, Y, degree=2)

svm_custom = SVC(kernel=my_kernel)

# Gaussian Process with kernel
gp = GaussianProcessClassifier(kernel=RBF(length_scale=1.0))

# Compute kernel matrix directly
X = np.random.randn(100, 10)
K = rbf_kernel(X, gamma=0.5)
print(f"Kernel matrix shape: {K.shape}")
```

### Kernel PCA

```python
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
t = np.linspace(0, 2*np.pi, 200)
X = np.vstack([
    np.cos(t) + 0.1*np.random.randn(200),
    np.sin(t) + 0.1*np.random.randn(200)
]).T

# Linear PCA can't separate this
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# Kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=t)
axes[0].set_title('Original Data')
axes[1].scatter(X_kpca[:, 0], X_kpca[:, 1], c=t)
axes[1].set_title('Kernel PCA (RBF)')
plt.show()
```

---

## 📊 Choosing a Kernel

| Data Type | Recommended Kernel | Why |
|-----------|-------------------|-----|
| **Linear separable** | Linear | Simple, fast |
| **Non-linear** | RBF | Flexible, universal approximator |
| **Polynomial relationships** | Polynomial | Explicit polynomial features |
| **Sparse similarity** | Laplacian | Robust to outliers |
| **Periodic data** | Periodic | Captures cycles |
| **Unknown** | RBF | Good default |

### Hyperparameter Selection

```
RBF Kernel: γ (or σ) is critical

Too small γ (large σ): 
    - All points look similar
    - Underfitting
    
Too large γ (small σ):
    - Only very close points similar
    - Overfitting
    
Typically: Cross-validation over grid
    γ ∈ {10⁻³, 10⁻², 10⁻¹, 1, 10, 10², 10³}
```

---

## 🔗 Applications

| Method | How Kernels Are Used |
|--------|---------------------|
| **SVM** | Decision boundary in feature space |
| **Kernel PCA** | Non-linear dimensionality reduction |
| **Gaussian Processes** | Covariance function |
| **Kernel Ridge Regression** | Non-linear regression |
| **MMD** | Distribution comparison |

---

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Kernel Trick | [../kernel-trick/](../kernel-trick/) |
| 📖 | SVM | [../svm/](../svm/) |
| 📖 | Gaussian Processes | [../gaussian-processes/](../gaussian-processes/) |
| 📄 | Learning with Kernels | [Book](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) |
| 🇨🇳 | 核函数详解 | [知乎](https://zhuanlan.zhihu.com/p/24291579) |
| 🇨🇳 | RBF核参数选择 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88660612) |
| 🇨🇳 | 支持向量机核技巧 | [B站](https://www.bilibili.com/video/BV1Hb411w7FN) |
| 🇨🇳 | 核方法综述 | [机器之心](https://www.jiqizhixin.com/articles/2018-01-25-6)

---

⬅️ [Back: Kernel Methods](../)

---

⬅️ [Back: Kernel Trick](../kernel-trick/) | ➡️ [Next: Rkhs](../rkhs/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
