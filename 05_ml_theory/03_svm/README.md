<!-- Navigation -->
<p align="center">
  <a href="../02_generalization/">⬅️ Prev: Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_regularization/">Next: Regularization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Support%20Vector%20Machines&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/svm-kernel-complete.svg" width="100%">

*Caption: SVM finds the maximum-margin hyperplane separating classes.*

---

## 📂 Overview

**Support Vector Machines (SVM)** are powerful classifiers that find the optimal separating hyperplane by maximizing the margin between classes. Combined with the kernel trick, they can handle non-linear decision boundaries.

---

## 📐 Hard-Margin SVM

### Problem Formulation

For linearly separable data with labels \(y_i \in \{-1, +1\}\):

**Primal Problem:**

```math
\min_{w, b} \frac{1}{2}\|w\|^2
\text{s.t. } y_i(w^\top x_i + b) \geq 1, \quad \forall i

```

**Margin:** The geometric margin is \(\gamma = \frac{2}{\|w\|}\).

### Lagrangian

```math
\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i[y_i(w^\top x_i + b) - 1]

```

**KKT Conditions:**

```math
\nabla_w \mathcal{L} = 0 \Rightarrow w = \sum_i \alpha_i y_i x_i
\nabla_b \mathcal{L} = 0 \Rightarrow \sum_i \alpha_i y_i = 0

```

### Dual Problem

```math
\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j
\text{s.t. } \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0

```

**Key insight:** Only inner products \(x_i^\top x_j\) appear → kernel trick!

---

## 📐 Soft-Margin SVM

For non-separable data, introduce slack variables \(\xi_i \geq 0\):

**Primal Problem:**

```math
\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
\text{s.t. } y_i(w^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0

```

**Dual Problem:**

```math
\max_\alpha \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j
\text{s.t. } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0

```

**Interpretation of \(C\):**
- Large \(C\): Small margin, fewer violations

- Small \(C\): Large margin, more violations

---

## 📐 Kernel SVM

### Kernel Trick

Replace inner product with kernel function:

```math
x_i^\top x_j \to k(x_i, x_j) = \phi(x_i)^\top \phi(x_j)

```

**Decision Function:**

```math
f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i k(x_i, x) + b\right)

```

### Common Kernels

| Kernel | Formula | Feature Space |
|--------|---------|---------------|
| Linear | \(x^\top y\) | Original \(\mathbb{R}^d\) |
| Polynomial | \((x^\top y + c)^p\) | \(\binom{d+p}{p}\) dimensions |
| RBF | \(\exp(-\gamma\|x-y\|^2)\) | Infinite dimensional |

### RBF Kernel Properties

```math
k(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)

```

**Theorem:** The RBF kernel corresponds to an infinite-dimensional feature space. It is a universal approximator.

---

## 📐 Theoretical Results

### Generalization Bound

**Theorem:** For SVM with margin \(\gamma\) on data with radius \(R\):

```math
R(f) \leq \hat{R}(f) + O\left(\frac{R^2/\gamma^2}{n}\right)

```

**Implication:** Larger margin → better generalization.

### Support Vectors

**Theorem:** The solution only depends on support vectors (points with \(\alpha_i > 0\)).

For hard-margin: support vectors lie on the margin.
For soft-margin: \(\alpha_i = C\) indicates violation.

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.optimize import minimize

class SVM:
    """
    Support Vector Machine implementation.
    
    Dual: max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼk(xᵢ,xⱼ)
    s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        self.kernel_type = kernel
    
    def kernel(self, X1, X2):
        """Compute kernel matrix."""
        if self.kernel_type == 'linear':
            return X1 @ X2.T
        elif self.kernel_type == 'rbf':
            sq_dists = (
                np.sum(X1**2, axis=1).reshape(-1, 1) +
                np.sum(X2**2, axis=1) -
                2 * X1 @ X2.T
            )
            return np.exp(-self.gamma * sq_dists)
        elif self.kernel_type == 'poly':
            return (X1 @ X2.T + 1) ** 3
    
    def fit(self, X, y):
        n = len(y)
        self.X = X
        self.y = y
        
        # Compute kernel matrix
        K = self.kernel(X, X)
        
        # Objective: -Σαᵢ + ½ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)
        def objective(alpha):
            return 0.5 * alpha @ (K * np.outer(y, y)) @ alpha - np.sum(alpha)
        
        def gradient(alpha):
            return (K * np.outer(y, y)) @ alpha - np.ones(n)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda a: np.dot(a, y)}  # Σαᵢyᵢ = 0
        ]
        bounds = [(0, self.C) for _ in range(n)]
        
        # Optimize
        result = minimize(
            objective, np.zeros(n), jac=gradient,
            bounds=bounds, constraints=constraints,
            method='SLSQP'
        )
        
        self.alpha = result.x
        
        # Find support vectors (α > 0)
        sv_mask = self.alpha > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.support_alpha = self.alpha[sv_mask]
        
        # Compute bias
        # For support vectors on margin: yᵢ(Σαⱼyⱼk(xⱼ,xᵢ) + b) = 1
        sv_on_margin = (self.alpha > 1e-5) & (self.alpha < self.C - 1e-5)
        if np.any(sv_on_margin):
            idx = np.where(sv_on_margin)[0][0]
            K_sv = self.kernel(self.support_vectors, X[idx:idx+1])
            self.b = y[idx] - np.sum(self.support_alpha * self.support_labels * K_sv.ravel())
        else:
            self.b = 0
        
        return self
    
    def decision_function(self, X):
        """f(x) = Σαᵢyᵢk(xᵢ,x) + b"""
        K = self.kernel(self.support_vectors, X)
        return np.sum(self.support_alpha[:, None] * self.support_labels[:, None] * K, axis=0) + self.b
    
    def predict(self, X):
        return np.sign(self.decision_function(X))
    
    @property
    def margin(self):
        """Margin = 2/||w|| (for linear kernel)."""
        if self.kernel_type == 'linear':
            w = np.sum(self.support_alpha[:, None] * self.support_labels[:, None] * self.support_vectors, axis=0)
            return 2 / np.linalg.norm(w)
        return None

class SMO:
    """
    Sequential Minimal Optimization for SVM.
    Platt's algorithm for efficient training.
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, tol=1e-3, max_iter=1000):
        self.C = C
        self.gamma = gamma
        self.kernel_type = kernel
        self.tol = tol
        self.max_iter = max_iter
    
    def kernel(self, x1, x2):
        if self.kernel_type == 'rbf':
            return np.exp(-self.gamma * np.sum((x1 - x2) ** 2))
        return np.dot(x1, x2)
    
    def fit(self, X, y):
        n = len(y)
        self.X, self.y = X, y
        self.alpha = np.zeros(n)
        self.b = 0
        
        # Cache for error
        self.E = -y.copy().astype(float)
        
        for _ in range(self.max_iter):
            changed = False
            for i in range(n):
                if self._should_optimize(i):
                    j = self._select_second(i)
                    if self._optimize_pair(i, j):
                        changed = True
            
            if not changed:
                break
        
        # Store support vectors
        sv_mask = self.alpha > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_labels = y[sv_mask]
        self.support_alpha = self.alpha[sv_mask]
        
        return self
    
    def _should_optimize(self, i):
        """Check KKT conditions."""
        r = self.E[i] * self.y[i]
        return (r < -self.tol and self.alpha[i] < self.C) or \
               (r > self.tol and self.alpha[i] > 0)
    
    def _select_second(self, i):
        """Heuristic: maximize |E_i - E_j|."""
        if self.E[i] > 0:
            return np.argmin(self.E)
        return np.argmax(self.E)
    
    def _optimize_pair(self, i, j):
        """Optimize alpha_i and alpha_j."""
        if i == j:
            return False
        
        # Compute bounds
        if self.y[i] != self.y[j]:
            L = max(0, self.alpha[j] - self.alpha[i])
            H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
        else:
            L = max(0, self.alpha[i] + self.alpha[j] - self.C)
            H = min(self.C, self.alpha[i] + self.alpha[j])
        
        if L >= H:
            return False
        
        # Compute eta
        k_ii = self.kernel(self.X[i], self.X[i])
        k_jj = self.kernel(self.X[j], self.X[j])
        k_ij = self.kernel(self.X[i], self.X[j])
        eta = 2 * k_ij - k_ii - k_jj
        
        if eta >= 0:
            return False
        
        # Update alpha_j
        alpha_j_new = self.alpha[j] - self.y[j] * (self.E[i] - self.E[j]) / eta
        alpha_j_new = np.clip(alpha_j_new, L, H)
        
        if abs(alpha_j_new - self.alpha[j]) < 1e-5:
            return False
        
        # Update alpha_i
        alpha_i_new = self.alpha[i] + self.y[i] * self.y[j] * (self.alpha[j] - alpha_j_new)
        
        # Update bias
        b1 = self.b - self.E[i] - self.y[i] * (alpha_i_new - self.alpha[i]) * k_ii \
             - self.y[j] * (alpha_j_new - self.alpha[j]) * k_ij
        b2 = self.b - self.E[j] - self.y[i] * (alpha_i_new - self.alpha[i]) * k_ij \
             - self.y[j] * (alpha_j_new - self.alpha[j]) * k_jj
        
        if 0 < alpha_i_new < self.C:
            self.b = b1
        elif 0 < alpha_j_new < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # Update alpha
        self.alpha[i] = alpha_i_new
        self.alpha[j] = alpha_j_new
        
        # Update error cache
        for k in range(len(self.y)):
            self.E[k] = self._decision(self.X[k]) - self.y[k]
        
        return True
    
    def _decision(self, x):
        return np.sum(self.alpha * self.y * 
                      np.array([self.kernel(x, xi) for xi in self.X])) + self.b
    
    def predict(self, X):
        return np.sign([self._decision(x) for x in X])

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Statistical Learning Theory | [Vapnik](https://www.springer.com/gp/book/9780387987804) |
| 📄 | SMO Algorithm | [Platt](https://www.microsoft.com/en-us/research/publication/sequential-minimal-optimization-a-fast-algorithm-for-training-support-vector-machines/) |
| 📖 | Learning with Kernels | [Schölkopf & Smola](https://mitpress.mit.edu/9780262536578/) |

---

⬅️ [Back: Generalization](../02_generalization/) | ➡️ [Next: Representation](../04_representation/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../02_generalization/">⬅️ Prev: Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_regularization/">Next: Regularization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
