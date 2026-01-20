<!-- Navigation -->
<p align="center">
  <a href="../03_kernel_trick/">⬅️ Prev: Kernel Trick</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_svm/">Next: SVM ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=RKHS%20Theory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/rkhs.svg" width="100%">

*Caption: RKHS is a Hilbert space with the reproducing property: f(x) = ⟨f, k(·,x)⟩. Every kernel defines a unique RKHS. The representer theorem shows solutions are linear combinations of kernel evaluations.*

---

## 📂 Overview

**Reproducing Kernel Hilbert Spaces (RKHS)** provide the rigorous mathematical foundation for kernel methods. They connect function spaces, kernels, regularization, and optimization in an elegant theoretical framework.

---

## 📐 Hilbert Space Background

### Definition: Hilbert Space

A **Hilbert space** \(\mathcal{H}\) is a complete inner product space:

1. **Inner product:** \(\langle \cdot, \cdot \rangle_\mathcal{H}: \mathcal{H} \times \mathcal{H} \to \mathbb{R}\)
2. **Norm induced by inner product:** \(\|f\|_\mathcal{H} = \sqrt{\langle f, f \rangle_\mathcal{H}}\)
3. **Complete:** Every Cauchy sequence converges in \(\mathcal{H}\)

### Properties

- **Cauchy-Schwarz:** \(|\langle f, g \rangle| \leq \|f\| \cdot \|g\|\)
- **Parallelogram law:** \(\|f + g\|^2 + \|f - g\|^2 = 2\|f\|^2 + 2\|g\|^2\)
- **Riesz representation:** Every continuous linear functional has form \(\ell(f) = \langle f, g \rangle\) for unique \(g\)

---

## 📐 Reproducing Kernel Hilbert Space

### Definition

A Hilbert space \(\mathcal{H}\) of functions \(f: \mathcal{X} \to \mathbb{R}\) is an **RKHS** if there exists a kernel \(k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}\) such that:

1. **Kernel function in space:** \(\forall x \in \mathcal{X}: k(\cdot, x) \in \mathcal{H}\)

2. **Reproducing property:**
```math
\forall f \in \mathcal{H}, \forall x \in \mathcal{X}: f(x) = \langle f, k(\cdot, x) \rangle_\mathcal{H}
```

### Consequence of Reproducing Property

The kernel computes inner products:
```math
k(x, x') = \langle k(\cdot, x), k(\cdot, x') \rangle_\mathcal{H}
```

**Proof:**
```math
k(x, x') = \langle k(\cdot, x'), k(\cdot, x) \rangle_\mathcal{H} \quad \text{(reproducing property with } f = k(\cdot, x') \text{)} \quad \blacksquare
```

---

## 📐 Moore-Aronszajn Theorem

### Statement

**Theorem:** For every positive semi-definite kernel \(k\), there exists a **unique** RKHS \(\mathcal{H}_k\) with \(k\) as its reproducing kernel.

### Construction (Proof Sketch)

1. **Define feature map:** \(\phi(x) = k(\cdot, x)\)

2. **Define pre-Hilbert space:**
```math
\mathcal{H}_0 = \left\{ f = \sum_{i=1}^n \alpha_i k(\cdot, x_i) : n \in \mathbb{N}, \alpha_i \in \mathbb{R}, x_i \in \mathcal{X} \right\}
```

3. **Define inner product:**
```math
\left\langle \sum_i \alpha_i k(\cdot, x_i), \sum_j \beta_j k(\cdot, y_j) \right\rangle = \sum_{i,j} \alpha_i \beta_j k(x_i, y_j)
```

4. **Verify well-defined:** Uses positive semi-definiteness of \(k\)

5. **Complete the space:** \(\mathcal{H}_k\) is the completion of \(\mathcal{H}_0\)

6. **Verify reproducing property:**
```math
\langle f, k(\cdot, x) \rangle = \left\langle \sum_i \alpha_i k(\cdot, x_i), k(\cdot, x) \right\rangle = \sum_i \alpha_i k(x_i, x) = f(x) \quad \blacksquare
```

---

## 📐 The Representer Theorem

### Statement

**Theorem (Kimeldorf & Wahba, 1971; Schölkopf et al., 2001):**

Consider the regularized empirical risk minimization:

```math
\min_{f \in \mathcal{H}_k} \left[ \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \|f\|_{\mathcal{H}_k}^2 \right]
```

where \(L\) is any loss function and \(\lambda > 0\).

**Then the minimizer has the form:**

```math
f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i)
```

for some \(\alpha_1, \ldots, \alpha_n \in \mathbb{R}\).

### Proof

**Step 1:** Decompose any \(f \in \mathcal{H}_k\):

Let \(\mathcal{S} = \text{span}\{k(\cdot, x_1), \ldots, k(\cdot, x_n)\}\).

For any \(f \in \mathcal{H}_k\), write \(f = f_\mathcal{S} + f_\perp\) where:
- \(f_\mathcal{S} \in \mathcal{S}\)
- \(f_\perp \perp \mathcal{S}\)

**Step 2:** Show \(f_\perp\) doesn't affect data fit:

For training point \(x_i\), using reproducing property:
```math
f(x_i) = \langle f, k(\cdot, x_i) \rangle = \langle f_\mathcal{S} + f_\perp, k(\cdot, x_i) \rangle = \langle f_\mathcal{S}, k(\cdot, x_i) \rangle = f_\mathcal{S}(x_i)
```

since \(\langle f_\perp, k(\cdot, x_i) \rangle = 0\) by orthogonality.

**Step 3:** Show \(f_\perp\) only increases regularization:

By Pythagorean theorem:
```math
\|f\|^2 = \|f_\mathcal{S}\|^2 + \|f_\perp\|^2 \geq \|f_\mathcal{S}\|^2
```

**Step 4:** Conclude:

The loss term depends only on \(f_\mathcal{S}\), and the regularizer is minimized when \(f_\perp = 0\).

Therefore, optimal \(f^* = f_\mathcal{S}^* \in \mathcal{S}\), which has form:
```math
f^*(x) = \sum_{i=1}^n \alpha_i k(x, x_i) \quad \blacksquare
```

### Implications

The Representer Theorem reduces infinite-dimensional optimization to finite-dimensional:

**Before:** Optimize over all functions in \(\mathcal{H}_k\) (infinite-dimensional)

**After:** Optimize over \(n\) coefficients \(\alpha \in \mathbb{R}^n\)

---

## 📐 RKHS Norm and Smoothness

### Norm Interpretation

For \(f = \sum_i \alpha_i k(\cdot, x_i)\):

```math
\|f\|_{\mathcal{H}_k}^2 = \sum_{i,j} \alpha_i \alpha_j k(x_i, x_j) = \alpha^\top K \alpha
```

where \(K_{ij} = k(x_i, x_j)\) is the Gram matrix.

### Smoothness Control

The RKHS norm controls smoothness of the function:

- **Small \(\|f\|_{\mathcal{H}}\):** Smooth, slowly varying function
- **Large \(\|f\|_{\mathcal{H}}\):** Can be wiggly, rapidly changing

**For RBF kernel** \(k(x, x') = \exp(-\gamma\|x-x'\|^2)\):

```math
\|f\|_{\mathcal{H}}^2 = \int \int f(x) k^{-1}(x, x') f(x') dx dx'
```

where \(k^{-1}\) penalizes high-frequency components.

---

## 📐 Kernel Ridge Regression

### Problem

Given data \(\{(x_i, y_i)\}_{i=1}^n\), solve:

```math
\min_{f \in \mathcal{H}_k} \frac{1}{n}\sum_{i=1}^n (f(x_i) - y_i)^2 + \lambda \|f\|_{\mathcal{H}_k}^2
```

### Solution via Representer Theorem

By representer theorem, \(f^*(x) = \sum_i \alpha_i k(x, x_i)\).

Let \(K\) be the Gram matrix and \(\mathbf{f} = K\alpha\).

Objective becomes:
```math
\frac{1}{n}\|K\alpha - y\|^2 + \lambda \alpha^\top K \alpha
```

Taking derivative and setting to zero:
```math
\frac{2}{n} K(K\alpha - y) + 2\lambda K\alpha = 0
K\alpha + n\lambda\alpha = y
(K + n\lambda I)\alpha = y
```

**Solution:**
```math
\boxed{\alpha^* = (K + n\lambda I)^{-1} y}
```

**Prediction:**
```math
f^*(x) = k_x^\top (K + n\lambda I)^{-1} y
```

where \(k_x = [k(x, x_1), \ldots, k(x, x_n)]^\top\).

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class RKHS:
    """
    Reproducing Kernel Hilbert Space implementation.
    
    Key properties:
    - Reproducing property: f(x) = ⟨f, k(·,x)⟩
    - Representer theorem: f* = Σ αᵢ k(·,xᵢ)
    - Kernel trick: ⟨φ(x), φ(x')⟩ = k(x, x')
    """
    
    @staticmethod
    def rbf_kernel(X, Y=None, gamma=1.0):
        """
        RBF kernel: k(x,x') = exp(-γ||x-x'||²)
        
        This kernel induces an infinite-dimensional RKHS!
        """
        if Y is None:
            Y = X
        X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
        Y_sq = np.sum(Y**2, axis=1).reshape(1, -1)
        sq_dists = X_sq + Y_sq - 2 * X @ Y.T
        return np.exp(-gamma * np.maximum(sq_dists, 0))
    
    @staticmethod
    def polynomial_kernel(X, Y=None, degree=3, coef0=1):
        """Polynomial kernel: k(x,x') = (x·x' + c)^d"""
        if Y is None:
            Y = X
        return (X @ Y.T + coef0) ** degree
    
    @staticmethod  
    def linear_kernel(X, Y=None):
        """Linear kernel: k(x,x') = x·x'"""
        if Y is None:
            Y = X
        return X @ Y.T

class KernelRidgeRegression:
    """
    Kernel Ridge Regression in RKHS.
    
    Solves: min_f (1/n)Σ(f(xᵢ) - yᵢ)² + λ||f||²_H
    
    By representer theorem: f*(x) = Σ αᵢ k(x, xᵢ)
    Solution: α* = (K + nλI)⁻¹ y
    """
    
    def __init__(self, kernel='rbf', gamma=1.0, alpha=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.alpha = alpha  # λ in the formulation
        
    def _compute_kernel(self, X, Y=None):
        if self.kernel == 'rbf':
            return RKHS.rbf_kernel(X, Y, self.gamma)
        elif self.kernel == 'linear':
            return RKHS.linear_kernel(X, Y)
        elif self.kernel == 'poly':
            return RKHS.polynomial_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Fit kernel ridge regression.
        
        α* = (K + nλI)⁻¹ y
        """
        self.X_train = X
        n = len(X)
        
        # Compute Gram matrix
        K = self._compute_kernel(X)
        
        # Solve (K + nλI)α = y
        # Using Cholesky decomposition for numerical stability
        A = K + n * self.alpha * np.eye(n)
        
        try:
            # Cholesky: A = LLᵀ, solve Lz = y, then Lᵀα = z
            c, lower = cho_factor(A)
            self.alpha_coef = cho_solve((c, lower), y)
        except np.linalg.LinAlgError:
            # Fall back to direct solve if not positive definite
            self.alpha_coef = np.linalg.solve(A, y)
        
        return self
    
    def predict(self, X):
        """
        Predict using representer theorem.
        
        f*(x) = k_x^T α = Σᵢ αᵢ k(x, xᵢ)
        """
        K_test = self._compute_kernel(X, self.X_train)
        return K_test @ self.alpha_coef
    
    def rkhs_norm(self):
        """
        Compute RKHS norm of learned function.
        
        ||f||²_H = α^T K α
        """
        K = self._compute_kernel(self.X_train)
        return np.sqrt(self.alpha_coef @ K @ self.alpha_coef)

class KernelSVM:
    """
    Support Vector Machine using RKHS formulation.
    
    Dual: max Σαᵢ - ½ΣΣ αᵢαⱼyᵢyⱼk(xᵢ,xⱼ)
         s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0
    
    Decision: f(x) = Σᵢ αᵢyᵢk(x,xᵢ) + b
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
    def _compute_kernel(self, X, Y=None):
        if self.kernel == 'rbf':
            return RKHS.rbf_kernel(X, Y, self.gamma)
        else:
            return RKHS.linear_kernel(X, Y)
    
    def fit(self, X, y):
        """Simplified SMO algorithm."""
        from scipy.optimize import minimize
        
        n = len(X)
        self.X_train = X
        self.y_train = y.astype(float)
        
        K = self._compute_kernel(X)
        
        # Dual objective (to maximize, so we minimize negative)
        def objective(alpha):
            return -np.sum(alpha) + 0.5 * np.sum(
                np.outer(alpha * y, alpha * y) * K
            )
        
        # Gradient
        def gradient(alpha):
            return -np.ones(n) + (alpha * y)[:, None] * (y * K).sum(axis=1)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda a: np.dot(a, y)}  # Σαᵢyᵢ = 0
        ]
        bounds = [(0, self.C) for _ in range(n)]
        
        result = minimize(objective, np.zeros(n), jac=gradient,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.alpha = result.x
        
        # Find support vectors
        sv_idx = np.where(self.alpha > 1e-5)[0]
        
        # Compute bias
        if len(sv_idx) > 0:
            margin_sv = sv_idx[np.where((self.alpha[sv_idx] > 1e-5) & 
                                         (self.alpha[sv_idx] < self.C - 1e-5))[0]]
            if len(margin_sv) > 0:
                s = margin_sv[0]
                self.b = y[s] - np.sum(self.alpha * y * K[s])
            else:
                self.b = 0
        else:
            self.b = 0
            
        return self
    
    def decision_function(self, X):
        K = self._compute_kernel(X, self.X_train)
        return K @ (self.alpha * self.y_train) + self.b
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

# Demonstration
if __name__ == "__main__":
    np.random.seed(42)
    
    print("=== Kernel Ridge Regression in RKHS ===\n")
    
    # Generate nonlinear data
    n = 100
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y_true = np.sin(X.ravel()) + 0.5 * np.sin(3 * X.ravel())
    y = y_true + 0.1 * np.random.randn(n)
    
    # Fit kernel ridge regression
    krr = KernelRidgeRegression(kernel='rbf', gamma=1.0, alpha=0.01)
    krr.fit(X, y)
    y_pred = krr.predict(X)
    
    print(f"RKHS norm of solution: {krr.rkhs_norm():.4f}")
    print(f"MSE: {np.mean((y_pred - y_true)**2):.6f}")
    
    # Compare different regularization strengths
    print("\n=== Effect of Regularization ===")
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        krr = KernelRidgeRegression(kernel='rbf', gamma=1.0, alpha=alpha)
        krr.fit(X, y)
        mse = np.mean((krr.predict(X) - y_true)**2)
        norm = krr.rkhs_norm()
        print(f"λ={alpha:.3f}: RKHS norm = {norm:.4f}, MSE = {mse:.6f}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Learning with Kernels | [Schölkopf & Smola](https://mitpress.mit.edu/9780262536578/learning-with-kernels/) |
| 📖 | PRML Ch. 6 | [Bishop](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | Representer Theorem | [Schölkopf et al., 2001](https://link.springer.com/chapter/10.1007/3-540-44581-1_27) |
| 📄 | Kernel Methods Review | [Hofmann et al., 2008](https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-3/Kernel-methods-in-machine-learning/10.1214/009053607000000677.full) |

---

⬅️ [Back: Kernels](../02_kernels/) | ➡️ [Next: SVM](../05_svm/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../03_kernel_trick/">⬅️ Prev: Kernel Trick</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_svm/">Next: SVM ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
