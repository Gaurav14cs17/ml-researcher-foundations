<!-- Navigation -->
<p align="center">
  <a href="../04_rkhs/">⬅️ Prev: RKHS</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../04_representation/">Next: Representation ➡️</a>
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

<img src="./images/svm-maximum-margin.svg" width="100%">

*Caption: SVM finds the hyperplane that maximizes the margin between classes. Support vectors (circled points) are the data points closest to the decision boundary. The kernel trick enables non-linear decision boundaries.*

---

## 📂 Overview

**Support Vector Machines (SVM)** find the optimal separating hyperplane that maximizes the margin between classes. The key insight: maximize distance to the nearest points (support vectors), which provides the best generalization.

---

## 📐 Mathematical Formulation

### Setting

Given training data \(\{(x_i, y_i)\}_{i=1}^n\) where \(x_i \in \mathbb{R}^d\) and \(y_i \in \{-1, +1\}\).

**Goal:** Find hyperplane \(w^\top x + b = 0\) that separates classes with maximum margin.

### Margin Definitions

**Functional Margin:** For point \((x_i, y_i)\):

$$\hat{\gamma}_i = y_i(w^\top x_i + b)$$

**Geometric Margin:** Distance from point to hyperplane:

$$\gamma_i = \frac{y_i(w^\top x_i + b)}{\|w\|}$$

**Margin of classifier:** Minimum distance to any training point:

$$\gamma = \min_{i=1,\ldots,n} \gamma_i$$

---

## 📐 Hard Margin SVM

### Primal Formulation

For linearly separable data:

$$\max_{w, b} \quad \gamma = \frac{\hat{\gamma}}{\|w\|}
\text{s.t.} \quad y_i(w^\top x_i + b) \geq \hat{\gamma} \quad \forall i$$

**Canonical form** (set \(\hat{\gamma} = 1\)):

$$\min_{w, b} \quad \frac{1}{2}\|w\|^2
\text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 \quad \forall i$$

**Interpretation:** Minimizing \(\|w\|\) maximizes margin \(\gamma = 1/\|w\|\).

### Lagrangian Formulation

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^n \alpha_i[y_i(w^\top x_i + b) - 1]$$

### KKT Conditions

**Stationarity:**

$$\nabla_w \mathcal{L} = w - \sum_{i=1}^n \alpha_i y_i x_i = 0 \implies \boxed{w = \sum_{i=1}^n \alpha_i y_i x_i}
\nabla_b \mathcal{L} = -\sum_{i=1}^n \alpha_i y_i = 0 \implies \boxed{\sum_{i=1}^n \alpha_i y_i = 0}$$

**Dual feasibility:** \(\alpha_i \geq 0\)

**Complementary slackness:** \(\alpha_i[y_i(w^\top x_i + b) - 1] = 0\)

### Dual Problem

Substituting KKT conditions into Lagrangian:

$$\max_\alpha \quad \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j=1}^n \alpha_i \alpha_j y_i y_j x_i^\top x_j
\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**Key insight:** Dual depends only on inner products \(x_i^\top x_j\) → kernel trick!

---

## 📐 Soft Margin SVM

### Motivation

Real data is rarely linearly separable. Allow some misclassifications with slack variables \(\xi_i\).

### Primal Formulation

$$\min_{w, b, \xi} \quad \frac{1}{2}\|w\|^2 + C\sum_{i=1}^n \xi_i
\text{s.t.} \quad y_i(w^\top x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad \forall i$$

**Interpretation:**
- \(C\) = regularization parameter (trade-off between margin and violations)

- \(\xi_i > 0\): point is within margin or misclassified

- \(\xi_i > 1\): point is misclassified

### Dual Formulation

$$\max_\alpha \quad \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j x_i^\top x_j
\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**Support Vectors:** Points with \(\alpha_i > 0\):

- \(0 < \alpha_i < C\): On margin boundary

- \(\alpha_i = C\): Inside margin or misclassified

---

## 📐 Kernel SVM

### Kernelized Dual

Replace inner products with kernel function:

$$\max_\alpha \quad \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j k(x_i, x_j)
\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

### Decision Function

$$f(x) = \text{sign}\left(\sum_{i=1}^n \alpha_i y_i k(x, x_i) + b\right)$$

**Only support vectors contribute** (points with \(\alpha_i > 0\)).

### Common Kernels

| Kernel | Formula | Use Case |
|--------|---------|----------|
| **Linear** | \(k(x,y) = x^\top y\) | Linearly separable |
| **Polynomial** | \(k(x,y) = (x^\top y + c)^d\) | Polynomial boundaries |
| **RBF** | \(k(x,y) = e^{-\gamma\|x-y\|^2}\) | General non-linear |
| **Sigmoid** | \(k(x,y) = \tanh(\alpha x^\top y + c)\) | Neural network-like |

---

## 📐 Theoretical Guarantees

### Margin-Based Generalization Bound

**Theorem:** With probability \(\geq 1 - \delta\):

$$R(h) \leq \hat{R}_\gamma(h) + \sqrt{\frac{c_1}{n}\left(\frac{R^2 \|w\|^2}{\gamma^2}\log n + \log\frac{1}{\delta}\right)}$$

where:

- \(\hat{R}_\gamma(h)\) = fraction of points with margin \(< \gamma\)

- \(R\) = radius of data

- Bound is independent of dimension!

### Structural Risk Minimization

SVM minimizes:

$$\text{Regularized Risk} = \underbrace{C\sum_i \max(0, 1 - y_i f(x_i))}_{\text{Hinge loss}} + \underbrace{\frac{1}{2}\|w\|^2}_{\text{Complexity penalty}}$$

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.optimize import minimize
from sklearn.svm import SVC

# ============================================================
# Hard Margin SVM (from scratch)
# ============================================================

class HardMarginSVM:
    """
    Hard margin SVM for linearly separable data.
    
    Primal: min ½||w||² s.t. yᵢ(w·xᵢ + b) ≥ 1
    Dual: max Σαᵢ - ½ΣΣαᵢαⱼyᵢyⱼxᵢ·xⱼ s.t. αᵢ≥0, Σαᵢyᵢ=0
    """
    
    def fit(self, X, y):
        n, d = X.shape
        
        # Compute Gram matrix
        K = X @ X.T
        
        # Dual objective (to minimize, so negate)
        def objective(alpha):
            return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y) * K) - np.sum(alpha)
        
        def gradient(alpha):
            return (alpha * y)[:, None] * (y * K).sum(axis=1) - 1
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda a: np.dot(a, y)}  # Σαᵢyᵢ = 0
        ]
        bounds = [(0, None) for _ in range(n)]  # αᵢ ≥ 0
        
        # Solve
        result = minimize(
            objective, 
            np.zeros(n), 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        self.alpha = result.x
        
        # Recover w = Σαᵢyᵢxᵢ
        self.w = np.sum((self.alpha * y)[:, None] * X, axis=0)
        
        # Find support vectors (αᵢ > ε)
        sv_idx = np.where(self.alpha > 1e-5)[0]
        
        # Compute b using any support vector
        # yᵢ(w·xᵢ + b) = 1 → b = yᵢ - w·xᵢ (for support vector)
        self.b = np.mean([y[i] - np.dot(self.w, X[i]) for i in sv_idx])
        
        return self
    
    def predict(self, X):
        return np.sign(X @ self.w + self.b)
    
    @property
    def margin(self):
        """Geometric margin = 1/||w||"""
        return 1 / np.linalg.norm(self.w)

# ============================================================
# Soft Margin Kernel SVM (SMO Algorithm)
# ============================================================

class SoftMarginSVM:
    """
    Soft margin SVM with kernel support.
    Uses simplified SMO optimization.
    """
    
    def __init__(self, C=1.0, kernel='rbf', gamma=1.0, degree=3, tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.tol = tol
    
    def _kernel(self, X, Y=None):
        if Y is None:
            Y = X
        
        if self.kernel == 'linear':
            return X @ Y.T
        elif self.kernel == 'rbf':
            X_sq = np.sum(X**2, axis=1).reshape(-1, 1)
            Y_sq = np.sum(Y**2, axis=1).reshape(1, -1)
            return np.exp(-self.gamma * (X_sq + Y_sq - 2 * X @ Y.T))
        elif self.kernel == 'poly':
            return (X @ Y.T + 1) ** self.degree
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        n = len(X)
        self.X = X
        self.y = y.astype(float)
        
        # Compute kernel matrix
        K = self._kernel(X)
        
        # Initialize
        self.alpha = np.zeros(n)
        self.b = 0
        
        # SMO main loop
        for _ in range(1000):
            for i in range(n):
                # Error for sample i
                E_i = self._decision(K[i]) - y[i]
                
                # Check KKT violation
                if (y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    # Select j randomly
                    j = np.random.choice([k for k in range(n) if k != i])
                    
                    E_j = self._decision(K[j]) - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    # Compute bounds
                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alpha[j] -= y[j] * (E_i - E_j) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    # Update alpha_i
                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])
                    
                    # Update bias
                    b1 = self.b - E_i - y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] \
                         - y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
        
        # Identify support vectors
        self.support_ = np.where(self.alpha > 1e-5)[0]
        
        return self
    
    def _decision(self, K_row):
        return np.sum(self.alpha * self.y * K_row) + self.b
    
    def decision_function(self, X):
        K = self._kernel(X, self.X)
        return np.array([self._decision(K[i]) for i in range(len(X))])
    
    def predict(self, X):
        return np.sign(self.decision_function(X))

# ============================================================
# Visualization
# ============================================================

def visualize_svm(X, y, svm, title="SVM Decision Boundary"):
    """Visualize SVM decision boundary and margins."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Decision function
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision regions
    ax.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=['red', 'blue'])
    
    # Plot decision boundary and margins
    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['red', 'black', 'blue'],
               linestyles=['--', '-', '--'], linewidths=[1.5, 2, 1.5])
    
    # Plot points
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=50, edgecolors='k', label='+1')
    ax.scatter(X[y == -1, 0], X[y == -1, 1], c='red', s=50, edgecolors='k', label='-1')
    
    # Highlight support vectors
    if hasattr(svm, 'support_'):
        ax.scatter(X[svm.support_, 0], X[svm.support_, 1], 
                   s=200, facecolors='none', edgecolors='green', linewidths=2,
                   label='Support Vectors')
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('svm_visualization.png', dpi=150)
    plt.show()

# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate linearly separable data
    print("=== Hard Margin SVM ===")
    n = 50
    X_pos = np.random.randn(n, 2) + np.array([2, 2])
    X_neg = np.random.randn(n, 2) + np.array([-2, -2])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*n + [-1]*n)
    
    svm_hard = HardMarginSVM()
    svm_hard.fit(X, y)
    print(f"Weight vector: {svm_hard.w}")
    print(f"Bias: {svm_hard.b:.4f}")
    print(f"Margin: {svm_hard.margin:.4f}")
    print(f"Accuracy: {np.mean(svm_hard.predict(X) == y):.2%}")
    
    # XOR problem (non-linear)
    print("\n=== Kernel SVM on XOR ===")
    X_xor = np.random.randn(100, 2)
    y_xor = np.sign(X_xor[:, 0] * X_xor[:, 1])
    
    svm_linear = SoftMarginSVM(kernel='linear', C=1.0)
    svm_linear.fit(X_xor, y_xor)
    
    svm_rbf = SoftMarginSVM(kernel='rbf', C=1.0, gamma=1.0)
    svm_rbf.fit(X_xor, y_xor)
    
    print(f"Linear kernel accuracy: {np.mean(svm_linear.predict(X_xor) == y_xor):.2%}")
    print(f"RBF kernel accuracy: {np.mean(svm_rbf.predict(X_xor) == y_xor):.2%}")
    print(f"Number of support vectors (RBF): {len(svm_rbf.support_)}")

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Original SVM Paper | Cortes & Vapnik (1995) |
| 📄 | SMO Algorithm | Platt (1998) |
| 📖 | Statistical Learning Theory | Vapnik (1998) |
| 📖 | Learning with Kernels | Schölkopf & Smola |

---

⬅️ [Back: RKHS](../04_rkhs/) | ➡️ [Back: Kernel Methods](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../04_rkhs/">⬅️ Prev: RKHS</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../04_representation/">Next: Representation ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
