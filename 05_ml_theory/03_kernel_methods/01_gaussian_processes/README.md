<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_kernels/">Next: Kernels ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Gaussian%20Processes&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/gaussian-processes.svg" width="100%">

*Caption: Gaussian Processes define distributions over functions with built-in uncertainty quantification.*

---

## 📂 Overview

**Gaussian Processes (GP)** are powerful non-parametric Bayesian models that define distributions over functions. They provide uncertainty estimates naturally and are widely used for regression, Bayesian optimization, and as priors in deep learning.

---

## 📐 Mathematical Definition

### Gaussian Process

**Definition:** A Gaussian Process is a collection of random variables, any finite subset of which has a joint Gaussian distribution.

```math
f \sim \mathcal{GP}(m(x), k(x, x'))
```

where:
- \(m(x) = \mathbb{E}[f(x)]\) is the mean function (often \(m(x) = 0\))
- \(k(x, x') = \text{Cov}(f(x), f(x'))\) is the covariance (kernel) function

**Property:** For any finite set \(\{x_1, \ldots, x_n\}\):

```math
\begin{bmatrix} f(x_1) \\ \vdots \\ f(x_n) \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} m(x_1) \\ \vdots \\ m(x_n) \end{bmatrix}, \begin{bmatrix} k(x_1, x_1) & \cdots & k(x_1, x_n) \\ \vdots & \ddots & \vdots \\ k(x_n, x_1) & \cdots & k(x_n, x_n) \end{bmatrix}\right)
```

---

## 📐 GP Regression

### Model

```math
y = f(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)
```

**Prior:** \(f \sim \mathcal{GP}(0, k)\)

**Joint Distribution:**

```math
\begin{bmatrix} \mathbf{f} \\ \mathbf{f}_* \end{bmatrix} \sim \mathcal{N}\left(\mathbf{0}, \begin{bmatrix} K & K_* \\ K_*^\top & K_{**} \end{bmatrix}\right)
```

where \(K = k(X, X)\), \(K_* = k(X, X_*)\), \(K_{**} = k(X_*, X_*)\).

### Posterior Predictive Distribution

**Theorem:** Given training data \((X, \mathbf{y})\) and test points \(X_*\):

```math
\mathbf{f}_* | X_*, X, \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \boldsymbol{\Sigma}_*)
\boldsymbol{\mu}_* = K_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}
\boldsymbol{\Sigma}_* = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*
```

**Proof:** By properties of Gaussian conditionals:

For \(\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix} \sim \mathcal{N}\left(\begin{bmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{bmatrix}, \begin{bmatrix} A & C \\ C^\top & B \end{bmatrix}\right)\):

```math
\mathbf{a} | \mathbf{b} \sim \mathcal{N}(\boldsymbol{\mu}_a + CB^{-1}(\mathbf{b} - \boldsymbol{\mu}_b), A - CB^{-1}C^\top)
```

Applying with observation noise \(\mathbf{y} = \mathbf{f} + \boldsymbol{\varepsilon}\). \(\blacksquare\)

---

## 📐 Marginal Likelihood

### Log Marginal Likelihood

```math
\log p(\mathbf{y}|X) = -\frac{1}{2}\mathbf{y}^\top(K + \sigma_n^2 I)^{-1}\mathbf{y} - \frac{1}{2}\log|K + \sigma_n^2 I| - \frac{n}{2}\log(2\pi)
```

**Interpretation:**
- **Data fit:** \(-\frac{1}{2}\mathbf{y}^\top(K + \sigma_n^2 I)^{-1}\mathbf{y}\)
- **Complexity penalty:** \(-\frac{1}{2}\log|K + \sigma_n^2 I|\)
- **Normalization:** \(-\frac{n}{2}\log(2\pi)\)

### Hyperparameter Optimization

Optimize kernel hyperparameters \(\boldsymbol{\theta}\) by maximizing log marginal likelihood:

```math
\frac{\partial \log p(\mathbf{y}|X, \boldsymbol{\theta})}{\partial \theta_j} = \frac{1}{2}\text{tr}\left((\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - K^{-1})\frac{\partial K}{\partial \theta_j}\right)
```

where \(\boldsymbol{\alpha} = (K + \sigma_n^2 I)^{-1}\mathbf{y}\).

---

## 📐 Common Kernels

### RBF (Squared Exponential)

```math
k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x - x'\|^2}{2\ell^2}\right)
```

- \(\sigma_f^2\): signal variance (output scale)
- \(\ell\): length scale (smoothness)

### Matérn

```math
k_\nu(r) = \sigma_f^2 \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}r}{\ell}\right)
```

Special cases:
- \(\nu = 1/2\): Exponential (Ornstein-Uhlenbeck)
- \(\nu = 3/2\): Once differentiable
- \(\nu = 5/2\): Twice differentiable
- \(\nu \to \infty\): RBF

### Periodic

```math
k(x, x') = \sigma_f^2 \exp\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)
```

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

class GaussianProcess:
    """
    Gaussian Process Regression.
    
    f ~ GP(0, k(x, x'))
    y = f(x) + ε, ε ~ N(0, σ_n²)
    
    Posterior: f* | X*, X, y ~ N(μ*, Σ*)
    """
    
    def __init__(self, kernel='rbf', length_scale=1.0, signal_var=1.0, noise_var=0.1):
        self.length_scale = length_scale
        self.signal_var = signal_var
        self.noise_var = noise_var
        self.kernel = kernel
    
    def _rbf_kernel(self, X1, X2):
        """RBF kernel: k(x,x') = σ_f² exp(-||x-x'||²/(2ℓ²))"""
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                  np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return self.signal_var * np.exp(-sq_dist / (2 * self.length_scale**2))
    
    def fit(self, X, y):
        """Fit GP by computing Cholesky decomposition."""
        self.X_train = X
        self.y_train = y
        n = len(X)
        
        # Compute kernel matrix with noise
        K = self._rbf_kernel(X, X) + self.noise_var * np.eye(n)
        
        # Cholesky decomposition for stable inversion
        self.L, self.lower = cho_factor(K)
        self.alpha = cho_solve((self.L, self.lower), y)
        
        return self
    
    def predict(self, X_test, return_std=True):
        """
        Posterior predictive distribution.
        
        μ* = K*ᵀ (K + σ²I)⁻¹ y
        Σ* = K** - K*ᵀ (K + σ²I)⁻¹ K*
        """
        K_star = self._rbf_kernel(X_test, self.X_train)
        
        # Mean: K*ᵀ α
        mean = K_star @ self.alpha
        
        if return_std:
            # Variance: K** - K*ᵀ (K + σ²I)⁻¹ K*
            v = cho_solve((self.L, self.lower), K_star.T)
            var = self._rbf_kernel(X_test, X_test) - K_star @ v
            std = np.sqrt(np.diag(var))
            return mean, std
        
        return mean
    
    def log_marginal_likelihood(self):
        """
        Log marginal likelihood for hyperparameter optimization.
        
        log p(y|X) = -½ yᵀ(K+σ²I)⁻¹y - ½ log|K+σ²I| - n/2 log(2π)
        """
        n = len(self.y_train)
        
        # Data fit term
        data_fit = -0.5 * self.y_train @ self.alpha
        
        # Complexity term
        complexity = -np.sum(np.log(np.diag(self.L)))
        
        # Normalization
        normalization = -0.5 * n * np.log(2 * np.pi)
        
        return data_fit + complexity + normalization
    
    def optimize_hyperparameters(self, X, y):
        """Optimize hyperparameters via marginal likelihood."""
        
        def neg_log_likelihood(params):
            self.length_scale = np.exp(params[0])
            self.signal_var = np.exp(params[1])
            self.noise_var = np.exp(params[2])
            self.fit(X, y)
            return -self.log_marginal_likelihood()
        
        init_params = np.log([self.length_scale, self.signal_var, self.noise_var])
        result = minimize(neg_log_likelihood, init_params, method='L-BFGS-B')
        
        # Set optimal parameters
        self.length_scale = np.exp(result.x[0])
        self.signal_var = np.exp(result.x[1])
        self.noise_var = np.exp(result.x[2])
        self.fit(X, y)
        
        return self

class BayesianOptimization:
    """
    Bayesian Optimization using GP surrogate.
    
    Uses Expected Improvement acquisition function.
    """
    
    def __init__(self, objective, bounds, n_init=5):
        self.objective = objective
        self.bounds = bounds
        self.n_init = n_init
        self.gp = GaussianProcess()
        self.X_observed = []
        self.y_observed = []
    
    def expected_improvement(self, X, xi=0.01):
        """
        Expected Improvement acquisition function.
        
        EI(x) = (μ(x) - f_best - ξ) Φ(Z) + σ(x) φ(Z)
        where Z = (μ(x) - f_best - ξ) / σ(x)
        """
        from scipy.stats import norm
        
        mu, sigma = self.gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        
        f_best = np.max(self.y_observed)
        
        Z = (mu - f_best - xi) / sigma
        ei = (mu - f_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        
        return ei
    
    def optimize(self, n_iter=20):
        """Run Bayesian optimization."""
        dim = len(self.bounds)
        
        # Initial random samples
        for _ in range(self.n_init):
            x = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
            y = self.objective(x)
            self.X_observed.append(x)
            self.y_observed.append(y)
        
        # Optimization loop
        for _ in range(n_iter):
            # Fit GP
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)
            
            # Find next point by maximizing EI
            best_ei = -np.inf
            best_x = None
            
            # Random search for maximum EI
            for _ in range(1000):
                x = np.array([np.random.uniform(b[0], b[1]) for b in self.bounds])
                ei = self.expected_improvement(x.reshape(1, -1))[0]
                if ei > best_ei:
                    best_ei = ei
                    best_x = x
            
            # Evaluate objective
            y_new = self.objective(best_x)
            self.X_observed.append(best_x)
            self.y_observed.append(y_new)
        
        # Return best observed point
        best_idx = np.argmax(self.y_observed)
        return self.X_observed[best_idx], self.y_observed[best_idx]

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    
    # Generate data
    X_train = np.random.uniform(-5, 5, (20, 1))
    y_train = np.sin(X_train.ravel()) + 0.1 * np.random.randn(20)
    
    # Fit GP
    gp = GaussianProcess(length_scale=1.0, noise_var=0.01)
    gp.fit(X_train, y_train)
    
    # Predict
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    mean, std = gp.predict(X_test)
    
    print(f"Log marginal likelihood: {gp.log_marginal_likelihood():.4f}")
    print(f"Length scale: {gp.length_scale:.4f}")
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Gaussian Processes for ML | [Rasmussen & Williams](https://gaussianprocess.org/gpml/) |
| 📖 | Bishop PRML Ch. 6 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | GPyTorch | [Docs](https://gpytorch.ai/) |

---

⬅️ [Back: Kernel Methods](../) | ➡️ [Next: Kernels](../02_kernels/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Kernel Methods</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_kernels/">Next: Kernels ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
