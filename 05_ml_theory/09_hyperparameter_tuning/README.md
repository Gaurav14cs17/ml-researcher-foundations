<!-- Navigation -->
<p align="center">
  <a href="../08_model_selection/">⬅️ Prev: Model Selection</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../10_interpretability/">Next: Interpretability ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Hyperparameter%20Tuning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/hyperparameter-tuning-complete.svg" width="100%">

*Caption: Hyperparameter optimization finds optimal model settings through search strategies.*

---

## 📂 Overview

**Hyperparameter tuning** is the process of finding optimal hyperparameters $\lambda$ that minimize validation error:

```math
\lambda^* = \arg\min_{\lambda \in \Lambda} \mathcal{L}_{\text{val}}(h_{\lambda})
```

where $h_\lambda$ is the model trained with hyperparameters $\lambda$.

---

## 📐 Search Strategies

### Grid Search

Exhaustive search over a discretized grid:

```math
\Lambda_{\text{grid}} = \{\lambda_1^{(1)}, \ldots, \lambda_1^{(k_1)}\} \times \cdots \times \{\lambda_d^{(1)}, \ldots, \lambda_d^{(k_d)}\}
```

**Complexity:** $O(\prod_{i=1}^d k_i)$

### Random Search

**Theorem (Bergstra & Bengio, 2012):** For a function with low effective dimensionality, random search is more efficient than grid search.

**Key insight:** Random search explores more unique values per dimension.

### Bayesian Optimization

**Objective:** Find $\lambda^* = \arg\min_\lambda f(\lambda)$ where $f$ is expensive to evaluate.

**Algorithm:**
1. Model $f(\lambda)$ with surrogate (GP or TPE)
2. Select next $\lambda$ using acquisition function
3. Evaluate $f(\lambda)$
4. Update surrogate
5. Repeat

---

## 📐 Gaussian Process Surrogate

### GP Prior

```math
f(\lambda) \sim \mathcal{GP}(m(\lambda), k(\lambda, \lambda'))
```

### Acquisition Functions

**Expected Improvement (EI):**

```math
\text{EI}(\lambda) = \mathbb{E}[\max(f^+ - f(\lambda), 0)]
```

where $f^+ = \min_i f(\lambda_i)$ is the best observed value.

**Closed form for GP:**

```math
\text{EI}(\lambda) = (f^+ - \mu(\lambda))\Phi(Z) + \sigma(\lambda)\phi(Z)
```

where $Z = \frac{f^+ - \mu(\lambda)}{\sigma(\lambda)}$.

**Upper Confidence Bound (UCB):**

```math
\text{UCB}(\lambda) = \mu(\lambda) - \kappa \sigma(\lambda)
```

where $\kappa$ controls exploration vs exploitation.

---

## 📐 Hyperband

### Successive Halving

Given budget $B$ and $n$ configurations:
1. Allocate $B/n$ resources to each configuration
2. Evaluate and keep top $1/\eta$ fraction
3. Repeat with increased budget

**Theorem:** Successive halving finds the best arm with at most $O(\frac{B}{\log(n)})$ additional evaluations compared to optimal.

### Hyperband Algorithm

Run successive halving with different $(n, B/n)$ trade-offs:

```math
s_{\max} = \lfloor \log_\eta(B) \rfloor
```

For each $s \in \{s_{\max}, \ldots, 0\}$:
- $n = \lceil \frac{B \eta^s}{(s+1)} \rceil$ initial configurations
- $r = B \eta^{-s}$ minimum budget per configuration

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

class BayesianOptimizer:
    """
    Bayesian Optimization with GP surrogate.
    
    min_λ f(λ) via sequential optimization
    """
    
    def __init__(self, bounds, n_initial=5, kernel=None):
        self.bounds = np.array(bounds)
        self.n_initial = n_initial
        
        if kernel is None:
            kernel = Matern(nu=2.5) + WhiteKernel()
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True
        )
        
        self.X = []
        self.y = []
    
    def _expected_improvement(self, X, xi=0.01):
        """
        Expected Improvement acquisition function.
        
        EI(x) = (f⁺ - μ(x))Φ(Z) + σ(x)φ(Z)
        where Z = (f⁺ - μ(x)) / σ(x)
        """
        mu, sigma = self.gp.predict(X.reshape(-1, len(self.bounds)), return_std=True)
        mu = mu.ravel()
        
        f_best = np.min(self.y)
        
        with np.errstate(divide='warn'):
            Z = (f_best - mu - xi) / sigma
            ei = (f_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return -ei  # Negative for minimization
    
    def suggest_next(self):
        """Suggest next point to evaluate."""
        dim = len(self.bounds)
        
        # Optimize acquisition function
        best_x = None
        best_acq = np.inf
        
        # Multi-start optimization
        for _ in range(20):
            x0 = np.random.uniform(
                self.bounds[:, 0], 
                self.bounds[:, 1]
            )
            
            result = minimize(
                self._expected_improvement,
                x0,
                method='L-BFGS-B',
                bounds=self.bounds
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        return best_x
    
    def optimize(self, objective, n_iterations=50):
        """Run Bayesian optimization."""
        
        # Initial random samples
        for _ in range(self.n_initial):
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            y = objective(x)
            self.X.append(x)
            self.y.append(y)
        
        # Bayesian optimization loop
        for i in range(n_iterations):

            # Fit GP
            X_arr = np.array(self.X)
            y_arr = np.array(self.y)
            self.gp.fit(X_arr, y_arr)
            
            # Get next point
            x_next = self.suggest_next()
            y_next = objective(x_next)
            
            self.X.append(x_next)
            self.y.append(y_next)
            
            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: best = {np.min(self.y):.4f}")
        
        # Return best
        best_idx = np.argmin(self.y)
        return self.X[best_idx], self.y[best_idx]

class Hyperband:
    """
    Hyperband: Principled early-stopping for hyperparameter optimization.
    """
    
    def __init__(self, max_budget=81, eta=3):
        self.max_budget = max_budget
        self.eta = eta
        self.s_max = int(np.floor(np.log(max_budget) / np.log(eta)))
    
    def successive_halving(self, configs, budgets, evaluate_fn):
        """Run successive halving."""
        for budget in budgets:

            # Evaluate all configurations
            results = [(c, evaluate_fn(c, budget)) for c in configs]
            
            # Sort by performance (lower is better)
            results.sort(key=lambda x: x[1])
            
            # Keep top 1/eta
            n_keep = max(1, len(configs) // self.eta)
            configs = [r[0] for r in results[:n_keep]]
        
        return configs[0], results[0][1]
    
    def run(self, sample_config_fn, evaluate_fn):
        """Run Hyperband."""
        best_config = None
        best_loss = np.inf
        
        for s in range(self.s_max, -1, -1):

            # Number of configurations
            n = int(np.ceil((self.s_max + 1) / (s + 1) * self.eta ** s))
            
            # Initial budget per configuration
            r = self.max_budget * self.eta ** (-s)
            
            # Sample configurations
            configs = [sample_config_fn() for _ in range(n)]
            
            # Budgets for successive halving
            budgets = [r * self.eta ** i for i in range(s + 1)]
            
            # Run successive halving
            config, loss = self.successive_halving(configs, budgets, evaluate_fn)
            
            if loss < best_loss:
                best_loss = loss
                best_config = config
                print(f"s={s}: New best = {best_loss:.4f}")
        
        return best_config, best_loss

# Example usage
if __name__ == "__main__":

    # Define objective function (e.g., validation loss)
    def objective(params):
        x, y = params

        # Branin function (common benchmark)
        a, b, c = 1, 5.1/(4*np.pi**2), 5/np.pi
        r, s, t = 6, 10, 1/(8*np.pi)
        return a*(y - b*x**2 + c*x - r)**2 + s*(1-t)*np.cos(x) + s
    
    # Bayesian optimization
    bounds = [(-5, 10), (0, 15)]
    optimizer = BayesianOptimizer(bounds)
    best_x, best_y = optimizer.optimize(objective, n_iterations=30)
    print(f"\nBest found: x = {best_x}, y = {best_y:.4f}")
```

---

## 📊 Method Comparison

| Method | Efficiency | Parallelization | Best For |
|--------|------------|-----------------|----------|
| Grid Search | Low | Excellent | Few params |
| Random Search | Medium | Excellent | Medium params |
| Bayesian | High | Limited | Expensive evals |
| Hyperband | High | Good | Deep learning |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Random Search | [Bergstra & Bengio](https://jmlr.org/papers/v13/bergstra12a.html) |
| 📄 | Hyperband | [Li et al.](https://arxiv.org/abs/1603.06560) |
| 📄 | BOHB | [Falkner et al.](https://arxiv.org/abs/1807.01774) |
| 📖 | Optuna | [Docs](https://optuna.readthedocs.io/) |

---

⬅️ [Back: Model Selection](../08_model_selection/) | ➡️ [Next: Interpretability](../10_interpretability/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../08_model_selection/">⬅️ Prev: Model Selection</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../10_interpretability/">Next: Interpretability ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
