<!-- Navigation -->
<p align="center">
  <a href="../02_pac_learning/">⬅️ Prev: PAC Learning</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../06_ensemble_methods/">Next: Ensemble Methods ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Structural%20Risk%20Minimization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/srm.svg" width="100%">

*Caption: SRM balances empirical risk with model complexity to prevent overfitting.*

---

## 📂 Overview

**Structural Risk Minimization (SRM)** extends Empirical Risk Minimization by incorporating model complexity. It provides the theoretical foundation for regularization in machine learning.

---

## 📐 The SRM Principle

### Nested Hypothesis Classes

Consider a sequence of nested hypothesis classes:

```math
\mathcal{H}_1 \subset \mathcal{H}_2 \subset \mathcal{H}_3 \subset \cdots
```

with increasing complexity: \(\text{VC}(\mathcal{H}_1) < \text{VC}(\mathcal{H}_2) < \cdots\)

**SRM Objective:**

```math
\hat{h} = \arg\min_{k} \left[\min_{h \in \mathcal{H}_k} \hat{R}(h) + \Phi(k, n, \delta)\right]
```

where \(\Phi(k, n, \delta)\) is the complexity penalty depending on VC dimension of $\mathcal{H}_k$.

---

## 📐 Theoretical Foundation

### Vapnik's SRM Bound

**Theorem:** With probability $\geq 1 - \delta$, for all $h \in \mathcal{H}$:

```math
R(h) \leq \hat{R}(h) + \sqrt{\frac{d(\ln(2n/d) + 1) + \ln(4/\delta)}{n}}
```

where \(d = \text{VC}(\mathcal{H})\).

### Regularization Form

SRM is equivalent to regularized ERM:

```math
\hat{h} = \arg\min_{h \in \mathcal{H}} \left[\hat{R}(h) + \lambda \Omega(h)\right]
```

**Common Regularizers:**

| Regularizer | \(\Omega(h)\) | Effect |
|-------------|---------------|--------|
| L2 (Ridge) | $\|w\|_2^2$ | Small weights |
| L1 (Lasso) | $\|w\|_1$ | Sparse weights |
| RKHS Norm | $\|h\|_{\mathcal{H}}^2$ | Smooth functions |
| Spectral Norm | \(\sigma_{\max}(W)\) | Lipschitz constraint |

---

## 📐 Connection to Bayesian Learning

### MAP Estimation

Regularized ERM is equivalent to Maximum A Posteriori (MAP) estimation:

```math
\hat{h}_{\text{MAP}} = \arg\max_h p(h | \mathcal{D}) = \arg\max_h p(\mathcal{D}|h) p(h)
```

Taking negative log:

```math
= \arg\min_h \left[-\log p(\mathcal{D}|h) - \log p(h)\right]
```

- \(-\log p(\mathcal{D}|h)\): Likelihood → Empirical risk
- \(-\log p(h)\): Prior → Regularization

**Specific Priors:**
- Gaussian prior \(p(w) \propto e^{-\lambda \|w\|^2}\) → L2 regularization
- Laplace prior \(p(w) \propto e^{-\lambda \|w\|_1}\) → L1 regularization

---

## 📐 Early Stopping as SRM

### Implicit Regularization

**Theorem (Ali Rahimi et al.):** For gradient descent on linear regression with step size $\eta$:

```math
w^{(t)} = \sum_{i=1}^n \alpha_i^{(t)} x_i
```

where after $t$ steps, this approximates ridge regression with \(\lambda = 1/(\eta t)\).

**Proof Sketch:** The gradient descent iterates stay in the span of training data. Early stopping limits the effective complexity of the learned function. $\blacksquare$

---

## 💻 Code Implementation

```python
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

class StructuralRiskMinimizer:
    """
    SRM via regularization path search.
    
    min_h [R̂(h) + λ Ω(h)]
    """
    
    def __init__(self, base_model_class=Ridge, 
                 lambda_range=np.logspace(-4, 4, 20)):
        self.base_model_class = base_model_class
        self.lambda_range = lambda_range
        self.best_lambda = None
        self.best_model = None
    
    def fit(self, X, y, cv=5):
        """Find optimal regularization via cross-validation."""
        best_score = -np.inf
        
        for lam in self.lambda_range:
            model = self.base_model_class(alpha=lam)
            scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                self.best_lambda = lam
        
        # Fit final model with best lambda
        self.best_model = self.base_model_class(alpha=self.best_lambda)
        self.best_model.fit(X, y)
        
        return self
    
    def predict(self, X):
        return self.best_model.predict(X)
    
    def srm_bound(self, n, d, delta=0.05):
        """
        Compute SRM generalization bound.
        
        R(h) ≤ R̂(h) + √((d(ln(2n/d)+1) + ln(4/δ))/n)
        """
        complexity_term = d * (np.log(2*n/d) + 1) + np.log(4/delta)
        return np.sqrt(complexity_term / n)

class EarlyStoppingRegularizer:
    """
    Implicit SRM via early stopping.
    
    Theory: Early stopping ≈ L2 regularization with λ = 1/(ηt)
    """
    
    def __init__(self, hidden_dim=64, lr=0.01, patience=10, max_epochs=1000):
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.patience = patience
        self.max_epochs = max_epochs
    
    def build_model(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
    
    def fit(self, X_train, y_train, X_val, y_val):
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        self.model = self.build_model(input_dim, output_dim)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val).reshape(-1, 1)
        
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(self.max_epochs):

            # Training step
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_train)
            loss = criterion(pred, y_train)
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = self.model.state_dict().copy()
                patience_counter = 0
                self.best_epoch = epoch
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break
        
        # Restore best model
        self.model.load_state_dict(best_state)
        self.train_losses = train_losses
        self.val_losses = val_losses
        
        # Effective regularization (approximate)
        self.effective_lambda = 1 / (self.lr * self.best_epoch)
        
        return self
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(X)).numpy()

def compare_srm_methods():
    """Compare different SRM approaches."""
    np.random.seed(42)
    
    # Generate data
    n_train, n_test = 100, 50
    d = 20
    
    # True signal is sparse
    true_w = np.zeros(d)
    true_w[:5] = [1, -0.5, 0.3, -0.2, 0.1]
    
    X = np.random.randn(n_train + n_test, d)
    y = X @ true_w + 0.1 * np.random.randn(n_train + n_test)
    
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    # Methods
    methods = {
        'No Regularization': Ridge(alpha=1e-10),
        'L2 (Ridge)': StructuralRiskMinimizer(Ridge),
        'L1 (Lasso)': StructuralRiskMinimizer(Lasso),
        'Elastic Net': StructuralRiskMinimizer(ElasticNet)
    }
    
    print("SRM Methods Comparison")
    print("=" * 50)
    
    for name, method in methods.items():
        if hasattr(method, 'fit') and hasattr(method, 'predict'):
            if isinstance(method, StructuralRiskMinimizer):
                method.fit(X_train, y_train)
            else:
                method.fit(X_train, y_train)
            
            train_mse = np.mean((method.predict(X_train) - y_train) ** 2)
            test_mse = np.mean((method.predict(X_test) - y_test) ** 2)
            
            print(f"{name:20s}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
            
            if hasattr(method, 'best_lambda'):
                print(f"                      Best λ = {method.best_lambda:.4f}")

if __name__ == "__main__":
    compare_srm_methods()
```

---

## 📊 SRM vs ERM

| Aspect | ERM | SRM |
|--------|-----|-----|
| Objective | \(\min \hat{R}(h)\) | \(\min \hat{R}(h) + \lambda\Omega(h)\) |
| Overfitting | Prone | Controlled |
| Generalization | No guarantee | Provable bounds |
| Computation | Simple | Requires tuning $\lambda$ |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Vapnik's Statistical Learning | [Book](https://link.springer.com/book/10.1007/978-1-4757-3264-1) |
| 📖 | Understanding ML | [Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📄 | Implicit Regularization | [arXiv](https://arxiv.org/abs/1705.01442) |

---

⬅️ [Back: PAC Learning](../02_pac_learning/) | ➡️ [Back: Risk Minimization](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../02_pac_learning/">⬅️ Prev: PAC Learning</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../06_ensemble_methods/">Next: Ensemble Methods ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
