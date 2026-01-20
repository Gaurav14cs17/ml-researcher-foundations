<!-- Navigation -->
<p align="center">
  <a href="../03_svm/">⬅️ Prev: SVM</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_representation/">Next: Representation ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Regularization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/l1-l2-regularization-complete.svg" width="100%">

*Caption: Regularization adds penalties to prevent overfitting. L1 promotes sparsity, L2 shrinks weights.*

---

## 📂 Overview

**Regularization** constrains model complexity to improve generalization. It is the practical implementation of Structural Risk Minimization (SRM).

---

## 📐 Mathematical Framework

### Regularized Loss Function

```math
\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}(\theta; \mathcal{D}) + \lambda \Omega(\theta)

```

where:

- \(\mathcal{L}(\theta; \mathcal{D})\): Data-dependent loss

- \(\Omega(\theta)\): Regularization penalty

- \(\lambda > 0\): Regularization strength

---

## 📐 L2 Regularization (Ridge)

### Formulation

```math
\mathcal{L}_{\text{L2}} = \mathcal{L}(\theta) + \lambda \|\theta\|_2^2 = \mathcal{L}(\theta) + \lambda \sum_j \theta_j^2

```

### Closed-Form Solution (Linear Regression)

```math
\hat{\theta} = (X^\top X + \lambda I)^{-1} X^\top y

```

**Theorem:** L2 regularization is equivalent to MAP estimation with Gaussian prior:

```math
p(\theta) \propto \exp\left(-\frac{\lambda}{2}\|\theta\|^2\right)

```

**Proof:** 

```math
\log p(\theta | \mathcal{D}) = \log p(\mathcal{D}|\theta) + \log p(\theta) + \text{const}
= -\mathcal{L}(\theta) - \frac{\lambda}{2}\|\theta\|^2 + \text{const} \quad \blacksquare

```

---

## 📐 L1 Regularization (Lasso)

### Formulation

```math
\mathcal{L}_{\text{L1}} = \mathcal{L}(\theta) + \lambda \|\theta\|_1 = \mathcal{L}(\theta) + \lambda \sum_j |\theta_j|

```

### Soft Thresholding

For \(\mathcal{L} = \frac{1}{2}(y - \theta)^2\):

```math
\hat{\theta} = \text{sign}(y) \cdot \max(|y| - \lambda, 0)

```

**Theorem:** L1 regularization produces sparse solutions.

**Proof (Geometry):** The L1 constraint \(\|\theta\|_1 \leq r\) forms a diamond (polytope). The optimal point where the loss contour touches this constraint is typically at a vertex, where some coordinates are zero. \(\blacksquare\)

### Bayesian Interpretation

L1 corresponds to a Laplace prior:

```math
p(\theta) \propto \exp(-\lambda \|\theta\|_1) = \prod_j \frac{\lambda}{2}\exp(-\lambda |\theta_j|)

```

---

## 📐 Elastic Net

```math
\mathcal{L}_{\text{EN}} = \mathcal{L}(\theta) + \lambda_1 \|\theta\|_1 + \lambda_2 \|\theta\|_2^2

```

**Advantages:**
- Combines sparsity (L1) with stability (L2)

- Handles correlated features (L1 alone selects one arbitrarily)

- Bounded selection: selects groups of correlated features

---

## 📐 Dropout

### Training

For each training step, randomly drop neurons with probability \(p\):

```math
\tilde{h}_i = \frac{m_i}{1-p} \cdot h_i, \quad m_i \sim \text{Bernoulli}(1-p)

```

### Inference

Use all neurons with scaled weights:

```math
h_i = h_i
$$  (no dropout, weights already scaled during training)

**Theorem (Approximate Bayesian Interpretation):** Dropout approximates Bayesian model averaging.

**Proof Sketch:** 
Each dropout mask defines a "thinned" network. Training with dropout is equivalent to training an exponentially large ensemble where each network shares parameters. At test time, averaging predictions approximates Bayesian model averaging. \(\blacksquare\)

---

## 📐 Early Stopping

### Implicit Regularization

**Theorem:** For gradient descent on linear regression, early stopping is equivalent to L2 regularization:

```

\theta^{(t)} \approx \theta_{\lambda}, \quad \text{where } \lambda = \frac{1}{\eta t}

```math
**Proof:**
For gradient descent with step size \(\eta\):

```

\theta^{(t+1)} = \theta^{(t)} - \eta X^\top(X\theta^{(t)} - y)

```math
The solution path is:

```

\theta^{(t)} = (I - (I - \eta X^\top X)^t)(X^\top X)^{-1}X^\top y

```math

This matches the ridge solution with \(\lambda = \frac{1}{\eta t}\) as \(\eta \to 0\). \(\blacksquare\)

---

## 💻 Code Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RegularizedLoss:
    """
    Regularized loss functions.
    
    L_reg = L + λ₁||θ||₁ + λ₂||θ||₂²
    """
    
    def __init__(self, base_loss, l1_weight=0.0, l2_weight=0.0):
        self.base_loss = base_loss
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
    
    def __call__(self, model, predictions, targets):
        loss = self.base_loss(predictions, targets)
        
        # L1 regularization
        if self.l1_weight > 0:
            l1_reg = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.l1_weight * l1_reg
        
        # L2 regularization
        if self.l2_weight > 0:
            l2_reg = sum(p.pow(2).sum() for p in model.parameters())
            loss = loss + self.l2_weight * l2_reg
        
        return loss

class RidgeRegression:
    """
    Ridge Regression: min ||y - Xθ||² + λ||θ||²
    
    Closed-form: θ = (X'X + λI)⁻¹X'y
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X, y):
        n, d = X.shape
        self.theta = np.linalg.solve(
            X.T @ X + self.alpha * np.eye(d),
            X.T @ y
        )
        return self
    
    def predict(self, X):
        return X @ self.theta

class LassoRegression:
    """
    Lasso Regression: min ||y - Xθ||² + λ||θ||₁
    
    Solved via coordinate descent.
    """
    
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
    
    def soft_threshold(self, x, threshold):
        """Soft thresholding operator."""
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def fit(self, X, y):
        n, d = X.shape
        self.theta = np.zeros(d)
        
        for _ in range(self.max_iter):
            theta_old = self.theta.copy()
            
            for j in range(d):
                # Compute residual without feature j
                residual = y - X @ self.theta + X[:, j] * self.theta[j]
                
                # Coordinate update with soft thresholding
                rho = X[:, j] @ residual
                z = X[:, j] @ X[:, j]
                
                self.theta[j] = self.soft_threshold(rho, n * self.alpha) / z
            
            # Check convergence
            if np.linalg.norm(self.theta - theta_old) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        return X @ self.theta

class DropoutLayer(nn.Module):
    """
    Dropout regularization.
    
    Training: h̃ᵢ = mᵢ/(1-p) · hᵢ, m ~ Bernoulli(1-p)
    Inference: h̃ᵢ = hᵢ
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.p))
            return x * mask / (1 - self.p)
        return x

class RegularizedNetwork(nn.Module):
    """Neural network with various regularization techniques."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        return self.fc3(x)

def train_with_early_stopping(model, train_loader, val_loader, 
                               max_epochs=1000, patience=10, lr=0.001):
    """
    Training with early stopping (implicit regularization).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch), y_batch).item()
        val_loss /= len(val_loader)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    return model

```

---

## 📊 Regularization Comparison

| Method | Effect | Sparsity | Computation |
|--------|--------|----------|-------------|
| L2 (Ridge) | Shrinks weights | No | Closed-form |
| L1 (Lasso) | Sparse weights | Yes | Iterative |
| Elastic Net | Both | Partial | Iterative |
| Dropout | Ensemble | N/A | Training only |
| Early Stopping | Implicit L2 | No | Monitor val loss |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | ESL | [Hastie et al.](https://hastie.su.domains/ElemStatLearn/) |
| 📄 | Dropout | [Srivastava et al.](https://jmlr.org/papers/v15/srivastava14a.html) |
| 📄 | Lasso | [Tibshirani](https://www.jstor.org/stable/2346178) |

---

⬅️ [Back: Kernel Methods](../03_kernel_methods/) | ➡️ [Next: Representation](../04_representation/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../03_svm/">⬅️ Prev: SVM</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_representation/">Next: Representation ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>

```