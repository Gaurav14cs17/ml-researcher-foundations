<!-- Navigation -->
<p align="center">
  <a href="../03_overfitting/">⬅️ Prev: Overfitting</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_vc_dimension/">Next: VC Dimension ➡️</a>
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

<img src="./images/regularization.svg" width="100%">

*Caption: Regularization constrains models to improve generalization. Explicit (L1/L2 penalties), implicit (SGD, early stopping), and data-based (augmentation, dropout) regularization all reduce overfitting.*

---

## 📂 Overview

**Regularization** is a fundamental technique for preventing overfitting by adding constraints or penalties that prefer simpler models. It trades a small increase in bias for a larger decrease in variance, improving generalization.

---

## 📐 Mathematical Framework

### General Regularized Objective

```math
\mathcal{L}_{\text{reg}}(\theta) = \underbrace{\mathcal{L}_{\text{data}}(\theta)}_{\text{empirical risk}} + \underbrace{\lambda \Omega(\theta)}_{\text{regularization penalty}}

```

where:

- \(\mathcal{L}_{\text{data}}\): Data fitting term (e.g., cross-entropy, MSE)

- \(\Omega(\theta)\): Regularization function (complexity penalty)

- \(\lambda > 0\): Regularization strength (hyperparameter)

---

## 📊 L2 Regularization (Ridge / Weight Decay)

### Definition

```math
\Omega(\theta) = \|\theta\|_2^2 = \sum_{i} \theta_i^2

```

**Full Objective:**

```math
\mathcal{L}_{\text{ridge}}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i) + \lambda \|\theta\|_2^2

```

### Gradient

```math
\nabla_\theta \Omega = 2\theta

```

**Effect:** Shrinks weights proportionally toward zero.

**Update Rule (SGD with weight decay):**

```math
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{data}} - 2\eta\lambda\theta = (1 - 2\eta\lambda)\theta - \eta \nabla_\theta \mathcal{L}_{\text{data}}

```

### Closed-Form Solution (Linear Regression)

For linear regression \(y = X\beta + \varepsilon\):

```math
\hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y

```

**Proof:**

Set gradient to zero:

```math
\nabla_\beta \left[\|y - X\beta\|^2 + \lambda\|\beta\|^2\right] = -2X^\top(y - X\beta) + 2\lambda\beta = 0
X^\top X\beta + \lambda\beta = X^\top y
(X^\top X + \lambda I)\beta = X^\top y
\boxed{\hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y}

```

### Bayesian Interpretation

**Theorem:** Ridge regression is equivalent to MAP estimation with Gaussian prior.

**Prior:** \(\theta \sim \mathcal{N}(0, \tau^2 I)\)

**Posterior:**

```math
p(\theta | \mathcal{D}) \propto p(\mathcal{D} | \theta) \cdot p(\theta)
\log p(\theta | \mathcal{D}) = -\frac{1}{2\sigma^2}\|y - X\theta\|^2 - \frac{1}{2\tau^2}\|\theta\|^2 + \text{const}

```

**MAP estimate:**

```math
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \log p(\theta | \mathcal{D}) = \arg\min_\theta \left[\|y - X\theta\|^2 + \frac{\sigma^2}{\tau^2}\|\theta\|^2\right]

```

Setting \(\lambda = \frac{\sigma^2}{\tau^2}\) gives ridge regression.

---

## 📊 L1 Regularization (Lasso)

### Definition

```math
\Omega(\theta) = \|\theta\|_1 = \sum_{i} |\theta_i|

```

**Full Objective:**

```math
\mathcal{L}_{\text{lasso}}(\theta) = \frac{1}{n}\sum_{i=1}^{n} \ell(f_\theta(x_i), y_i) + \lambda \|\theta\|_1

```

### Subgradient

```math
\partial_{\theta_i} \Omega = \text{sign}(\theta_i) = \begin{cases} 1 & \theta_i > 0 \\ [-1, 1] & \theta_i = 0 \\ -1 & \theta_i < 0 \end{cases}

```

**Effect:** Constant push toward zero → **sparse solutions**.

### Why L1 Produces Sparsity

**Geometric Intuition:**

The L1 ball \(\|\theta\|_1 \leq t\) has corners aligned with axes. The solution lies where the level curves of \(\mathcal{L}_{\text{data}}\) first touch the constraint region—likely at a corner where some \(\theta_i = 0\).

**Proximal Operator (Soft Thresholding):**

```math
\text{prox}_{\lambda\|\cdot\|_1}(\theta) = \text{sign}(\theta) \odot \max(|\theta| - \lambda, 0)

```

This explicitly sets small weights to exactly zero.

### Bayesian Interpretation

**Theorem:** Lasso is equivalent to MAP estimation with Laplace prior.

**Laplace Prior:**

```math
p(\theta_i) = \frac{1}{2b} \exp\left(-\frac{|\theta_i|}{b}\right)
\log p(\theta) = -\frac{1}{b}\|\theta\|_1 + \text{const}

```

---

## 📊 Elastic Net

### Definition

Combines L1 and L2:

```math
\Omega(\theta) = \alpha\|\theta\|_1 + (1-\alpha)\|\theta\|_2^2

```

**Full Objective:**

```math
\mathcal{L}_{\text{elastic}} = \mathcal{L}_{\text{data}} + \lambda \left[\alpha\|\theta\|_1 + (1-\alpha)\|\theta\|_2^2\right]

```

### Advantages

| Property | L1 | L2 | Elastic Net |
|----------|----|----|-------------|
| **Sparsity** | ✓ | ✗ | ✓ |
| **Unique solution** | ✗ | ✓ | ✓ |
| **Grouped selection** | ✗ | ✓ | ✓ |
| **Handles correlated features** | ✗ | ✓ | ✓ |

---

## 📊 Dropout

### Definition

During training, randomly zero out each hidden unit with probability \(p\):

```math
\tilde{h}_i = h_i \cdot m_i, \quad m_i \sim \text{Bernoulli}(1-p)

```

During inference, scale activations:

```math
\tilde{h}_i = (1-p) \cdot h_i

```

### Mathematical Interpretation

**Theorem (Wager et al., 2013):** Dropout is approximately equivalent to L2 regularization:

```math
\mathbb{E}[\mathcal{L}_{\text{dropout}}] \approx \mathcal{L}_{\text{data}} + \lambda \sum_i p(1-p) \left(\frac{\partial \mathcal{L}}{\partial h_i}\right)^2 h_i^2

```

### Ensemble Interpretation

**Theorem:** Dropout trains an ensemble of \(2^n\) networks (all subnetworks), with weight sharing.

At test time, the scaled network approximates the geometric mean of all subnetwork predictions:

```math
\tilde{f}(x) \approx \left(\prod_{S \subseteq \{1,...,n\}} f_S(x)\right)^{1/2^n}

```

---

## 📊 Comparison of L1 vs L2

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| **Penalty** | \(\sum_i \|\theta_i\|\) | \(\sum_i \theta_i^2\) |
| **Gradient** | \(\text{sign}(\theta)\) | \(2\theta\) |
| **Sparsity** | Yes (zeros out weights) | No (shrinks but ≠ 0) |
| **Solution uniqueness** | May not be unique | Always unique |
| **Prior (Bayesian)** | Laplace | Gaussian |
| **Use case** | Feature selection | General regularization |
| **Computational** | Requires special solvers | Closed-form for linear |

### Visual Comparison

```
L1 Constraint Region (Diamond):       L2 Constraint Region (Circle):
         
           |                                    |
           |                                  ╱   ╲
        ╱  |  ╲                              |     |
      ╱    |    ╲                            |     |
    ╱      |      ╲                         -+-----+-
    - - - -+- - - -                          |     |
    ╲      |      ╱                           ╲   ╱
      ╲    |    ╱                              ╲ ╱
        ╲  |  ╱                                 |
           |                                    |

Corners → Solutions at axes            Smooth → Solutions rarely at axes
(sparse)                               (non-sparse)

```

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# L2 Regularization via Weight Decay (Recommended)
# ============================================================

# Option 1: Use AdamW (decoupled weight decay)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01  # L2 penalty coefficient
)

# Option 2: Standard Adam with weight_decay (coupled, less effective)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01
)

# ============================================================
# Manual L1 Regularization
# ============================================================

def l1_regularization(model, l1_lambda=0.001):
    """
    Compute L1 penalty: λ * Σ|θ_i|
    
    This encourages sparsity in model weights.
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm

# Training loop with L1
for epoch in range(epochs):
    for x, y in train_loader:
        optimizer.zero_grad()
        
        # Data loss
        pred = model(x)
        data_loss = criterion(pred, y)
        
        # L1 regularization
        l1_loss = l1_regularization(model, l1_lambda=0.001)
        
        # Total loss
        total_loss = data_loss + l1_loss
        
        total_loss.backward()
        optimizer.step()

# ============================================================
# Elastic Net Regularization
# ============================================================

def elastic_net_regularization(model, l1_lambda=0.001, l2_lambda=0.01, alpha=0.5):
    """
    Elastic Net: α * L1 + (1-α) * L2
    
    Combines benefits of both:
    - Sparsity from L1
    - Stability from L2
    """
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    l2_norm = sum(p.pow(2).sum() for p in model.parameters())
    
    return alpha * l1_lambda * l1_norm + (1 - alpha) * l2_lambda * l2_norm

# ============================================================
# Dropout Layer (Built-in)
# ============================================================

class RegularizedMLP(nn.Module):
    """MLP with multiple regularization techniques."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)  # Dropout active only in training
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ============================================================
# Batch Normalization (Implicit Regularization)
# ============================================================

class BatchNormMLP(nn.Module):
    """MLP with batch normalization for regularization."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ============================================================
# Early Stopping (Implicit Regularization)
# ============================================================

class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    
    This is a form of implicit regularization that limits
    the effective capacity of the model.
    """
    
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                
        return self.should_stop

# ============================================================
# Spectral Normalization (Lipschitz Regularization)
# ============================================================

from torch.nn.utils import spectral_norm

class SpectralNormMLP(nn.Module):
    """MLP with spectral normalization for Lipschitz constraint."""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Spectral norm constrains largest singular value of weight matrix
        self.fc1 = spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = spectral_norm(nn.Linear(hidden_dim, output_dim))
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ============================================================
# Complete Training Example
# ============================================================

def train_with_regularization(model, train_loader, val_loader, epochs=100):
    """Complete training loop with multiple regularization techniques."""
    
    # L2 via weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            
            # Add L1 if desired
            # loss += l1_regularization(model, l1_lambda=0.001)
            
            loss.backward()
            
            # Gradient clipping (another form of regularization)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x)
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Loss = {val_loss:.4f}")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

```

---

## 📊 Summary of Regularization Techniques

| Technique | Type | Mechanism | When to Use |
|-----------|------|-----------|-------------|
| **L2 / Weight Decay** | Explicit | Gaussian prior, shrinks weights | Default choice |
| **L1 / Lasso** | Explicit | Laplace prior, sparse solutions | Feature selection |
| **Elastic Net** | Explicit | Combines L1 + L2 | Correlated features |
| **Dropout** | Implicit | Ensemble of subnetworks | Deep networks |
| **Batch Norm** | Implicit | Normalizes activations | CNNs, deep nets |
| **Early Stopping** | Implicit | Limits optimization | Universal |
| **Data Augmentation** | Data-based | Increases effective dataset | Vision, NLP |
| **Label Smoothing** | Output | Softens targets | Classification |
| **Gradient Clipping** | Optimization | Bounds gradient norm | RNNs, Transformers |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Dropout (Srivastava et al., 2014) | [JMLR](https://jmlr.org/papers/v15/srivastava14a.html) |
| 📖 | ESL Ch. 3: Linear Methods | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 📖 | Bishop PRML Ch. 3.1-3.2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | Decoupled Weight Decay (Loshchilov, 2019) | [arXiv](https://arxiv.org/abs/1711.05101) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../03_overfitting/">⬅️ Prev: Overfitting</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../05_vc_dimension/">Next: VC Dimension ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
