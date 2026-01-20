<!-- Navigation -->
<p align="center">
  <a href="../01_self_supervised/">⬅️ Prev: Self-Supervised</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Learning Frameworks</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_unsupervised/">Next: Unsupervised ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Supervised%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/supervised.svg" width="100%">

*Caption: Supervised learning uses labeled data to learn a mapping from inputs to outputs.*

---

## 📂 Overview

**Supervised learning** is the most common ML paradigm where models learn from labeled examples \((x, y)\) to predict outputs for new inputs.

---

## 📐 Mathematical Framework

### Problem Setting

Given training data \(\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n\) sampled i.i.d. from distribution \(P_{XY}\):

```math
\min_{f \in \mathcal{F}} R(f) = \min_{f \in \mathcal{F}} \mathbb{E}_{(x,y) \sim P}[\ell(f(x), y)]
```

Since \(P\) is unknown, we minimize empirical risk:

```math
\hat{R}(f) = \frac{1}{n}\sum_{i=1}^n \ell(f(x_i), y_i)
```

---

## 📊 Classification

### Multi-class Classification

**Softmax function:**

```math
P(y = c | x) = \text{softmax}(f(x))_c = \frac{\exp(f_c(x))}{\sum_{j=1}^C \exp(f_j(x))}
```

**Cross-Entropy Loss:**

```math
\mathcal{L}_{\text{CE}} = -\sum_{c=1}^C y_c \log(\hat{y}_c) = -\log(\hat{y}_{c^*})
```

where \(c^*\) is the true class.

### Binary Classification

**Sigmoid function:**

```math
\sigma(z) = \frac{1}{1 + e^{-z}}
```

**Binary Cross-Entropy:**

```math
\mathcal{L}_{\text{BCE}} = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]
```

### Theoretical Properties

**Theorem (Bayes Optimal Classifier):** The classifier minimizing 0-1 loss is:

```math
h^*(x) = \arg\max_c P(Y = c | X = x)
```

**Proof:** 

```math
\text{error}(h) = \mathbb{E}[\mathbb{1}[h(X) \neq Y]] = \mathbb{E}_X[\mathbb{E}_{Y|X}[\mathbb{1}[h(X) \neq Y]]]
```

For each \(x\), the inner expectation is minimized by choosing \(h(x) = \arg\max_c P(Y=c|X=x)\). \(\blacksquare\)

---

## 📊 Regression

### Ordinary Least Squares

```math
\mathcal{L}_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n (y_i - f(x_i))^2
```

**Closed-form solution (linear case):**

```math
\hat{\beta} = (X^\top X)^{-1} X^\top y
```

**Theorem:** OLS is BLUE (Best Linear Unbiased Estimator) under Gauss-Markov conditions.

### Robust Regression

**MAE Loss (L1):**

```math
\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^n |y_i - f(x_i)|
```

**Huber Loss:**

```math
\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}
```

---

## 💻 Code Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupervisedLearning:
    """Framework for supervised learning with various losses."""
    
    @staticmethod
    def cross_entropy_loss(logits, targets):
        """
        Multi-class cross-entropy loss.
        
        L = -Σ_c y_c log(ŷ_c) = -log(ŷ_{true_class})
        
        Args:
            logits: Raw model outputs [batch_size, num_classes]
            targets: Class indices [batch_size]
        """
        return F.cross_entropy(logits, targets)
    
    @staticmethod
    def binary_cross_entropy(probs, targets):
        """
        Binary cross-entropy loss.
        
        L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
        """
        return F.binary_cross_entropy(probs, targets)
    
    @staticmethod
    def mse_loss(predictions, targets):
        """
        Mean squared error for regression.
        
        L = (1/n) Σ (y - ŷ)²
        """
        return F.mse_loss(predictions, targets)
    
    @staticmethod
    def huber_loss(predictions, targets, delta=1.0):
        """
        Huber loss - robust to outliers.
        
        L = { ½r² if |r| ≤ δ
            { δ|r| - ½δ² otherwise
        """
        return F.huber_loss(predictions, targets, delta=delta)

class LinearRegression:
    """
    Closed-form linear regression.
    
    β̂ = (X'X)⁻¹X'y
    """
    
    def fit(self, X, y):
        # Add bias term
        X = np.column_stack([np.ones(len(X)), X])
        
        # Closed-form solution
        self.beta = np.linalg.solve(X.T @ X, X.T @ y)
        return self
    
    def predict(self, X):
        X = np.column_stack([np.ones(len(X)), X])
        return X @ self.beta

class LogisticRegression:
    """
    Logistic regression via gradient descent.
    
    P(y=1|x) = σ(w'x + b)
    L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
    """
    
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        
        for _ in range(self.max_iter):
            # Forward pass
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)
            
            # Gradients
            error = y_pred - y
            grad_w = (1/n) * X.T @ error
            grad_b = (1/n) * np.sum(error)
            
            # Update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        
        return self
    
    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

class NeuralNetClassifier(nn.Module):
    """Multi-layer perceptron for classification."""
    
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def train_model(self, train_loader, epochs=100, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                logits = self(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")
```

---

## 📊 Loss Function Comparison

| Loss | Formula | Use Case | Properties |
|------|---------|----------|------------|
| **Cross-Entropy** | \(-\sum y_c \log \hat{y}_c\) | Classification | Convex, smooth |
| **MSE** | \(\frac{1}{n}\sum(y-\hat{y})^2\) | Regression | Penalizes large errors |
| **MAE** | \(\frac{1}{n}\sum\|y-\hat{y}\|\) | Robust regression | Robust to outliers |
| **Huber** | MSE if small, MAE if large | Robust regression | Best of both |
| **Hinge** | \(\max(0, 1-y\hat{y})\) | SVM | Margin-based |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | ESL | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 🎓 | Stanford CS229 | [Course](http://cs229.stanford.edu/) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../01_self_supervised/">⬅️ Prev: Self-Supervised</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Learning Frameworks</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_unsupervised/">Next: Unsupervised ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
