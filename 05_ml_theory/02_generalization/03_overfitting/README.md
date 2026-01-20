<!-- Navigation -->
<p align="center">
  <a href="../02_complexity/">⬅️ Prev: Complexity</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_regularization/">Next: Regularization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Overfitting&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/overfitting-visual.svg" width="100%">

*Caption: Overfitting occurs when a model memorizes training data including noise, failing to generalize.*

---

## 📂 Overview

**Overfitting** is one of the central problems in machine learning. It occurs when a model fits training data too well, capturing noise rather than the underlying pattern.

---

## 📐 Mathematical Framework

### Generalization Gap

The **generalization gap** is the difference between test and training error:

$$
\text{Gap} = R(h) - \hat{R}(h) = \mathbb{E}_{(x,y) \sim P}[\ell(h(x), y)] - \frac{1}{n}\sum_{i=1}^n \ell(h(x_i), y_i)
$$

**Overfitting indicator:** $\hat{R}(h) \ll R(h)$

### Bias-Variance Connection

For squared error loss:

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}
$$

- **Underfitting:** High bias (model too simple)
- **Overfitting:** High variance (model too complex)

### VC Theory Bound

**Theorem:** With probability $\geq 1 - \delta$:

$$
R(h) \leq \hat{R}(h) + O\left(\sqrt{\frac{d \log(n/d) + \log(1/\delta)}{n}}\right)
$$

**Implication:** Overfitting risk increases with $d/n$ (complexity/data ratio).

---

## 📐 Double Descent Phenomenon

### Classical vs Modern View

**Classical:** Error U-shaped with complexity
**Modern:** After interpolation, error decreases again!

$$
\text{Test Error} = \begin{cases}
\searrow & \text{(underfit → optimal)} \\
\nearrow & \text{(optimal → interpolation)} \\
\searrow & \text{(interpolation → overparameterized)}
\end{cases}
$$

### Implicit Regularization

**Theorem (Minimum Norm Interpolation):** For linear regression with $n < d$, gradient descent converges to:

$$
\hat{w} = X^\top(XX^\top)^{-1}y = \arg\min_w \|w\|_2 \text{ s.t. } Xw = y
$$

This is the **minimum-norm interpolating solution**.

---

## 📐 Solutions to Overfitting

### 1. Regularization

**L2 Regularization (Ridge):**

$$
\min_w \frac{1}{n}\sum_{i=1}^n \ell(w^\top x_i, y_i) + \lambda\|w\|_2^2
$$

**L1 Regularization (Lasso):**

$$
\min_w \frac{1}{n}\sum_{i=1}^n \ell(w^\top x_i, y_i) + \lambda\|w\|_1
$$

### 2. Early Stopping

**Theorem (Early Stopping ≈ Regularization):** For gradient descent on linear regression:

$$
w^{(t)} \approx w_\lambda \quad \text{where } \lambda = \frac{1}{\eta t}
$$

Early stopping is equivalent to implicit L2 regularization.

### 3. Dropout

At training time, randomly drop neurons with probability $p$:

$$
\tilde{h}_i = \frac{1}{1-p} m_i \cdot h_i, \quad m_i \sim \text{Bernoulli}(1-p)
$$

**Theoretical interpretation:** Approximate Bayesian inference / ensemble averaging.

---

## 💻 Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import Pipeline

def demonstrate_overfitting():
    """Demonstrate overfitting with polynomial regression."""
    np.random.seed(42)
    
    # Generate data
    n = 30
    X = np.sort(np.random.uniform(0, 1, n)).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + 0.1 * np.random.randn(n)
    
    X_test = np.linspace(0, 1, 100).reshape(-1, 1)
    y_true = np.sin(2 * np.pi * X_test).ravel()
    
    degrees = [1, 4, 15]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, degree in zip(axes, degrees):

        # Fit polynomial
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        X_test_poly = poly.transform(X_test)
        
        model = LinearRegression()
        model.fit(X_poly, y)
        
        y_pred = model.predict(X_test_poly)
        train_error = np.mean((model.predict(X_poly) - y) ** 2)
        test_error = np.mean((y_pred - y_true) ** 2)
        
        ax.scatter(X, y, color='blue', label='Training data')
        ax.plot(X_test, y_true, 'g--', label='True function')
        ax.plot(X_test, y_pred, 'r-', label='Prediction')
        ax.set_title(f'Degree {degree}\nTrain: {train_error:.4f}, Test: {test_error:.4f}')
        ax.legend()
        ax.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()

def early_stopping_demo():
    """Demonstrate early stopping."""
    np.random.seed(42)
    
    # Generate data
    n_train, n_val = 80, 20
    X = np.random.randn(n_train + n_val, 10)
    y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_train + n_val)
    
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    
    # Train with early stopping
    w = np.zeros(10)
    lr = 0.01
    max_epochs = 1000
    patience = 20
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    best_w = w.copy()
    patience_counter = 0
    
    for epoch in range(max_epochs):

        # Gradient descent step
        pred_train = X_train @ w
        grad = (2/n_train) * X_train.T @ (pred_train - y_train)
        w = w - lr * grad
        
        # Compute losses
        train_loss = np.mean((X_train @ w - y_train) ** 2)
        val_loss = np.mean((X_val @ w - y_val) ** 2)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_w = w.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best: epoch {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Early Stopping Demo')
    plt.show()

def regularization_effect():
    """Show effect of regularization strength."""
    np.random.seed(42)
    
    # High-dimensional data (overfitting scenario)
    n, d = 50, 100
    X = np.random.randn(n, d)
    true_w = np.zeros(d)
    true_w[:5] = [1, -1, 0.5, -0.5, 0.25]  # Only 5 non-zero
    y = X @ true_w + 0.1 * np.random.randn(n)
    
    alphas = np.logspace(-4, 2, 20)
    train_errors = []
    test_errors = []
    
    # Split data
    X_train, X_test = X[:40], X[40:]
    y_train, y_test = y[:40], y[40:]
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        train_errors.append(np.mean((model.predict(X_train) - y_train) ** 2))
        test_errors.append(np.mean((model.predict(X_test) - y_test) ** 2))
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_errors, 'b-', label='Training Error')
    plt.semilogx(alphas, test_errors, 'r-', label='Test Error')
    plt.xlabel('Regularization Strength (α)')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Regularization: Training vs Test Error')
    
    best_alpha = alphas[np.argmin(test_errors)]
    plt.axvline(x=best_alpha, color='g', linestyle='--', label=f'Optimal α = {best_alpha:.4f}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    demonstrate_overfitting()
    early_stopping_demo()
    regularization_effect()
```

---

## 📊 Signs of Overfitting

| Indicator | Training | Validation | Action |
|-----------|----------|------------|--------|
| Perfect fit | ~0% error | High error | Reduce complexity |
| Gap growing | Decreasing | Increasing | Early stop |
| Large weights | Good fit | Poor fit | Add regularization |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | ESL Ch. 7 | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 📄 | Double Descent | [arXiv](https://arxiv.org/abs/1912.02292) |
| 📄 | Dropout | [Srivastava et al.](https://jmlr.org/papers/v15/srivastava14a.html) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../02_complexity/">⬅️ Prev: Complexity</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_regularization/">Next: Regularization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
