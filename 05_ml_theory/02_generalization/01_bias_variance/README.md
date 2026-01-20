<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_complexity/">Next: Complexity ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Bias-Variance%20Tradeoff&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/bias-variance-tradeoff.svg" width="100%">

*Caption: The bias-variance tradeoff shows how model complexity affects generalization. Simple models (left) have high bias and underfit, while complex models (right) have high variance and overfit. The optimal model balances both to minimize total error.*

---

## 📂 Overview

The bias-variance tradeoff is **the most fundamental concept in ML theory**. It explains why more complex models don't always perform better and provides the theoretical foundation for regularization, model selection, and ensemble methods.

---

## 📐 Mathematical Decomposition

### Setting

Let the true relationship be:

```math
y = f(x) + \varepsilon, \quad \text{where } \varepsilon \sim \mathcal{N}(0, \sigma^2)
```

Given training data \(\mathcal{D}\), we learn an estimator \(\hat{f}(x; \mathcal{D})\).

### Theorem: Bias-Variance Decomposition

**For squared error loss, the expected prediction error decomposes as:**

```math
\mathbb{E}_{\mathcal{D}, \varepsilon}\left[(y - \hat{f}(x))^2\right] = \underbrace{\text{Bias}^2(\hat{f}(x))}_{\text{systematic error}} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{sensitivity to data}} + \underbrace{\sigma^2}_{\text{irreducible noise}}
```

where:

```math
\text{Bias}(\hat{f}(x)) = \mathbb{E}_{\mathcal{D}}[\hat{f}(x)] - f(x)
\text{Var}(\hat{f}(x)) = \mathbb{E}_{\mathcal{D}}\left[(\hat{f}(x) - \mathbb{E}_{\mathcal{D}}[\hat{f}(x)])^2\right]
```

---

### Complete Proof

**Step 1: Expand the squared error**

```math
\mathbb{E}[(y - \hat{f})^2] = \mathbb{E}[(y - f + f - \hat{f})^2]
= \mathbb{E}[(y - f)^2] + \mathbb{E}[(f - \hat{f})^2] + 2\mathbb{E}[(y - f)(f - \hat{f})]
```

**Step 2: Evaluate the cross-term**

Since \(\varepsilon = y - f\) is independent of \(\hat{f}\) and \(\mathbb{E}[\varepsilon] = 0\):

```math
\mathbb{E}[(y - f)(f - \hat{f})] = \mathbb{E}[\varepsilon(f - \hat{f})] = \mathbb{E}[\varepsilon] \cdot \mathbb{E}[f - \hat{f}] = 0
```

**Step 3: First term (noise)**

```math
\mathbb{E}[(y - f)^2] = \mathbb{E}[\varepsilon^2] = \sigma^2
```

**Step 4: Second term (decompose further)**

Let \(\bar{f} = \mathbb{E}_{\mathcal{D}}[\hat{f}]\) (expected prediction across datasets).

```math
\mathbb{E}[(f - \hat{f})^2] = \mathbb{E}[(f - \bar{f} + \bar{f} - \hat{f})^2]
= (f - \bar{f})^2 + \mathbb{E}[(\bar{f} - \hat{f})^2] + 2(f - \bar{f})\mathbb{E}[\bar{f} - \hat{f}]
```

**Step 5: The cross-term vanishes**

```math
\mathbb{E}[\bar{f} - \hat{f}] = \bar{f} - \mathbb{E}[\hat{f}] = \bar{f} - \bar{f} = 0
```

**Step 6: Final result**

```math
\mathbb{E}[(f - \hat{f})^2] = \underbrace{(f - \bar{f})^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat{f} - \bar{f})^2]}_{\text{Variance}}
```

Therefore:

```math
\boxed{\mathbb{E}[(y - \hat{f})^2] = \text{Bias}^2 + \text{Variance} + \sigma^2}
```

---

## 📊 The Tradeoff Visualized

```
Expected Error
      ↑
      |   
      |    ╲                           ╱
      |     ╲    Total Error          ╱
      |      ╲                       ╱
      |       ╲    ____________     ╱
      |        ╲  ╱            ╲   ╱
      |         ╲╱              ╲ ╱
      |         ╱╲   Variance   ╱╲
      |        ╱  ╲            ╱  ╲
      |  Bias²╱    ╲__________╱
      |------------------------------→ Model Complexity
           Simple              Complex
         (underfit)           (overfit)
                    ↑
                 Optimal
```

| Model Complexity | Bias | Variance | Total Error | Regime |
|-----------------|------|----------|-------------|--------|
| **Too Simple** | High | Low | High | Underfitting |
| **Optimal** | Medium | Medium | **Minimum** | Good generalization |
| **Too Complex** | Low | High | High | Overfitting |

---

## 🔑 Key Insights

### High Bias (Underfitting)

**Definition:** The model is too simple to capture the underlying pattern.

```math
\text{Bias}^2 = (\mathbb{E}[\hat{f}] - f)^2 \gg 0
```

**Symptoms:**
- High training error
- High test error  
- Training error ≈ Test error
- Learning curve plateaus early

**Examples:**
- Linear model for quadratic data
- Shallow network for complex patterns
- Insufficient features

**Mathematical Example:**

True function: \(f(x) = x^2\)

Model: \(\hat{f}(x) = ax + b\) (linear)

Best linear fit minimizes \(\int (x^2 - ax - b)^2 dx\), but cannot represent curvature.

```math
\text{Bias}^2 = \mathbb{E}_x[(x^2 - ax^* - b^*)^2] > 0
```

---

### High Variance (Overfitting)

**Definition:** The model is too sensitive to the specific training data.

```math
\text{Var}(\hat{f}) = \mathbb{E}[(\hat{f} - \mathbb{E}[\hat{f}])^2] \gg 0
```

**Symptoms:**
- Low training error
- High test error
- Training error << Test error
- Large gap between train/test curves

**Examples:**
- High-degree polynomial
- Deep network without regularization
- k-NN with k=1
- Decision tree with no pruning

**Mathematical Example:**

Consider polynomial regression of degree \(d\) on \(n\) points:

```math
\text{Var}(\hat{f}(x)) = \sigma^2 \cdot \mathbf{x}^\top (\mathbf{X}^\top\mathbf{X})^{-1} \mathbf{x}
```

As \(d \to n\), \((\mathbf{X}^\top\mathbf{X})^{-1}\) becomes ill-conditioned and variance explodes.

---

## 📐 Formal Analysis for Linear Regression

For linear regression \(\hat{f}(x) = x^\top \hat{\beta}\) where \(\hat{\beta} = (X^\top X)^{-1} X^\top y\):

### Bias

```math
\mathbb{E}[\hat{\beta}] = (X^\top X)^{-1} X^\top \mathbb{E}[y] = (X^\top X)^{-1} X^\top X\beta = \beta
```

**Linear regression is unbiased** when the model is correctly specified.

### Variance

```math
\text{Var}(\hat{\beta}) = (X^\top X)^{-1} X^\top \text{Var}(y) X (X^\top X)^{-1} = \sigma^2 (X^\top X)^{-1}
```

**Prediction variance at point x:**

```math
\text{Var}(\hat{f}(x)) = \sigma^2 x^\top (X^\top X)^{-1} x
```

### Ridge Regression Reduces Variance

With L2 regularization: \(\hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y\)

**Bias (now non-zero):**

```math
\text{Bias}(\hat{\beta}_{\text{ridge}}) = -\lambda (X^\top X + \lambda I)^{-1} \beta
```

**Variance (reduced):**

```math
\text{Var}(\hat{\beta}_{\text{ridge}}) = \sigma^2 (X^\top X + \lambda I)^{-1} X^\top X (X^\top X + \lambda I)^{-1}
```

**Theorem:** There exists \(\lambda^* > 0\) such that \(\text{MSE}(\hat{\beta}_{\text{ridge}}) < \text{MSE}(\hat{\beta}_{\text{OLS}})\).

---

## 🌍 Modern Deep Learning Perspective

### Classical View
```math
\text{Test Error} = \text{Bias}^2 + \text{Variance} + \sigma^2
```

Increasing model complexity: Bias ↓, Variance ↑

### Double Descent Phenomenon

Modern overparameterized networks exhibit **double descent**:

```
Test Error
    ↑
    |   
    |    ╲        ↑              
    |     ╲  Classical|     /‾‾‾‾↘
    |      ╲    regime|    /      ‾‾‾→
    |       ╲_________|___/
    |                 |   ↑
    |                 |   Interpolation
    |                 |   threshold
    |------------------------------→ # Parameters
         Underparameterized  Overparameterized
```

**Why Double Descent?**
1. **Implicit regularization** from SGD
2. **Flat minima** have better generalization
3. **Lottery ticket hypothesis** - sparse subnetworks generalize
4. **Neural tangent kernel** regime in wide networks

---

## 💻 Code Examples

### Demonstrating Bias-Variance with Bootstrap

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

def true_function(x):
    """Ground truth function f(x) = sin(2πx)."""
    return np.sin(2 * np.pi * x)

def bias_variance_demo(n_samples=30, n_bootstrap=100, noise_std=0.3):
    """
    Demonstrate bias-variance tradeoff via bootstrap.
    
    For each polynomial degree:
    - Bias² = (E[f̂] - f)²  (systematic error)
    - Var = E[(f̂ - E[f̂])²] (sensitivity to data)
    """
    np.random.seed(42)
    
    # Generate training data
    X = np.random.rand(n_samples)
    y = true_function(X) + noise_std * np.random.randn(n_samples)
    
    # Test points
    X_test = np.linspace(0, 1, 100)
    y_true = true_function(X_test)
    
    degrees = [1, 4, 15]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    results = []
    
    for ax, degree in zip(axes, degrees):
        # Bootstrap predictions: simulate training on different datasets
        predictions = np.zeros((n_bootstrap, len(X_test)))
        
        for b in range(n_bootstrap):
            # Resample with replacement (different training set)
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]
            
            # Fit polynomial of given degree
            model = make_pipeline(
                PolynomialFeatures(degree), 
                LinearRegression()
            )
            model.fit(X_boot.reshape(-1, 1), y_boot)
            predictions[b] = model.predict(X_test.reshape(-1, 1))
        
        # E[f̂(x)] - average prediction across bootstrap samples
        mean_pred = predictions.mean(axis=0)
        
        # Bias² = (E[f̂] - f)²
        bias_squared = (mean_pred - y_true) ** 2
        
        # Variance = E[(f̂ - E[f̂])²]
        variance = predictions.var(axis=0)
        
        # Total expected error (ignoring irreducible noise)
        total_error = bias_squared + variance
        
        # Store results
        results.append({
            'degree': degree,
            'bias_sq': bias_squared.mean(),
            'variance': variance.mean(),
            'total': total_error.mean()
        })
        
        # Visualization
        for pred in predictions[:20]:
            ax.plot(X_test, pred, alpha=0.1, color='steelblue')
        ax.plot(X_test, mean_pred, 'b-', linewidth=2, label='E[f̂(x)]')
        ax.plot(X_test, y_true, 'r--', linewidth=2, label='f(x) true')
        ax.scatter(X, y, color='black', s=20, zorder=5)
        ax.set_title(f'Degree {degree}\n'
                    f'Bias²={bias_squared.mean():.3f}, '
                    f'Var={variance.mean():.3f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('bias_variance_demo.png', dpi=150)
    plt.show()
    
    return results

# Run demonstration
results = bias_variance_demo()
for r in results:
    print(f"Degree {r['degree']:2d}: Bias²={r['bias_sq']:.4f}, "
          f"Var={r['variance']:.4f}, Total={r['total']:.4f}")
```

### Bias-Variance Decomposition Function

```python
def bias_variance_decomposition(model_class, X_train, y_train, X_test, y_test,
                                 n_bootstrap=200, **model_params):
    """
    Compute bias-variance decomposition via bootstrap.
    
    Mathematical formulation:
    E[(y - f̂)²] = Bias²(f̂) + Var(f̂) + σ²
    
    Args:
        model_class: Scikit-learn compatible model class
        X_train, y_train: Training data
        X_test, y_test: Test data (y_test approximates f(x))
        n_bootstrap: Number of bootstrap samples
        **model_params: Model hyperparameters
        
    Returns:
        dict with bias_squared, variance, expected_error, mse
    """
    n_test = len(X_test)
    predictions = np.zeros((n_bootstrap, n_test))
    
    for b in range(n_bootstrap):
        # Bootstrap sample (different training set)
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_boot, y_boot = X_train[idx], y_train[idx]
        
        # Fit model
        model = model_class(**model_params)
        model.fit(X_boot, y_boot)
        predictions[b] = model.predict(X_test)
    
    # E[f̂(x)]
    mean_pred = predictions.mean(axis=0)
    
    # Bias² = (E[f̂] - f)²
    # Note: y_test is our proxy for f(x)
    bias_squared = (mean_pred - y_test) ** 2
    
    # Variance = E[(f̂ - E[f̂])²]
    variance = predictions.var(axis=0)
    
    # Expected squared error
    expected_error = bias_squared + variance
    
    # Actual MSE (includes noise)
    mse = ((predictions - y_test) ** 2).mean()
    
    return {
        'bias_squared': bias_squared.mean(),
        'variance': variance.mean(),
        'expected_error': expected_error.mean(),
        'mse': mse
    }

# Example: Decision Tree depth analysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

# Generate data
X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train, X_test = X[:400], X[400:]
y_train, y_test = y[:400], y[400:]

print("Decision Tree Bias-Variance Analysis")
print("-" * 50)
for max_depth in [1, 3, 5, 10, None]:
    result = bias_variance_decomposition(
        DecisionTreeRegressor,
        X_train, y_train, X_test, y_test,
        max_depth=max_depth, random_state=42
    )
    depth_str = str(max_depth) if max_depth else '∞'
    print(f"Depth {depth_str:>3}: Bias²={result['bias_squared']:8.2f}, "
          f"Var={result['variance']:8.2f}, Total={result['expected_error']:8.2f}")
```

### Model Selection via Cross-Validation

```python
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt

def plot_bias_variance_tradeoff(X, y, model_class, param_name, param_values,
                                 cv=5, scoring='neg_mean_squared_error'):
    """
    Visualize bias-variance tradeoff using cross-validation.
    
    Train error ≈ proxy for bias (underfitting)
    Val - Train gap ≈ proxy for variance (overfitting)
    """
    train_scores = []
    val_scores = []
    val_stds = []
    
    for param in param_values:
        model = model_class(**{param_name: param})
        
        # Training score (fit on all data)
        model.fit(X, y)
        train_score = -model_class(**{param_name: param}).fit(X, y).score(X, y)
        train_scores.append(-scoring_to_mse(model, X, y))
        
        # Validation scores via CV
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        val_scores.append(-cv_scores.mean())
        val_stds.append(cv_scores.std())
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, train_scores, 'b-o', label='Training Error', linewidth=2)
    ax.plot(param_values, val_scores, 'r-o', label='Validation Error', linewidth=2)
    ax.fill_between(param_values, 
                    np.array(val_scores) - np.array(val_stds),
                    np.array(val_scores) + np.array(val_stds),
                    alpha=0.2, color='red')
    
    # Mark optimal
    best_idx = np.argmin(val_scores)
    ax.axvline(param_values[best_idx], color='green', linestyle='--',
               label=f'Optimal {param_name}={param_values[best_idx]}')
    
    # Annotate regions
    ax.annotate('High Bias\n(Underfitting)', 
                xy=(param_values[0], val_scores[0]),
                xytext=(param_values[1], max(val_scores)*0.9),
                fontsize=10, ha='center')
    ax.annotate('High Variance\n(Overfitting)', 
                xy=(param_values[-1], val_scores[-1]),
                xytext=(param_values[-2], max(val_scores)*0.9),
                fontsize=10, ha='center')
    
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Mean Squared Error', fontsize=12)
    ax.set_title('Bias-Variance Tradeoff via Cross-Validation', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return param_values[best_idx]
```

---

## 🔗 Strategies to Reduce Bias and Variance

| Strategy | Reduces | Mechanism |
|----------|---------|-----------|
| **More features** | Bias | Increases model capacity |
| **Complex model** | Bias | Better approximation of f(x) |
| **More training data** | Variance | Better estimate of E[f̂] |
| **Regularization (L1/L2)** | Variance | Constrains hypothesis space |
| **Bagging/Ensembles** | Variance | Averages out fluctuations |
| **Dropout** | Variance | Implicit ensemble |
| **Early stopping** | Variance | Limits effective complexity |
| **Cross-validation** | Both | Selects optimal complexity |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | ESL Ch. 7: Model Assessment | [StatLearn](https://hastie.su.domains/ElemStatLearn/) |
| 📖 | Bishop PRML Ch. 3.2 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📄 | Deep Double Descent | [arXiv](https://arxiv.org/abs/1912.02292) |
| 📄 | Reconciling Modern ML | [arXiv](https://arxiv.org/abs/1903.07571) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../02_complexity/">Next: Complexity ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
