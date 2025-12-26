<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Matrix%20Factorization&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=32&desc=NMF%20·%20Recommender%20Systems%20·%20Topic%20Modeling&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.05_Matrix_Factorization-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-NMF_Recommenders-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Matrix factorization decomposes a matrix into simpler factors.** This is the foundation of recommender systems, topic modeling, and signal separation.

- 🎬 **Recommenders**: User-item ratings ≈ User preferences × Item features
- 📰 **Topic Modeling**: Document-word matrix ≈ Document-topic × Topic-word
- ➕ **NMF**: Non-negative factorization for interpretable parts

---

## 📑 Table of Contents

1. [Overview](#1-overview)
2. [Low-Rank Matrix Factorization](#2-low-rank-matrix-factorization)
3. [Non-Negative Matrix Factorization](#3-non-negative-matrix-factorization-nmf)
4. [Recommender Systems](#4-recommender-systems)
5. [Code Implementation](#5-code-implementation)
6. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/matrix-factorization-recommender.svg" width="100%">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MATRIX FACTORIZATION FOR RECOMMENDATIONS                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   RATINGS MATRIX R        ≈      USER FACTORS U    ×    ITEM FACTORS V      │
│   (n_users × n_items)           (n_users × k)          (k × n_items)        │
│                                                                              │
│   ┌─────────────────┐       ┌───────────┐       ┌─────────────────┐        │
│   │ ? 5 ? 3 ? 1 ? │       │ u₁ᵀ      │       │ v₁ v₂ ... vₘ   │        │
│   │ 4 ? ? ? 2 ? 5 │   ≈   │ u₂ᵀ      │   ×   │                 │        │
│   │ ? ? 3 ? ? 4 ? │       │ ...       │       │    k × m        │        │
│   │ 1 ? ? 5 ? ? 2 │       │ uₙᵀ      │       │                 │        │
│   └─────────────────┘       └───────────┘       └─────────────────┘        │
│      Known ratings           User latent         Item latent                │
│      ? = unknown             features            features                   │
│                                                                              │
│   PREDICTION:                                                                │
│   ─────────────                                                              │
│   r̂ᵢⱼ = uᵢᵀ vⱼ = Σₖ uᵢₖ vₖⱼ                                                │
│                                                                              │
│   User i's preference for item j = dot product of their latent vectors      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Overview

### 📌 General Problem

Given matrix $R \in \mathbb{R}^{m \times n}$, find factors:

$$R \approx UV^T$$

where $U \in \mathbb{R}^{m \times k}$ and $V \in \mathbb{R}^{n \times k}$ with $k \ll \min(m, n)$.

### 📐 Relation to SVD

SVD gives the optimal low-rank approximation:
$$A = U\Sigma V^T \approx U_k \Sigma_k V_k^T$$

But SVD requires:
- Complete matrix (no missing entries)
- No constraints (allows negative values)

Matrix factorization methods relax these.

---

## 2. Low-Rank Matrix Factorization

### 📌 Problem Formulation

$$\min_{U, V} \|R - UV^T\|_F^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$$

- Frobenius norm: $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2}$
- Regularization $\lambda$ prevents overfitting

### 🔍 Alternating Least Squares (ALS)

```
Initialize U, V randomly

Repeat until convergence:
    # Fix V, solve for U
    For each user i:
        uᵢ = (VᵀV + λI)⁻¹ Vᵀ rᵢ
        where rᵢ = ratings by user i
    
    # Fix U, solve for V
    For each item j:
        vⱼ = (UᵀU + λI)⁻¹ Uᵀ rⱼ
        where rⱼ = ratings of item j
```

**Why ALS?**
- With one factor fixed, problem is convex (quadratic)
- Each sub-problem has closed-form solution
- Easily parallelizable across users/items

---

## 3. Non-Negative Matrix Factorization (NMF)

### 📌 Problem

$$\min_{U \geq 0, V \geq 0} \|R - UV^T\|_F^2$$

subject to $U_{ij} \geq 0$ and $V_{ij} \geq 0$.

### 🔍 Why Non-Negativity?

```
Non-negativity gives INTERPRETABLE parts-based representation:

Example: Face recognition
  R = images (pixels × images)
  U = basis images (parts: eyes, nose, mouth)
  V = coefficients (how much of each part)

  Face ≈ 0.3×(eyes) + 0.5×(nose) + 0.2×(mouth)

With negative values (like SVD), parts could "cancel out"
→ Less interpretable
```

### 📐 Multiplicative Update Rules

```
Initialize U, V with random positive values

Repeat until convergence:
    # Update U
    Uᵢₖ ← Uᵢₖ × (RV)ᵢₖ / (UV^TV)ᵢₖ
    
    # Update V
    Vⱼₖ ← Vⱼₖ × (R^TU)ⱼₖ / (VU^TU)ⱼₖ

These updates:
1. Keep values non-negative (positive × positive)
2. Decrease the objective function
3. Converge to a local minimum
```

### 🔍 Proof: Updates Decrease Objective

```
The objective D = ‖R - UV^T‖²_F

For the update rule Uᵢₖ ← Uᵢₖ × (RV)ᵢₖ / (UV^TV)ᵢₖ:

Using auxiliary function technique (Lee & Seung, 2001):

1. Define G(U, U') ≥ D(U) with equality at U = U'

2. New U minimizes G(U, U^old)

3. D(U^new) ≤ G(U^new, U^old) ≤ G(U^old, U^old) = D(U^old)

Therefore objective decreases (or stays same) at each step.  ∎
```

---

## 4. Recommender Systems

### 📌 Problem Setup

- Users: $m$ users
- Items: $n$ items
- Ratings: $R_{ij}$ = rating user $i$ gave item $j$ (only some known)

### 📐 Matrix Factorization Model

$$\hat{r}_{ij} = b + b_i + b_j + \mathbf{u}_i^T \mathbf{v}_j$$

where:
- $b$: global bias
- $b_i$: user bias
- $b_j$: item bias
- $\mathbf{u}_i$: user latent factors
- $\mathbf{v}_j$: item latent factors

### 🔍 Optimization

Minimize over observed ratings $\Omega$:
$$\min \sum_{(i,j) \in \Omega} (r_{ij} - \hat{r}_{ij})^2 + \lambda(\|U\|_F^2 + \|V\|_F^2 + \sum_i b_i^2 + \sum_j b_j^2)$$

### 💡 Example: Netflix

```
User 1 watches: Action, Sci-Fi movies (high ratings)
User 2 watches: Romance, Comedy movies (high ratings)

Latent factors might capture:
  Dimension 1: Action vs Romance preference
  Dimension 2: Old vs New movies
  Dimension 3: Serious vs Light tone
  ...

User vectors encode preferences along these dimensions
Movie vectors encode characteristics along same dimensions
```

---

## 5. Code Implementation

```python
import numpy as np
from sklearn.decomposition import NMF

class MatrixFactorization:
    """Matrix factorization for recommender systems."""
    
    def __init__(self, n_factors=10, learning_rate=0.01, reg=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = reg
        self.n_epochs = n_epochs
    
    def fit(self, R):
        """
        Fit the model using Stochastic Gradient Descent.
        R: Rating matrix (0 = missing)
        """
        self.n_users, self.n_items = R.shape
        
        # Initialize factors
        self.U = np.random.normal(0, 0.1, (self.n_users, self.n_factors))
        self.V = np.random.normal(0, 0.1, (self.n_items, self.n_factors))
        
        # Biases
        self.b = np.mean(R[R > 0])
        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_items)
        
        # Get observed indices
        observed = np.where(R > 0)
        
        for epoch in range(self.n_epochs):
            for i, j in zip(*observed):
                # Prediction
                pred = self.predict_single(i, j)
                error = R[i, j] - pred
                
                # Update biases
                self.b_u[i] += self.lr * (error - self.reg * self.b_u[i])
                self.b_i[j] += self.lr * (error - self.reg * self.b_i[j])
                
                # Update factors
                self.U[i] += self.lr * (error * self.V[j] - self.reg * self.U[i])
                self.V[j] += self.lr * (error * self.U[i] - self.reg * self.V[j])
            
            if epoch % 10 == 0:
                loss = self.compute_loss(R, observed)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict_single(self, i, j):
        """Predict rating for user i, item j."""
        return self.b + self.b_u[i] + self.b_i[j] + np.dot(self.U[i], self.V[j])
    
    def predict(self):
        """Predict all ratings."""
        return self.b + self.b_u[:, np.newaxis] + self.b_i + self.U @ self.V.T
    
    def compute_loss(self, R, observed):
        """Compute regularized loss."""
        loss = 0
        for i, j in zip(*observed):
            loss += (R[i, j] - self.predict_single(i, j))**2
        loss += self.reg * (np.sum(self.U**2) + np.sum(self.V**2))
        return loss

def nmf_topic_modeling(document_term_matrix, n_topics=10):
    """
    NMF for topic modeling.
    
    Document-Term Matrix ≈ Document-Topic × Topic-Term
    """
    nmf = NMF(n_components=n_topics, init='nndsvd', random_state=42)
    
    # W: document-topic matrix
    # H: topic-term matrix
    W = nmf.fit_transform(document_term_matrix)
    H = nmf.components_
    
    return W, H, nmf

# Example usage
np.random.seed(42)

# Simulate ratings (0 = missing)
R = np.random.randint(0, 6, (100, 50))
R[np.random.rand(*R.shape) < 0.7] = 0  # 70% missing

mf = MatrixFactorization(n_factors=10, n_epochs=100)
mf.fit(R)

# Predict missing ratings
predictions = mf.predict()
print(f"Predicted ratings range: [{predictions.min():.2f}, {predictions.max():.2f}]")
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📄 | [Matrix Factorization for Recommenders](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) | Netflix paper |
| 📄 | [NMF (Lee & Seung)](https://www.nature.com/articles/44565) | Original NMF |
| 🎥 | [Recommender Systems](https://www.coursera.org/learn/recommender-systems) | Coursera |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Eigenvalues](../04_eigenvalues/README.md) | [Linear Algebra](../README.md) | [Matrix Properties](../06_matrix_properties/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
