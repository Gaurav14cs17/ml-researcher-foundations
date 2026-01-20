<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Distance%20Metrics&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Measuring%20Similarity%20in%20High-Dimensional%20Spaces&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-03_Distance_Metrics-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-L1_L2_Cosine_Mahalanobis-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Distance metrics quantify similarity between data points.** The right metric depends on your data type and what notion of "similar" matters for your task.

- 📏 **Euclidean (L2)**: Straight-line distance, most common

- 🏙️ **Manhattan (L1)**: Grid distance, robust to outliers

- 🧭 **Cosine**: Angle between vectors, scale-invariant

- 📊 **Mahalanobis**: Accounts for feature correlations

---

## 📑 Table of Contents

1. [Metric Space Properties](#1-metric-space-properties)

2. [Lp Norms and Distances](#2-lp-norms-and-distances)

3. [Cosine Similarity](#3-cosine-similarity)

4. [Mahalanobis Distance](#4-mahalanobis-distance)

5. [Probabilistic Distances](#5-probabilistic-distances)

6. [Code Implementation](#6-code-implementation)

7. [Choosing the Right Metric](#7-choosing-the-right-metric)

8. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/distance-metrics-complete.svg" width="100%">

```
+-----------------------------------------------------------------------------+
|                    DISTANCE METRICS COMPARISON                               |
+-----------------------------------------------------------------------------+
|                                                                              |
|   L2 (Euclidean)           L1 (Manhattan)           Cosine                  |
|   --------------           ---------------          ------                  |
|                                                                              |
|        ●                        ●                      ╲ ●                  |
|       /|                        |                       ╲ ↖ angle θ        |
|      / |  d = √(Δx²+Δy²)       |   d = |Δx|+|Δy|        ╲                   |
|     /  |                       +---●                     ● origin          |
|    ●---+                                                                    |
|                                                                              |
|   "Crow flies"             "City blocks"            "Direction only"        |
|   Sensitive to scale       More robust             Scale invariant          |
|                                                                              |
|   +===================================================================+     |
|   |  CURSE OF DIMENSIONALITY:                                          |     |
|   |  In high dimensions, all points become "equally far" from each    |     |
|   |  other. Cosine similarity often works better than L2.             |     |
|   +===================================================================+     |
|                                                                              |
+-----------------------------------------------------------------------------+

```

---

## 1. Metric Space Properties

### 📌 Definition

A function $d: X \times X \to \mathbb{R}$ is a **metric** if it satisfies:

| Property | Formula | Intuition |
|----------|---------|-----------|
| **Non-negativity** | $d(x, y) \geq 0$ | Distance is never negative |
| **Identity** | $d(x, y) = 0 \Leftrightarrow x = y$ | Zero distance means same point |
| **Symmetry** | $d(x, y) = d(y, x)$ | A to B = B to A |
| **Triangle inequality** | $d(x, z) \leq d(x, y) + d(y, z)$ | Direct path is shortest |

### ⚠️ Note on Similarity vs Distance

```
Similarity ≠ Distance

• Similarity: Higher = more similar (range often [0, 1] or [-1, 1])
• Distance: Lower = more similar (range [0, ∞))

Conversion examples:
  distance = 1 - similarity           (for similarity in [0, 1])
  distance = 1 - |similarity|         (for similarity in [-1, 1])
  similarity = 1 / (1 + distance)     (always in (0, 1])
  similarity = exp(-distance)         (always in (0, 1])

```

---

## 2. Lp Norms and Distances

### 📌 General Lp Norm

```math
\|x\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}

```

### 📌 Lp Distance

```math
d_p(x, y) = \|x - y\|_p

```

### 📊 Special Cases

| p | Name | Formula | Unit Ball Shape |
|---|------|---------|-----------------|
| 1 | Manhattan/Taxicab | $\sum_i |x_i - y_i|$ | Diamond |
| 2 | Euclidean | $\sqrt{\sum_i (x_i - y_i)^2}$ | Circle/Sphere |
| $\infty$ | Chebyshev/Max | $\max_i |x_i - y_i|$ | Square/Hypercube |

### 🔍 Properties

```
L1 (Manhattan):
• Robust to outliers (linear penalty)
• Good for sparse data
• Encourages sparsity (LASSO uses L1 regularization)

L2 (Euclidean):
• Most common, matches geometric intuition
• Differentiable everywhere (smooth optimization)
• Sensitive to outliers (quadratic penalty)

L∞ (Chebyshev):
• Only considers the largest difference
• Used in minimax optimization
• Robust to differences in low-variance dimensions

```

### 💡 Example

```
x = [1, 2, 3]
y = [4, 5, 6]

L1: |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9

L2: √((1-4)² + (2-5)² + (3-6)²) = √(9 + 9 + 9) = √27 ≈ 5.20

L∞: max(|1-4|, |2-5|, |3-6|) = max(3, 3, 3) = 3

```

---

## 3. Cosine Similarity

### 📌 Definition

```math
\cos(\theta) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}}

```

### 📌 Cosine Distance

```math
d_{\cos}(\mathbf{x}, \mathbf{y}) = 1 - \cos(\theta)

```

### 📐 Properties

```
Range: cos(θ) ∈ [-1, 1]
  • cos(θ) = 1:  Vectors point same direction (identical after scaling)
  • cos(θ) = 0:  Orthogonal (unrelated)
  • cos(θ) = -1: Opposite directions

KEY PROPERTY: Scale invariant!
  cos(x, y) = cos(αx, βy) for any α, β > 0

This makes it perfect for:
  • Text embeddings (document length doesn't matter)
  • User preference vectors
  • Neural network embeddings

```

### 🔍 Relationship to Euclidean Distance

For **normalized vectors** ($\|x\| = \|y\| = 1$):

```math
\|x - y\|_2^2 = 2(1 - \cos(\theta)) = 2 \cdot d_{\cos}(x, y)

```

This is why many embedding methods normalize vectors!

---

## 4. Mahalanobis Distance

### 📌 Definition

```math
d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T \Sigma^{-1} (\mathbf{x} - \mathbf{y})}

```

where $\Sigma$ is the covariance matrix of the data.

### 📐 Properties

```
Key insight: Accounts for correlations between features!

If Σ = I (identity), Mahalanobis = Euclidean

When features are correlated:
• Euclidean treats all directions equally
• Mahalanobis "stretches" space to account for correlations
• Points that seem close in Euclidean may be far in Mahalanobis

Applications:
• Anomaly detection (how many "standard deviations" away?)
• Classification with class-specific covariances
• Gaussian distributions (x ~ N(μ, Σ) → d_M(x, μ) ~ χ distribution)

```

### 💡 Example

```
Consider 2D data with correlation ρ = 0.8 between features:

Points: x = (3, 3), y = (0, 0)

Euclidean: d = √(9 + 9) = √18 ≈ 4.24

With covariance Σ = [[1, 0.8], [0.8, 1]]:
Σ⁻¹ = [[2.78, -2.22], [-2.22, 2.78]]

Mahalanobis: d = √((3,3) · Σ⁻¹ · (3,3)ᵀ) ≈ 2.24

The correlation makes the actual "statistical distance" smaller!

```

---

## 5. Probabilistic Distances

### 📊 For Probability Distributions

| Distance | Formula | Use Case |
|----------|---------|----------|
| **KL Divergence** | $D_{KL}(P \| Q) = \sum_i p_i \log \frac{p_i}{q_i}$ | Compression, VAEs |
| **JS Divergence** | $\frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$ | Symmetric, GANs |
| **Wasserstein** | $\inf_\gamma \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]$ | Optimal transport |
| **Total Variation** | $\frac{1}{2}\sum_i |p_i - q_i|$ | Statistical testing |

### 📐 KL Divergence

```
KL Divergence is NOT symmetric: D_KL(P||Q) ≠ D_KL(Q||P)

Interpretation:
  D_KL(P||Q) = Expected extra bits needed to encode samples from P
               using a code optimized for Q

In ML:
  D_KL(p_data || p_model) → "model misses modes of data"
  D_KL(p_model || p_data) → "model puts mass where data isn't"

Used in:
  • VAE: D_KL(q(z|x) || p(z)) as regularization
  • Cross-entropy loss = constant + D_KL(true || predicted)

```

---

## 6. Code Implementation

```python
import numpy as np
import torch
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import entropy

# ============================================================
# Lp DISTANCES
# ============================================================

def lp_distance(x, y, p=2):
    """Compute Lp distance between vectors."""
    if p == np.inf:
        return np.max(np.abs(x - y))
    return np.sum(np.abs(x - y)**p)**(1/p)

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

print(f"L1 distance: {lp_distance(x, y, p=1)}")      # 9
print(f"L2 distance: {lp_distance(x, y, p=2):.4f}") # 5.196
print(f"L∞ distance: {lp_distance(x, y, p=np.inf)}") # 3

# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(x, y):
    """Compute cosine similarity."""
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def cosine_distance(x, y):
    """Compute cosine distance."""
    return 1 - cosine_similarity(x, y)

# Example: Text-like embeddings (direction matters, not magnitude)
doc1 = np.array([1, 1, 0, 0, 0])  # "cat", "dog"
doc2 = np.array([0.5, 0.5, 0, 0, 0])  # Same topics, different counts
doc3 = np.array([0, 0, 1, 1, 1])  # Different topics

print(f"Cosine(doc1, doc2): {cosine_similarity(doc1, doc2):.4f}")  # 1.0 (same direction!)
print(f"Cosine(doc1, doc3): {cosine_similarity(doc1, doc3):.4f}")  # 0.0 (orthogonal)

# ============================================================
# MAHALANOBIS DISTANCE
# ============================================================

def compute_mahalanobis(x, y, data):
    """Compute Mahalanobis distance using data covariance."""
    cov = np.cov(data.T)
    cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(len(cov)))  # Regularization
    diff = x - y
    return np.sqrt(diff @ cov_inv @ diff)

# Example: Correlated data
np.random.seed(42)
# Generate correlated 2D data
mean = [0, 0]
cov = [[1, 0.8], [0.8, 1]]
data = np.random.multivariate_normal(mean, cov, 100)

x = np.array([2, 2])
y = np.array([0, 0])

euclidean = np.linalg.norm(x - y)
mahal = compute_mahalanobis(x, y, data)
print(f"Euclidean: {euclidean:.4f}")
print(f"Mahalanobis: {mahal:.4f}")

# ============================================================
# PROBABILISTIC DISTANCES
# ============================================================

def kl_divergence(p, q, eps=1e-10):
    """KL divergence: D_KL(P || Q)."""
    p = p + eps
    q = q + eps
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    """Jensen-Shannon divergence (symmetric)."""
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

# Example: Compare distributions
p = np.array([0.9, 0.05, 0.05])  # Peaked
q = np.array([0.33, 0.33, 0.34])  # Uniform-ish

print(f"KL(P||Q): {kl_divergence(p, q):.4f}")
print(f"KL(Q||P): {kl_divergence(q, p):.4f}")  # Different!
print(f"JS(P, Q): {js_divergence(p, q):.4f}")  # Symmetric

# ============================================================
# PAIRWISE DISTANCES (EFFICIENT)
# ============================================================

def pairwise_l2_efficient(X, Y):
    """Efficient pairwise L2 using (x-y)² = x² + y² - 2xy."""
    XX = (X**2).sum(axis=1, keepdims=True)
    YY = (Y**2).sum(axis=1, keepdims=True)
    XY = X @ Y.T
    return np.sqrt(np.maximum(XX + YY.T - 2*XY, 0))

X = np.random.randn(1000, 128)
Y = np.random.randn(500, 128)

# This is O(n*m*d) but memory-efficient and vectorized
D = pairwise_l2_efficient(X, Y)
print(f"Pairwise distance matrix shape: {D.shape}")

# ============================================================
# PYTORCH IMPLEMENTATIONS
# ============================================================

def pytorch_distances():
    """Distance computations in PyTorch."""
    x = torch.randn(32, 128)  # Batch of 32 vectors, dim 128
    y = torch.randn(32, 128)
    
    # L2 distance
    l2_dist = torch.norm(x - y, dim=1)
    
    # Cosine similarity (built-in)
    cos_sim = torch.nn.functional.cosine_similarity(x, y, dim=1)
    
    # Pairwise distances for contrastive learning
    def pairwise_cosine(X):
        X_norm = X / X.norm(dim=1, keepdim=True)
        return X_norm @ X_norm.T
    
    similarity_matrix = pairwise_cosine(x)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")

pytorch_distances()

```

---

## 7. Choosing the Right Metric

### 📊 Decision Guide

| Scenario | Recommended Metric | Why |
|----------|-------------------|-----|
| General continuous data | L2 (Euclidean) | Most intuitive |
| Sparse high-dimensional | L1 or Cosine | More robust |
| Text/embeddings | Cosine | Scale invariant |
| Correlated features | Mahalanobis | Accounts for correlation |
| Binary features | Hamming or Jaccard | Counts differences |
| Probability distributions | KL or JS | Information-theoretic |
| Robust to outliers | L1 or Huber | Linear penalty |

### ⚠️ Curse of Dimensionality

```
In high dimensions (d >> 100):

Problem: All points become "equally far" apart!

max_distance / min_distance → 1 as d → ∞

Solutions:

1. Use cosine similarity (only cares about direction)

2. Reduce dimensionality first (PCA, t-SNE)

3. Learn the metric (Siamese networks, triplet loss)

4. Use approximate methods (LSH, FAISS)

```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📖 | [Scipy Distance Functions](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html) | Complete reference |
| 📄 | [Distance Measures Survey](https://www.sciencedirect.com/science/article/pii/S0306457309000259) | Academic overview |
| 🎥 | [StatQuest: Distance Metrics](https://www.youtube.com/watch?v=7pOSCaOoYNI) | Visual explanation |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Optimization](../03_optimization/README.md) | [Mathematics](../README.md) | [Back to Main](../README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
