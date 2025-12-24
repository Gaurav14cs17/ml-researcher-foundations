# Vector Spaces

> **The mathematical foundation of linear algebra**

---

## 🎯 Visual Overview

<img src="./images/vector-space.svg" width="100%">

*Caption: A vector space is a set of vectors that can be added together and scaled by numbers (scalars). The key axioms ensure these operations behave intuitively. ℝⁿ is the most common example in ML.*

---

## 📂 Overview

Vector spaces provide the mathematical framework for linear algebra. They define what it means to add vectors and multiply them by scalars.

---

## 🔑 Key Concepts

| Concept | Definition |
|---------|------------|
| **Vector Space** | Set with addition (+) and scalar multiplication (·) |
| **Span** | All possible linear combinations of vectors |
| **Basis** | Minimal set of vectors that span the space |
| **Dimension** | Number of vectors in any basis |
| **Subspace** | Subset that is itself a vector space |

---

## 📐 Important Properties

```
Linear Independence:
α₁v₁ + α₂v₂ + ... + αₙvₙ = 0  →  all αᵢ = 0

Dimension theorem:
dim(V) = dim(range(T)) + dim(null(T))

Change of basis:
[v]_B' = P⁻¹[v]_B  where P is change-of-basis matrix
```

---

## 🌍 ML Applications

| Application | How Vector Spaces Are Used |
|-------------|---------------------------|
| **Embeddings** | Words/images as vectors in ℝⁿ |
| **PCA** | Find subspace of maximum variance |
| **Attention** | Query, Key, Value in different spaces |
| **Linear Layers** | y = Wx + b (linear transformation) |

---

## 💻 Code

```python
import numpy as np

# Check linear independence
def is_independent(vectors):
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)

# Find basis via SVD
U, S, Vt = np.linalg.svd(A)
rank = np.sum(S > 1e-10)
basis = U[:, :rank]  # Orthonormal basis for column space
```


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Linear Algebra](../)

---

⬅️ [Back: Transformations](../transformations/) | ➡️ [Next: Vectors Matrices](../vectors-matrices/)
