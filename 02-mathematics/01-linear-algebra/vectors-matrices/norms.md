# Norms

> **Measuring size of vectors and matrices**

---

## 📐 Vector Norms

| Norm | Formula | Use |
|------|---------|-----|
| L1 | \|\|x\|\|₁ = Σ\|xᵢ\| | Sparsity (Lasso) |
| L2 | \|\|x\|\|₂ = √(Σxᵢ²) | Euclidean distance |
| L∞ | \|\|x\|\|∞ = max\|xᵢ\| | Max element |
| Lp | \|\|x\|\|ₚ = (Σ\|xᵢ\|ᵖ)^(1/p) | General |

---

## 📐 Matrix Norms

| Norm | Formula | Use |
|------|---------|-----|
| Frobenius | \|\|A\|\|_F = √(ΣΣaᵢⱼ²) | Matrix "size" |
| Spectral | \|\|A\|\|₂ = σ_max(A) | Largest singular value |
| Nuclear | \|\|A\|\|_* = Σσᵢ | Low-rank (matrix completion) |

---

## 💻 Code

```python
import numpy as np

x = np.array([3, 4])

np.linalg.norm(x, ord=1)    # L1: 7
np.linalg.norm(x, ord=2)    # L2: 5
np.linalg.norm(x, ord=np.inf)  # L∞: 4

A = np.random.randn(3, 4)
np.linalg.norm(A, 'fro')    # Frobenius
np.linalg.norm(A, 2)        # Spectral
np.linalg.norm(A, 'nuc')    # Nuclear
```

---

<- [Back](./README.md)


