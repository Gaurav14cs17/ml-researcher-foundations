# 🎯 VC Dimension

> **Measuring the capacity and expressiveness of hypothesis classes**

<img src="./images/vc-dimension.svg" width="100%">

---

## 🎯 Core Concept

The **Vapnik-Chervonenkis (VC) dimension** quantifies the capacity of a hypothesis class - how complex patterns it can represent. It's the largest number of points that can be "shattered" (perfectly classified in all possible ways).

---

## 📐 Definition

```
VC(H) = max{n : H can shatter some set of n points}

"Shatter" = achieve all 2ⁿ possible binary labelings

If no finite n exists, VC(H) = ∞
```

---

## 📊 Common VC Dimensions

| Hypothesis Class | VC Dimension | Intuition |
|------------------|--------------|-----------|
| Linear classifiers in ℝᵈ | d + 1 | d weights + bias |
| Axis-aligned rectangles (ℝ²) | 4 | 4 corners |
| Intervals on ℝ | 2 | 2 endpoints |
| k-nearest neighbors | ∞ | Can memorize |
| Neural nets (W weights) | O(W log W) | Upper bound |

---

## 🔑 Key Results

### Generalization Bound

With probability ≥ 1-δ over training sets of size n:

```
R(h) ≤ R̂(h) + O(√(VC(H)log(n)/n + log(1/δ)/n))

Where:
  R(h)  = true risk (test error)
  R̂(h) = empirical risk (train error)
```

**Implications:**
- ↑ More data (n↑) → tighter bound
- ↓ Lower capacity (VC↓) → better generalization
- Trade-off between fit and complexity

---

## 💻 Example: Linear Classifiers in 2D

```python
import numpy as np
import matplotlib.pyplot as plt

# VC dimension = 3 for 2D linear classifiers

# Can shatter 3 points (8 labelings possible)
points_3 = np.array([[0, 0], [1, 0], [0.5, 1]])

# ALL 8 labelings achievable with a line:
# [+,+,+], [+,+,-], [+,-,+], [-,+,+]
# [+,-,-], [-,+,-], [-,-,+], [-,-,-]

# Cannot shatter 4 points in XOR configuration:
points_4 = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
labels_xor = np.array([1, -1, -1, 1])

# No line can separate XOR pattern!
# This proves VC dimension ≤ 3
```

---

## 🌍 Modern Perspective

### Classical View
```
Low VC dimension → Good generalization
High VC dimension → Overfitting
```

### Reality with Deep Learning
```
Neural networks have very high (even infinite) VC dimension
Yet they generalize well!

Why? Other factors matter:
• SGD implicit regularization
• Flatness of minima (sharpness)
• Rademacher complexity
• PAC-Bayes bounds
• Compression via pruning
```

---

## 📖 Detailed Content

[→ VC Theory Deep Dive](./vc-theory.md)

---

## 📚 Resources

### Papers
- **Vapnik & Chervonenkis (1971)** - Original paper
- **Understanding Deep Learning Requires Rethinking Generalization** (Zhang et al., 2017)
- **Reconciling modern ML and VC theory** (Nagarajan & Kolter, 2019)

### Books
- **Understanding Machine Learning** - Shalev-Shwartz & Ben-David (Ch 6)
- **Statistical Learning Theory** - Vapnik (1998)

---

⬅️ [Back: Generalization](../)

---

⬅️ [Back: Regularization](../regularization/)
