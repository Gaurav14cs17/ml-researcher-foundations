# VC Dimension

> **Measuring the capacity of a hypothesis class**

---

## 🎯 Visual Overview

<img src="./images/vc-dimension.svg" width="100%">

*Caption: VC dimension measures how many points a hypothesis class can "shatter" (perfectly classify in all possible ways). Linear classifiers in 2D have VC=3 - they can shatter 3 non-collinear points but not 4.*

---

## 📐 Definition

```
VC dimension = Largest set that H can shatter

"Shatter" = Achieve all 2ⁿ possible labelings

VC(H) = max{n : H shatters some set of n points}
```

---

## 📊 Examples

| Hypothesis Class | VC Dimension |
|------------------|--------------|
| Linear classifiers in ℝᵈ | d + 1 |
| Axis-aligned rectangles in ℝ² | 4 |
| Intervals in ℝ | 2 |
| k-nearest neighbors | ∞ |
| Neural networks with W weights | O(W log W) |

---

## 🔑 Generalization Bound

```
With probability ≥ 1-δ:

R(h) ≤ R̂(h) + O(√(VC(H)log(n)/n + log(1/δ)/n))

More data (n↑) or lower capacity (VC↓) → better generalization
```

---

## 💻 Example: 2D Linear Classifier

```python
# VC dimension of linear classifiers in R²
# = 2 + 1 = 3

# Can shatter 3 points (not collinear):
#    ○            ●            ○
#   / \          / \          / \
#  ●   ●        ○   ○        ●   ○
#  All 8 labelings achievable!

# Cannot shatter 4 points (some labeling impossible):
#  ●   ○
#  ○   ●
#  XOR pattern - no line separates!
```

---

## 🌍 Modern Perspective

```
Classical: Low VC → good generalization
Reality: Neural nets have high VC but generalize well

Why? Other capacity measures:
• Rademacher complexity
• PAC-Bayes bounds
• Flatness of minima
• Implicit regularization
```

---

<- [Back](./README.md)


