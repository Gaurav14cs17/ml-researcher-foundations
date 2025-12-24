# Moments

> **Summarizing distributions with numbers**

---

## 📐 Definitions

```
k-th raw moment:    E[Xᵏ]
k-th central moment: E[(X - μ)ᵏ]

Key moments:
• 1st: Mean (μ = E[X])
• 2nd central: Variance (σ² = E[(X-μ)²])
• 3rd central: Skewness (asymmetry)
• 4th central: Kurtosis (tail weight)
```

---

## 📊 Standardized Moments

| Moment | Formula | Meaning |
|--------|---------|---------|
| Skewness | E[(X-μ)³]/σ³ | Left/right asymmetry |
| Kurtosis | E[(X-μ)⁴]/σ⁴ | Heavy tails |

```
Skewness:
• = 0: Symmetric
• > 0: Right tail longer
• < 0: Left tail longer

Kurtosis (excess):
• = 0: Normal-like tails
• > 0: Heavier tails
• < 0: Lighter tails
```

---

## 💻 Code

```python
import numpy as np
from scipy import stats

data = np.random.randn(10000)

# Moments
mean = np.mean(data)           # 1st moment
variance = np.var(data)         # 2nd central moment
skewness = stats.skew(data)     # 3rd standardized
kurtosis = stats.kurtosis(data) # 4th standardized (excess)

# Raw moments
second_raw = np.mean(data**2)
# Note: Var = E[X²] - E[X]² = second_raw - mean²
```

---

## 🌍 In ML

| Application | Moments Used |
|-------------|--------------|
| Batch Norm | Mean, Variance |
| Adam | 1st, 2nd moment estimates |
| Distribution matching | All moments (MMD) |

---

<- [Back](./README.md)


