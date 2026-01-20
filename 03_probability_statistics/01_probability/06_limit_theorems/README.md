<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Limit%20Theorems&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/limit-theorems.svg" width="100%">

*Caption: LLN: sample mean → population mean. CLT: sum of any i.i.d. RVs → Gaussian. These explain why sampling works and why Gaussians appear everywhere.*

---

## 📂 Overview

Limit theorems explain why statistics and machine learning work. They guarantee that averaging many samples gives accurate estimates, and that sums become approximately Gaussian.

---

## 📐 Law of Large Numbers (LLN)

### Weak Law of Large Numbers

**Theorem:** For i.i.d. random variables $X\_1, X\_2, \ldots, X\_n$ with $E[X\_i] = \mu$ and finite variance:

```math
\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i \xrightarrow{p} \mu \quad \text{as } n \to \infty
```

**Meaning:** The sample mean converges in probability to the true mean.

### Strong Law of Large Numbers

**Theorem:** Under the same conditions:

```math
P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1
```

**Meaning:** Almost sure convergence - stronger than convergence in probability.

### Proof of Weak LLN (using Chebyshev)

```math
P(|\bar{X}_n - \mu| > \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n\epsilon^2} \to 0
```

as $n \to \infty$. $\quad \blacksquare$

---

## 📐 Central Limit Theorem (CLT)

### Statement

**Theorem:** For i.i.d. random variables $X\_1, \ldots, X\_n$ with $E[X\_i] = \mu$ and $\text{Var}(X\_i) = \sigma^2 < \infty$:

```math
\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty
```

**Equivalently:**
```math
\bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
```

**Key Insight:** The sum of ANY i.i.d. random variables → Gaussian!

### Proof Sketch (via Characteristic Functions)

**Step 1:** Standardize variables
```math
Z_i = \frac{X_i - \mu}{\sigma}
```

So $E[Z\_i] = 0$ and $\text{Var}(Z\_i) = 1$.

**Step 2:** Define standardized sum
```math
S_n = \frac{1}{\sqrt{n}}\sum_{i=1}^{n} Z_i = \frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma}
```

**Step 3:** Compute characteristic function
```math
\phi_{S_n}(t) = E[e^{itS_n}] = \left[\phi_Z\left(\frac{t}{\sqrt{n}}\right)\right]^n
```

**Step 4:** Taylor expansion of $\phi\_Z$
```math
\phi_Z(s) = 1 + is \cdot E[Z] - \frac{s^2}{2}E[Z^2] + O(s^3)
= 1 - \frac{s^2}{2} + O(s^3)
```

**Step 5:** Substitute $s = t/\sqrt{n}$
```math
\phi_Z\left(\frac{t}{\sqrt{n}}\right) = 1 - \frac{t^2}{2n} + O\left(\frac{1}{n^{3/2}}\right)
```

**Step 6:** Take limit
```math
\phi_{S_n}(t) = \left[1 - \frac{t^2}{2n}\right]^n \to e^{-t^2/2}
```

This is the characteristic function of $\mathcal{N}(0, 1)$! $\quad \blacksquare$

---

## 📐 Berry-Esseen Theorem

**Rate of convergence:**

```math
\sup_z \left|P\left(\frac{\sqrt{n}(\bar{X}_n - \mu)}{\sigma} \leq z\right) - \Phi(z)\right| \leq \frac{C \cdot E[|X - \mu|^3]}{\sigma^3 \sqrt{n}}
```

where $C \leq 0.4748$ and $\Phi$ is the standard normal CDF.

**Key insight:** Convergence rate is $O(1/\sqrt{n})$.

---

## 📐 Concentration Inequalities

### Markov's Inequality

For non-negative random variable $X$ and $a > 0$:

```math
P(X \geq a) \leq \frac{E[X]}{a}
```

**Proof:**
```math
E[X] = E[X \cdot \mathbf{1}_{X \geq a}] + E[X \cdot \mathbf{1}_{X < a}] \geq a \cdot P(X \geq a) \quad \blacksquare
```

### Chebyshev's Inequality

For any random variable $X$ with finite variance:

```math
P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}
```

**Proof:** Apply Markov to $(X - \mu)^2$ with $a = k^2\sigma^2$. $\quad \blacksquare$

### Hoeffding's Inequality

For i.i.d. bounded random variables $X\_i \in [a\_i, b\_i]$:

```math
P\left(\bar{X}_n - \mu \geq t\right) \leq \exp\left(-\frac{2n^2 t^2}{\sum_i (b_i - a_i)^2}\right)
```

For $X\_i \in [a, b]$:

```math
P\left(|\bar{X}_n - \mu| \geq t\right) \leq 2\exp\left(-\frac{2nt^2}{(b-a)^2}\right)
```

**Application:** Guarantees for sample mean estimation.

### Chernoff Bound

For $X = \sum\_{i=1}^n X\_i$ where $X\_i \in \{0, 1\}$ are independent with $E[X\_i] = p\_i$:

```math
P(X \geq (1+\delta)\mu) \leq e^{-\frac{\delta^2 \mu}{2+\delta}}
P(X \leq (1-\delta)\mu) \leq e^{-\frac{\delta^2 \mu}{2}}
```

where $\mu = E[X] = \sum\_i p\_i$.

---

## 📐 Why These Matter for ML

### 1. SGD Convergence

Mini-batch gradient is an unbiased estimator of full gradient:

```math
\hat{\nabla} f = \frac{1}{m}\sum_{i=1}^{m} \nabla f_i(\theta)
```

By LLN: $\hat{\nabla} f \xrightarrow{p} \nabla f$ as $m \to \infty$

By CLT: $\hat{\nabla} f \approx \mathcal{N}(\nabla f, \frac{\sigma^2}{m})$

### 2. Generalization Bounds

Test accuracy converges to true accuracy:

```math
\text{Acc}_{\text{test}} \xrightarrow{p} \text{Acc}_{\text{true}}
```

Hoeffding gives confidence intervals for performance estimates.

### 3. Monte Carlo Methods

For any function $g$:

```math
E[g(X)] = \int g(x) p(x) dx \approx \frac{1}{n}\sum_{i=1}^{n} g(x_i)
```

Error rate: $O(1/\sqrt{n})$ regardless of dimension!

### 4. Why Gaussians Appear Everywhere

- **Sum of many small effects:** CLT applies
- **Gradient noise in SGD:** Sum of per-sample gradients
- **Measurement errors:** Sum of many error sources
- **Neural network activations:** Sum of many weighted inputs

---

## 💻 Code Examples

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrate LLN
np.random.seed(42)
true_mean = 5.0
n_samples = 10000

# Sample from exponential (non-Gaussian!)
samples = np.random.exponential(scale=true_mean, size=n_samples)

# Compute running average
running_avg = np.cumsum(samples) / np.arange(1, n_samples + 1)

print(f"After {n_samples} samples: mean = {running_avg[-1]:.3f} (true = {true_mean})")

# Demonstrate CLT
def demonstrate_clt(distribution, n_samples_per_mean=30, n_means=1000):
    """
    Show that sample means become Gaussian regardless of original distribution
    """
    sample_means = []
    for _ in range(n_means):
        samples = distribution(size=n_samples_per_mean)
        sample_means.append(np.mean(samples))
    
    # Test for normality
    _, p_value = stats.normaltest(sample_means)
    return np.array(sample_means), p_value

# Try with different distributions
distributions = {
    'Uniform': lambda size: np.random.uniform(0, 1, size),
    'Exponential': lambda size: np.random.exponential(1, size),
    'Bernoulli': lambda size: np.random.binomial(1, 0.3, size),
}

for name, dist in distributions.items():
    means, p = demonstrate_clt(dist)
    print(f"{name}: normality test p-value = {p:.4f} (>0.05 = normal)")

# Confidence interval using CLT
def confidence_interval(data, confidence=0.95):
    """
    Compute confidence interval for mean using CLT
    """
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)  # Standard error = σ/√n
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, (mean - h, mean + h)

# Monte Carlo integration
def monte_carlo_integral(f, a, b, n_samples=10000):
    """
    Estimate ∫_a^b f(x) dx using Monte Carlo
    By LLN: (b-a) * E[f(X)] where X ~ Uniform(a,b)
    """
    x = np.random.uniform(a, b, n_samples)
    return (b - a) * np.mean(f(x))

# Example: ∫_0^1 x^2 dx = 1/3
estimate = monte_carlo_integral(lambda x: x**2, 0, 1)
print(f"Monte Carlo estimate of ∫x² dx: {estimate:.4f} (true = 0.3333)")

# Hoeffding bound
def hoeffding_bound(epsilon, n, a=0, b=1):
    """
    P(|X̄ - μ| ≥ ε) ≤ 2exp(-2nε²/(b-a)²)
    """
    return 2 * np.exp(-2 * n * epsilon**2 / (b - a)**2)

# How many samples for 95% confidence within ±0.01?
epsilon = 0.01
confidence = 0.95
n = int(np.log(2 / (1 - confidence)) / (2 * epsilon**2))
print(f"Need n={n} samples for 95% confidence within ±0.01")
```

---

## 📊 Summary of Key Results

| Theorem | Statement | Use in ML |
|---------|-----------|-----------|
| **Weak LLN** | $\bar{X}\_n \xrightarrow{p} \mu$ | Sample mean → true mean |
| **Strong LLN** | $\bar{X}\_n \xrightarrow{a.s.} \mu$ | Almost sure convergence |
| **CLT** | $\sqrt{n}(\bar{X}\_n - \mu)/\sigma \xrightarrow{d} \mathcal{N}(0,1)$ | Why Gaussians everywhere |
| **Hoeffding** | $P(\|\bar{X}\_n - \mu\| \geq t) \leq 2e^{-2nt^2}$ | Sample complexity bounds |
| **Chebyshev** | $P(\|X - \mu\| \geq k\sigma) \leq 1/k^2$ | Basic concentration |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | Wasserman: All of Statistics | [Book](https://www.stat.cmu.edu/~larry/all-of-statistics/) |
| 🇨🇳 | 大数定律与中心极限 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率收敛 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 概率论课程 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Expectation](../05_expectation/) | ➡️ [Next: Random Variables](../07_random_variables/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
