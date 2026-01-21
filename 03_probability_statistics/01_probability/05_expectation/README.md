<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Expected%20Value%20%26%20Moments&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/expectation.svg" width="100%">

*Caption: E[X] is the weighted average of outcomes. Key properties: linearity (E[aX+b]=aE[X]+b), addition (E[X+Y]=E[X]+E[Y]). LOTUS allows computing E[g(X)] without knowing g(X)'s distribution.*

---

## 📂 Overview

Expected value is the most important summary statistic of a random variable. In ML, loss functions are expected values: $\mathcal{L} = E[\ell(\hat{y}, y)]$.

---

## 📐 Mathematical Definitions

### Discrete Random Variable

$$E[X] = \sum_{x} x \cdot P(X = x)$$

### Continuous Random Variable

$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

---

## 📐 Key Properties and Proofs

### Linearity of Expectation

**Theorem:**

$$E[aX + b] = a \cdot E[X] + b$$

**Proof (Discrete case):**

$$E[aX + b] = \sum_x (ax + b) \cdot P(X = x)
= a \sum_x x \cdot P(X = x) + b \sum_x P(X = x)
= a \cdot E[X] + b \cdot 1 = a \cdot E[X] + b \quad \blacksquare$$

---

### Addition Property

**Theorem:**

$$E[X + Y] = E[X] + E[Y] \quad \text{(ALWAYS, even if X, Y are dependent!)}$$

**Proof (Discrete case):**

$$E[X + Y] = \sum_x \sum_y (x + y) \cdot P(X = x, Y = y)
= \sum_x \sum_y x \cdot P(X = x, Y = y) + \sum_x \sum_y y \cdot P(X = x, Y = y)
= \sum_x x \sum_y P(X = x, Y = y) + \sum_y y \sum_x P(X = x, Y = y)
= \sum_x x \cdot P(X = x) + \sum_y y \cdot P(Y = y)
= E[X] + E[Y] \quad \blacksquare$$

---

### Multiplication Property

**Theorem:** If X and Y are **independent**:

$$E[XY] = E[X] \cdot E[Y]$$

**Proof:**

For independent random variables, $P(X=x, Y=y) = P(X=x) \cdot P(Y=y)$

$$E[XY] = \sum_x \sum_y xy \cdot P(X = x, Y = y)
= \sum_x \sum_y xy \cdot P(X = x) \cdot P(Y = y)
= \left(\sum_x x \cdot P(X = x)\right) \left(\sum_y y \cdot P(Y = y)\right)
= E[X] \cdot E[Y] \quad \blacksquare$$

---

### LOTUS (Law of the Unconscious Statistician)

**Theorem:** For any function g(X):

**Discrete:**

$$E[g(X)] = \sum_x g(x) \cdot P(X = x)$$

**Continuous:**

$$E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f(x) \, dx$$

**Why "unconscious"?** We compute E[g(X)] without needing to find the distribution of g(X)!

**Example:** Computing $E[X^2]$ without finding the distribution of $X^2$:

$$E[X^2] = \sum_x x^2 \cdot P(X = x)$$

---

## 📐 Variance

### Definition

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**Proof of alternative form:**

$$\text{Var}(X) = E[(X - \mu)^2]
= E[X^2 - 2\mu X + \mu^2]
= E[X^2] - 2\mu E[X] + \mu^2
= E[X^2] - 2\mu^2 + \mu^2
= E[X^2] - \mu^2 = E[X^2] - (E[X])^2 \quad \blacksquare$$

### Variance Properties

**Theorem:**

$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$

**Proof:**

$$\text{Var}(aX + b) = E[(aX + b - E[aX + b])^2]
= E[(aX + b - aE[X] - b)^2]
= E[(a(X - E[X]))^2]
= a^2 E[(X - E[X])^2] = a^2 \text{Var}(X) \quad \blacksquare$$

---

**Theorem (Variance of Sum):**

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$$

If X and Y are **independent**:

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**Proof:**

$$\text{Var}(X + Y) = E[(X + Y - E[X + Y])^2]
= E[((X - \mu_X) + (Y - \mu_Y))^2]
= E[(X - \mu_X)^2] + E[(Y - \mu_Y)^2] + 2E[(X - \mu_X)(Y - \mu_Y)]
= \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y) \quad \blacksquare$$

---

## 📐 Moments

### Definitions

| Moment Type | Formula | Description |
|-------------|---------|-------------|
| k-th raw moment | $E[X^k]$ | Moments about zero |
| k-th central moment | $E[(X - \mu)^k]$ | Moments about mean |
| 1st moment | $\mu = E[X]$ | Mean |
| 2nd central moment | $\sigma^2 = E[(X-\mu)^2]$ | Variance |
| 3rd central moment | $E[(X-\mu)^3]$ | Related to skewness |
| 4th central moment | $E[(X-\mu)^4]$ | Related to kurtosis |

### Standardized Moments

**Skewness (Asymmetry):**

$$\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}$$

- $\gamma_1 = 0$: Symmetric distribution

- $\gamma_1 > 0$: Right tail longer (positive skew)

- $\gamma_1 < 0$: Left tail longer (negative skew)

**Kurtosis (Tail Weight):**

$$\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4}$$

**Excess Kurtosis:** $\gamma_2 - 3$ (Gaussian has excess kurtosis = 0)

- $\gamma_2 - 3 = 0$: Normal-like tails

- $\gamma_2 - 3 > 0$: Heavier tails than normal

- $\gamma_2 - 3 < 0$: Lighter tails than normal

---

## 📐 Moment Generating Functions

**Definition:**

$$M_X(t) = E[e^{tX}]$$

**Why useful?**

$$\frac{d^n M_X}{dt^n}\bigg|_{t=0} = E[X^n]$$

**Proof:**

$$M_X(t) = E[e^{tX}] = E\left[\sum_{n=0}^{\infty} \frac{(tX)^n}{n!}\right] = \sum_{n=0}^{\infty} \frac{t^n E[X^n]}{n!}$$

Taking the n-th derivative and evaluating at t=0:

$$\frac{d^n M_X}{dt^n}\bigg|_{t=0} = E[X^n] \quad \blacksquare$$

**Example: Gaussian MGF**

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$M_X(t) = \exp\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

---

## 📐 Covariance

### Definition

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

**Proof of alternative form:**

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]
= E[XY - X\mu_Y - Y\mu_X + \mu_X\mu_Y]
= E[XY] - \mu_Y E[X] - \mu_X E[Y] + \mu_X\mu_Y
= E[XY] - \mu_X\mu_Y = E[XY] - E[X]E[Y] \quad \blacksquare$$

### Correlation

$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}$$

**Properties:**
- $-1 \leq \rho \leq 1$

- $|\rho| = 1$ ⟺ Perfect linear relationship

- $\rho = 0$ ⟺ No linear relationship (uncorrelated)

---

## 📐 Jensen's Inequality

**Theorem:** For a convex function $g$:

$$g(E[X]) \leq E[g(X)]$$

For a concave function $g$:

$$g(E[X]) \geq E[g(X)]$$

**Applications in ML:**
- **Log-likelihood lower bound:** $E[\log p] \leq \log E[p]$ (log is concave)

- **ELBO in VAE:** Uses Jensen's inequality

- **KL divergence non-negativity:** Uses convexity of $-\log$

---

## 💻 Code Examples

```python
import numpy as np
import torch
from scipy import stats

# Empirical expectation (Monte Carlo estimation)
samples = np.random.normal(0, 1, 10000)
E_X = samples.mean()       # ≈ 0
E_X2 = (samples**2).mean() # ≈ 1 (variance for standard normal)
Var_X = E_X2 - E_X**2      # Alternative variance formula

print(f"E[X] ≈ {E_X:.4f}")
print(f"E[X²] ≈ {E_X2:.4f}")
print(f"Var(X) ≈ {Var_X:.4f}")

# Moments using scipy
data = np.random.randn(10000)
mean = np.mean(data)           # 1st moment
variance = np.var(data)         # 2nd central moment
skewness = stats.skew(data)     # 3rd standardized
kurtosis = stats.kurtosis(data) # Excess kurtosis

print(f"Skewness: {skewness:.4f}")  # ≈ 0 for Gaussian
print(f"Excess Kurtosis: {kurtosis:.4f}")  # ≈ 0 for Gaussian

# Loss function is an expectation
def cross_entropy_loss(logits, targets):
    """E[-log p(y|x)] over data distribution"""
    log_probs = torch.log_softmax(logits, dim=-1)
    return -log_probs[range(len(targets)), targets].mean()

# Monte Carlo estimation of expectation
def monte_carlo_expectation(f, distribution, n_samples=10000):
    """Estimate E[f(X)] via Monte Carlo"""
    samples = distribution.sample((n_samples,))
    return f(samples).mean()

# Example: E[X^2] for X ~ N(0, 1)
normal = torch.distributions.Normal(0, 1)
E_X2_mc = monte_carlo_expectation(lambda x: x**2, normal)
print(f"E[X²] via Monte Carlo: {E_X2_mc:.4f}")  # ≈ 1

# Reparameterization trick (VAE) - preserves gradients through sampling
def sample_gaussian(mu, log_var):
    """
    Sample z ~ N(mu, sigma²) while allowing gradient flow
    E_z[f(z)] can be estimated and differentiated
    """
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)  # Sample from N(0,1)
    return mu + eps * std  # z ~ N(mu, sigma²)

# Covariance matrix estimation
X = np.random.randn(1000, 3)  # 1000 samples, 3 features
cov_matrix = np.cov(X.T)      # 3x3 covariance matrix
corr_matrix = np.corrcoef(X.T) # Correlation matrix

# Moments in Adam optimizer
class AdamMoments:
    """Adam uses 1st and 2nd moment estimates of gradients"""
    def __init__(self, beta1=0.9, beta2=0.999):
        self.m = None  # 1st moment (mean)
        self.v = None  # 2nd moment (uncentered variance)
        self.beta1, self.beta2 = beta1, beta2
        self.t = 0
    
    def update(self, grad):
        self.t += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)
        
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        
        # Bias correction
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return m_hat, v_hat

```

---

## 🌍 ML Applications

| Application | Usage |
|-------------|-------|
| **Loss Functions** | $\mathcal{L} = E[\ell(f(x), y)]$ |
| **Batch Normalization** | Uses running mean and variance |
| **Adam Optimizer** | 1st and 2nd moment estimates of gradients |
| **Dropout** | $E[\text{dropout mask}] = 1 - p$ |
| **Monte Carlo Methods** | Estimate expectations via sampling |
| **Policy Gradient (RL)** | $\nabla J = E[\nabla \log \pi \cdot R]$ |
| **VAE** | ELBO uses expectations |
| **MMD** | Maximum Mean Discrepancy matches moments |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML Ch. 1 | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🇨🇳 | 期望与方差 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率论基础 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 概率论课程 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Distributions](../04_distributions/) | ➡️ [Next: Limit Theorems](../06_limit_theorems/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
