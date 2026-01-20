<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Learning Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../02_generalization/">Next: Generalization ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=PAC%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/pac-vc-dimension-complete.svg" width="100%">

---

## 📂 Overview

**PAC (Probably Approximately Correct) Learning** is a fundamental framework in computational learning theory, introduced by Leslie Valiant in 1984. It provides a rigorous mathematical definition of what it means for a learning algorithm to be successful.

---

## 📐 Formal Definition

### PAC Learnability

**Definition:** A concept class \(\mathcal{C}\) is **PAC-learnable** if there exists an algorithm \(\mathcal{A}\) and a polynomial \(p(\cdot, \cdot, \cdot, \cdot)\) such that:

For all:
- Target concept \(c \in \mathcal{C}\)
- Distribution \(\mathcal{D}\) over input space \(\mathcal{X}\)
- Accuracy parameter \(\epsilon > 0\)
- Confidence parameter \(\delta > 0\)

When given \(m \geq p(1/\epsilon, 1/\delta, n, \text{size}(c))\) i.i.d. samples from \(\mathcal{D}\):

```math
\Pr_{S \sim \mathcal{D}^m}\left[\text{error}_{\mathcal{D}}(h_S) \leq \epsilon\right] \geq 1 - \delta

```

where:
- \(h_S = \mathcal{A}(S)\) is the hypothesis output by algorithm
- \(\text{error}_{\mathcal{D}}(h) = \Pr_{x \sim \mathcal{D}}[h(x) \neq c(x)]\)

**In plain English:** "With high probability (\(1-\delta\)), the learned hypothesis is approximately correct (error \(\leq \epsilon\))."

---

## 📐 Sample Complexity Bounds

### Finite Hypothesis Class

**Theorem:** For a finite hypothesis class \(|\mathcal{H}| < \infty\), the sample complexity is:

```math
m \geq \frac{1}{\epsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)

```

**Proof:**

Let \(h\) be a "bad" hypothesis with \(\text{error}_{\mathcal{D}}(h) > \epsilon\).

For any single training example \((x, y)\):

```math
\Pr[(x, y) \text{ consistent with } h] = 1 - \text{error}_{\mathcal{D}}(h) < 1 - \epsilon

```

For \(m\) i.i.d. examples, probability \(h\) is consistent with all:

```math
\Pr[h \text{ consistent with } S] < (1 - \epsilon)^m \leq e^{-\epsilon m}

```

By union bound over all bad hypotheses:

```math
\Pr[\exists \text{ bad } h \text{ consistent with } S] \leq |\mathcal{H}| \cdot e^{-\epsilon m}

```

Setting this \(\leq \delta\):

```math
|\mathcal{H}| \cdot e^{-\epsilon m} \leq \delta
m \geq \frac{1}{\epsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right) \quad \blacksquare

```

### VC Dimension Bound

**Theorem (Fundamental Theorem of PAC Learning):** For hypothesis class \(\mathcal{H}\) with VC dimension \(d < \infty\):

```math
m = O\left(\frac{d + \ln(1/\delta)}{\epsilon^2}\right)

```

More precisely:

```math
m \geq \frac{c}{\epsilon^2}\left(d \ln\frac{1}{\epsilon} + \ln\frac{1}{\delta}\right)

```

for some constant \(c\).

---

## 📐 Realizable vs Agnostic PAC Learning

### Realizable Case

**Assumption:** There exists \(c^* \in \mathcal{H}\) with zero error.

```math
\min_{h \in \mathcal{H}} \text{error}_{\mathcal{D}}(h) = 0

```

**Goal:** Find \(h\) with \(\text{error}_{\mathcal{D}}(h) \leq \epsilon\).

### Agnostic Case (More Realistic)

**No assumption** about whether true concept is in \(\mathcal{H}\).

```math
\text{OPT} = \min_{h \in \mathcal{H}} \text{error}_{\mathcal{D}}(h) \geq 0

```

**Goal:** Find \(h\) with:

```math
\text{error}_{\mathcal{D}}(h) \leq \text{OPT} + \epsilon

```

**Theorem (Agnostic PAC):** For \(\mathcal{H}\) with VC dimension \(d\):

```math
m = O\left(\frac{d + \ln(1/\delta)}{\epsilon^2}\right)

```

guarantees with probability \(\geq 1 - \delta\):

```math
\text{error}_{\mathcal{D}}(h_S) \leq \min_{h \in \mathcal{H}} \text{error}_{\mathcal{D}}(h) + \epsilon

```

---

## 📐 Key Theorems and Proofs

### Hoeffding's Inequality

**Theorem:** Let \(X_1, \ldots, X_m\) be i.i.d. bounded random variables with \(X_i \in [a, b]\). Then:

```math
\Pr\left[\left|\frac{1}{m}\sum_{i=1}^m X_i - \mathbb{E}[X_1]\right| > \epsilon\right] \leq 2\exp\left(-\frac{2m\epsilon^2}{(b-a)^2}\right)

```

### Generalization Bound (Single Hypothesis)

For fixed hypothesis \(h\), with probability \(\geq 1 - \delta\):

```math
\left|\hat{\text{error}}_S(h) - \text{error}_{\mathcal{D}}(h)\right| \leq \sqrt{\frac{\ln(2/\delta)}{2m}}

```

where \(\hat{\text{error}}_S(h) = \frac{1}{m}\sum_{i=1}^m \mathbb{1}[h(x_i) \neq y_i]\).

**Proof:** Apply Hoeffding with \(X_i = \mathbb{1}[h(x_i) \neq y_i]\), so \(X_i \in [0, 1]\). \(\blacksquare\)

### Uniform Convergence

**Theorem:** For finite \(\mathcal{H}\), with probability \(\geq 1 - \delta\):

```math
\forall h \in \mathcal{H}: \left|\hat{\text{error}}_S(h) - \text{error}_{\mathcal{D}}(h)\right| \leq \sqrt{\frac{\ln(2|\mathcal{H}|/\delta)}{2m}}

```

**Proof:** Apply union bound over all \(h \in \mathcal{H}\):

```math
\Pr[\exists h: |\hat{\text{error}} - \text{error}| > \epsilon] \leq \sum_{h \in \mathcal{H}} \Pr[|\hat{\text{error}}_h - \text{error}_h| > \epsilon] \leq |\mathcal{H}| \cdot 2e^{-2m\epsilon^2}

```

Setting this to \(\delta\) and solving for \(\epsilon\). \(\blacksquare\)

---

## 💻 Code Implementation

```python
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def sample_complexity_finite(H_size, epsilon, delta):
    """
    PAC sample complexity for finite hypothesis class.
    
    m ≥ (1/ε)(ln|H| + ln(1/δ))
    
    Args:
        H_size: Size of hypothesis class |H|
        epsilon: Accuracy parameter (error tolerance)
        delta: Confidence parameter (failure probability)
        
    Returns:
        int: Required number of samples
    """
    return int(np.ceil((1/epsilon) * (np.log(H_size) + np.log(1/delta))))

def sample_complexity_vc(vc_dim, epsilon, delta):
    """
    PAC sample complexity using VC dimension.
    
    m = O((d/ε²)(ln(1/ε) + ln(1/δ)))
    
    Using tighter bound:
    m ≥ (8/ε²)(d·ln(2e·m/d) + ln(4/δ))
    
    Args:
        vc_dim: VC dimension of hypothesis class
        epsilon: Accuracy parameter
        delta: Confidence parameter
        
    Returns:
        int: Required number of samples
    """
    # Approximate formula (not solving implicit equation)
    return int(np.ceil(
        (8/epsilon**2) * (vc_dim * np.log(2/epsilon) + np.log(4/delta))
    ))

def generalization_bound(m, H_size=None, vc_dim=None, delta=0.05):
    """
    Compute generalization bound given sample size.
    
    With probability ≥ 1-δ:
    |train_error - true_error| ≤ bound
    
    Args:
        m: Number of samples
        H_size: Size of finite hypothesis class (optional)
        vc_dim: VC dimension (optional)
        delta: Confidence parameter
        
    Returns:
        float: Generalization bound
    """
    if H_size is not None:
        # Finite hypothesis class
        return np.sqrt(np.log(2 * H_size / delta) / (2 * m))
    elif vc_dim is not None:
        # VC dimension bound
        return np.sqrt((8 / m) * (vc_dim * np.log(2 * np.e * m / vc_dim) + np.log(4 / delta)))
    else:
        raise ValueError("Provide either H_size or vc_dim")

def plot_sample_complexity():
    """Visualize how sample complexity scales with parameters."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Sample complexity vs epsilon
    epsilons = np.linspace(0.01, 0.5, 100)
    m_finite = [sample_complexity_finite(1000, e, 0.05) for e in epsilons]
    m_vc = [sample_complexity_vc(10, e, 0.05) for e in epsilons]
    
    axes[0].plot(epsilons, m_finite, label='Finite H (|H|=1000)')
    axes[0].plot(epsilons, m_vc, label='VC bound (d=10)')
    axes[0].set_xlabel('ε (error tolerance)')
    axes[0].set_ylabel('Sample Complexity m')
    axes[0].set_title('Sample Complexity vs Accuracy')
    axes[0].legend()
    axes[0].set_yscale('log')
    
    # 2. Sample complexity vs VC dimension
    vc_dims = np.arange(1, 101)
    m_by_vc = [sample_complexity_vc(d, 0.1, 0.05) for d in vc_dims]
    
    axes[1].plot(vc_dims, m_by_vc)
    axes[1].set_xlabel('VC Dimension d')
    axes[1].set_ylabel('Sample Complexity m')
    axes[1].set_title('Sample Complexity vs VC Dimension')
    
    # 3. Generalization bound vs sample size
    samples = np.arange(100, 10001, 100)
    bounds = [generalization_bound(m, vc_dim=10) for m in samples]
    
    axes[2].plot(samples, bounds)
    axes[2].set_xlabel('Sample Size m')
    axes[2].set_ylabel('Generalization Bound')
    axes[2].set_title('Generalization Bound vs Sample Size')
    
    plt.tight_layout()
    plt.savefig('pac_learning_analysis.png', dpi=150)
    plt.show()

class PACLearner:
    """
    Demonstrates PAC learning with empirical risk minimization.
    """
    
    def __init__(self, hypothesis_class):
        """
        Args:
            hypothesis_class: List of hypothesis functions
        """
        self.H = hypothesis_class
        
    def fit(self, X, y):
        """
        ERM: Find hypothesis with minimum training error.
        
        h* = argmin_h (1/m) Σ I[h(x_i) ≠ y_i]
        """
        best_h = None
        best_error = float('inf')
        
        for h in self.H:
            predictions = np.array([h(x) for x in X])
            error = np.mean(predictions != y)
            
            if error < best_error:
                best_error = error
                best_h = h
        
        self.best_hypothesis = best_h
        self.train_error = best_error
        return self
    
    def predict(self, X):
        return np.array([self.best_hypothesis(x) for x in X])
    
    def pac_bound(self, m, delta=0.05):
        """
        Compute PAC generalization bound.
        
        With prob ≥ 1-δ: true_error ≤ train_error + bound
        """
        return np.sqrt(np.log(2 * len(self.H) / delta) / (2 * m))

# Example: Learning axis-aligned rectangles
def create_rectangle_hypothesis(x1_min, x1_max, x2_min, x2_max):
    """Create hypothesis for axis-aligned rectangle in 2D."""
    def h(x):
        return 1 if (x1_min <= x[0] <= x1_max and x2_min <= x[1] <= x2_max) else 0
    return h

if __name__ == "__main__":
    # Sample complexity examples
    print("=== PAC Learning Sample Complexity ===\n")
    
    # Finite hypothesis class
    H_size = 10000
    epsilon = 0.05
    delta = 0.01
    m = sample_complexity_finite(H_size, epsilon, delta)
    print(f"Finite H (|H|={H_size}), ε={epsilon}, δ={delta}")
    print(f"  Required samples: {m}")
    
    # VC dimension
    vc_dim = 10
    m = sample_complexity_vc(vc_dim, epsilon, delta)
    print(f"\nVC bound (d={vc_dim}), ε={epsilon}, δ={delta}")
    print(f"  Required samples: {m}")
    
    # Generalization bound
    print("\n=== Generalization Bounds ===")
    for n in [100, 1000, 10000]:
        bound = generalization_bound(n, vc_dim=10)
        print(f"  n={n:>5}: bound = {bound:.4f}")
    
    # Plot analysis
    plot_sample_complexity()

```

---

## 📊 Summary: PAC Learning Framework

| Component | Mathematical Form | Meaning |
|-----------|-------------------|---------|
| **Accuracy** \(\epsilon\) | \(\text{error}(h) \leq \epsilon\) | How close to perfect |
| **Confidence** \(\delta\) | \(\Pr[\text{success}] \geq 1-\delta\) | Probability of achieving accuracy |
| **Sample Complexity** | \(m(\epsilon, \delta)\) | How many examples needed |
| **Computational Complexity** | \(\text{poly}(1/\epsilon, 1/\delta, n)\) | Time to find hypothesis |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Valiant (1984) - Original PAC paper | [ACM](https://dl.acm.org/doi/10.1145/1968.1972) |
| 📖 | Understanding Machine Learning (Ch. 2-4) | [Shalev-Shwartz](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | Foundations of Machine Learning | [Mohri et al.](https://cs.nyu.edu/~mohri/mlbook/) |
| 📄 | Blumer et al. (1989) - VC dimension | [ML Journal](https://link.springer.com/article/10.1023/A:1022602019183) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../">⬅️ Back: Learning Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../../02_generalization/">Next: Generalization ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
