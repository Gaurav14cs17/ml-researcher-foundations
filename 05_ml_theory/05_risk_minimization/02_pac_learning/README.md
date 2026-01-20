<!-- Navigation -->
<p align="center">
  <a href="../01_erm/">⬅️ Prev: ERM</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_structural_risk/">Next: SRM ➡️</a>
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

<img src="./images/pac-learning.svg" width="100%">

*Caption: PAC learning provides formal guarantees for machine learning algorithms.*

---

## 📂 Overview

**PAC (Probably Approximately Correct)** learning theory, introduced by Valiant (1984), provides a mathematical framework for understanding when learning is possible and how much data is needed.

---

## 📐 PAC Learning Definition

### Formal Definition

A concept class \(\mathcal{C}\) is **PAC-learnable** if there exists an algorithm \(\mathcal{A}\) and a polynomial \(p(\cdot, \cdot, \cdot, \cdot)\) such that:

For all:
- Concepts \(c \in \mathcal{C}\)
- Distributions \(\mathcal{D}\) over \(\mathcal{X}\)
- Accuracy parameter \(\varepsilon > 0\)
- Confidence parameter \(\delta > 0\)

Given \(m \geq p(1/\varepsilon, 1/\delta, n, \text{size}(c))\) samples drawn i.i.d. from \(\mathcal{D}\):

```math
\Pr[R(h) \leq \varepsilon] \geq 1 - \delta

```

where \(R(h) = \Pr_{x \sim \mathcal{D}}[h(x) \neq c(x)]\) is the true error.

---

## 📐 Sample Complexity

### Finite Hypothesis Class

**Theorem:** For a finite hypothesis class \(\mathcal{H}\), the sample complexity is:

```math
m \geq \frac{1}{\varepsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)

```

**Proof:**

Let \(h^* \in \mathcal{H}\) be a hypothesis with \(R(h^*) > \varepsilon\).

For each training example:

```math
\Pr_{(x,y) \sim \mathcal{D}}[h^*(x) = c(x)] \leq 1 - \varepsilon

```

For \(m\) independent samples:

```math
\Pr[\text{all samples labeled correctly by } h^*] \leq (1-\varepsilon)^m \leq e^{-\varepsilon m}

```

By union bound over all "bad" hypotheses:

```math
\Pr[\exists h \in \mathcal{H}: R(h) > \varepsilon \text{ and } \hat{R}(h) = 0] \leq |\mathcal{H}| \cdot e^{-\varepsilon m}

```

Setting this \(\leq \delta\) and solving for \(m\). \(\blacksquare\)

### VC Dimension Based

For hypothesis class with VC dimension \(d\):

```math
m \geq \frac{1}{\varepsilon}\left(d\ln\frac{2}{\varepsilon} + \ln\frac{1}{\delta}\right) = O\left(\frac{d + \ln(1/\delta)}{\varepsilon}\right)

```

---

## 📐 Key Theorems

### Fundamental Theorem of PAC Learning

**Theorem:** The following are equivalent for a hypothesis class \(\mathcal{H}\):
1. \(\mathcal{H}\) is PAC-learnable
2. \(\mathcal{H}\) has finite VC dimension
3. \(\mathcal{H}\) is uniformly learnable (ERM works)

### Occam's Razor Bound

**Theorem:** For a hypothesis \(h \in \mathcal{H}\) with description length \(|h|\) bits:

```math
\Pr[R(h) \leq \hat{R}(h) + \sqrt{\frac{|h|\ln 2 + \ln(1/\delta)}{2m}}] \geq 1 - \delta

```

**Implication:** Shorter (simpler) hypotheses generalize better.

### No Free Lunch Theorem

**Theorem:** For any learning algorithm \(\mathcal{A}\) and sample size \(m < |\mathcal{X}|/2\):

There exists a distribution \(\mathcal{D}\) and labeling such that:
1. There exists a perfect classifier in \(\mathcal{H}\)
2. \(\mathcal{A}\)'s expected error \(\geq 1/4\)

**Implication:** No universal best algorithm. Prior assumptions are necessary.

---

## 💻 Code Implementation

```python
import numpy as np
from math import log, ceil

def pac_sample_complexity_finite(H_size, epsilon, delta):
    """
    Sample complexity for finite hypothesis class.
    
    m ≥ (1/ε)(ln|H| + ln(1/δ))
    
    Args:
        H_size: Size of hypothesis class |H|
        epsilon: Accuracy parameter
        delta: Confidence parameter
    
    Returns:
        Minimum number of samples needed
    """
    return ceil((1/epsilon) * (log(H_size) + log(1/delta)))

def pac_sample_complexity_vc(vc_dim, epsilon, delta):
    """
    Sample complexity for hypothesis class with VC dimension d.
    
    m ≥ (1/ε)(d·ln(2/ε) + ln(1/δ))
    
    Args:
        vc_dim: VC dimension of hypothesis class
        epsilon: Accuracy parameter
        delta: Confidence parameter
    
    Returns:
        Minimum number of samples needed
    """
    return ceil((1/epsilon) * (vc_dim * log(2/epsilon) + log(1/delta)))

def occam_bound(description_length, train_error, m, delta):
    """
    Occam's razor generalization bound.
    
    R(h) ≤ R̂(h) + √((|h|·ln(2) + ln(1/δ)) / (2m))
    
    Args:
        description_length: Bits needed to describe hypothesis
        train_error: Training error
        m: Number of samples
        delta: Confidence parameter
    
    Returns:
        Upper bound on true error
    """
    complexity_term = (description_length * log(2) + log(1/delta)) / (2 * m)
    return train_error + np.sqrt(complexity_term)

class PACLearner:
    """
    Empirical Risk Minimization with PAC guarantees.
    """
    
    def __init__(self, hypothesis_class, epsilon=0.1, delta=0.05):
        self.hypothesis_class = hypothesis_class
        self.epsilon = epsilon
        self.delta = delta
    
    def sample_complexity(self):
        """Calculate required sample size."""
        return pac_sample_complexity_finite(
            len(self.hypothesis_class), 
            self.epsilon, 
            self.delta
        )
    
    def fit(self, X, y):
        """Find hypothesis minimizing empirical risk."""
        best_h = None
        best_error = float('inf')
        
        for h in self.hypothesis_class:
            # Compute empirical risk
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
    
    def pac_bound(self, m):
        """Compute PAC generalization bound."""
        H_size = len(self.hypothesis_class)
        return self.train_error + np.sqrt((log(H_size) + log(1/self.delta)) / (2*m))

# Example: Learning threshold functions
def create_threshold_class(thresholds):
    """Create hypothesis class of threshold functions."""
    return [lambda x, t=t: 1 if x >= t else 0 for t in thresholds]

# Demo
if __name__ == "__main__":
    print("PAC Learning Sample Complexity Examples")
    print("=" * 50)
    
    # Example 1: Finite hypothesis class
    print("\n1. Finite Hypothesis Class")
    print("-" * 30)
    for H_size in [10, 100, 1000, 10000]:
        m = pac_sample_complexity_finite(H_size, epsilon=0.1, delta=0.05)
        print(f"|H| = {H_size:5d}: m ≥ {m}")
    
    # Example 2: VC dimension
    print("\n2. VC Dimension Based")
    print("-" * 30)
    for d in [1, 5, 10, 50, 100]:
        m = pac_sample_complexity_vc(d, epsilon=0.1, delta=0.05)
        print(f"VC(H) = {d:3d}: m ≥ {m}")
    
    # Example 3: Effect of epsilon and delta
    print("\n3. Effect of ε and δ (VC=10)")
    print("-" * 30)
    for eps in [0.1, 0.05, 0.01]:
        for delta in [0.1, 0.05, 0.01]:
            m = pac_sample_complexity_vc(10, eps, delta)
            print(f"ε={eps:.2f}, δ={delta:.2f}: m ≥ {m}")
    
    # Example 4: Occam's razor
    print("\n4. Occam's Razor Bound")
    print("-" * 30)
    for desc_len in [10, 50, 100]:
        bound = occam_bound(desc_len, train_error=0.0, m=1000, delta=0.05)
        print(f"Description length = {desc_len}: R(h) ≤ {bound:.4f}")

```

---

## 📊 Key Concepts Summary

| Concept | Meaning | Importance |
|---------|---------|------------|
| \(\varepsilon\) | Max error we tolerate | Smaller = harder |
| \(\delta\) | Probability of failure | Smaller = more confident |
| VC Dimension | Capacity of \(\mathcal{H}\) | Larger = need more data |
| Sample Complexity | Data needed | Must be polynomial |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Valiant (1984) | [Original PAC paper](https://dl.acm.org/doi/10.1145/1968.1972) |
| 📖 | Understanding ML | [Shalev-Shwartz & Ben-David](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) |
| 📖 | Foundations of ML | [Mohri et al.](https://cs.nyu.edu/~mohri/mlbook/) |

---

⬅️ [Back: ERM](../01_erm/) | ➡️ [Next: Structural Risk](../03_structural_risk/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../01_erm/">⬅️ Prev: ERM</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 Risk Minimization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_structural_risk/">Next: SRM ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
