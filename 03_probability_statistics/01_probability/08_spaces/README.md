<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Probability%20Spaces&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/sample-spaces.svg" width="100%">

*Caption: Sample space Ω contains all possible outcomes. Events are subsets of Ω. Kolmogorov axioms define probability: non-negativity, normalization (P(Ω)=1), and additivity.*

---

## 📂 Overview

Sample spaces and events are the building blocks of probability theory. Understanding these concepts is essential for modeling uncertainty in machine learning.

---

## 📐 Sample Space

### Definition

**Sample space** $\Omega$ is the set of all possible outcomes of an experiment.

**Examples:**
- Coin flip: $\Omega = \{\text{H}, \text{T}\}$

- Die roll: $\Omega = \{1, 2, 3, 4, 5, 6\}$

- Real-valued measurement: $\Omega = \mathbb{R}$

- Image: $\Omega = [0, 255]^{H \times W \times 3}$

---

## 📐 Events

### Definition

An **event** $A$ is a subset of the sample space: $A \subseteq \Omega$

**Examples:**
- "Heads": $A = \{\text{H}\}$

- "Even number": $A = \{2, 4, 6\}$

- "Pixel > 128": $A = \{x \in [0, 255] : x > 128\}$

### Event Operations

| Operation | Symbol | Definition |
|-----------|--------|------------|
| Complement | $A^c$ | Outcomes not in $A$ |
| Union | $A \cup B$ | Outcomes in $A$ or $B$ |
| Intersection | $A \cap B$ | Outcomes in both $A$ and $B$ |
| Difference | $A \setminus B$ | Outcomes in $A$ but not $B$ |

---

## 📐 σ-Algebra (Sigma-Algebra)

### Definition

A **σ-algebra** $\mathcal{F}$ on $\Omega$ is a collection of subsets satisfying:

1. $\Omega \in \mathcal{F}$

2. $A \in \mathcal{F} \Rightarrow A^c \in \mathcal{F}$ (closed under complements)

3. $A_1, A_2, \ldots \in \mathcal{F} \Rightarrow \bigcup_{i=1}^{\infty} A_i \in \mathcal{F}$ (closed under countable unions)

### Why σ-Algebras?

- Define which subsets can have probability assigned

- Required for measure theory foundations

- For finite $\Omega$, typically $\mathcal{F} = 2^\Omega$ (all subsets)

- For $\mathbb{R}$, use the **Borel σ-algebra** (contains all intervals)

---

## 📐 Probability Measure

### Kolmogorov Axioms

A **probability measure** $P$ on $(\Omega, \mathcal{F})$ satisfies:

**Axiom 1 (Non-negativity):**

$$P(A) \geq 0 \quad \forall A \in \mathcal{F}$$

**Axiom 2 (Normalization):**

$$P(\Omega) = 1$$

**Axiom 3 (Countable Additivity):**
For disjoint events $A_1, A_2, \ldots$:

$$P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i)$$

### Probability Space

A **probability space** is the triple $(\Omega, \mathcal{F}, P)$.

---

## 📐 Derived Properties

### Theorem: Properties from Axioms

**1. Empty set:**

$$P(\emptyset) = 0$$

**Proof:** $\Omega$ and $\emptyset$ are disjoint, $\Omega = \Omega \cup \emptyset$.

$$P(\Omega) = P(\Omega) + P(\emptyset) \Rightarrow P(\emptyset) = 0 \quad \blacksquare$$

**2. Complement:**

$$P(A^c) = 1 - P(A)$$

**Proof:** $A$ and $A^c$ are disjoint, $A \cup A^c = \Omega$.

$$1 = P(\Omega) = P(A) + P(A^c) \Rightarrow P(A^c) = 1 - P(A) \quad \blacksquare$$

**3. Monotonicity:**

$$A \subseteq B \Rightarrow P(A) \leq P(B)$$

**Proof:** $B = A \cup (B \setminus A)$ where the union is disjoint.

$$P(B) = P(A) + P(B \setminus A) \geq P(A) \quad \blacksquare$$

**4. Inclusion-Exclusion:**

$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Proof:** Write $A \cup B$ as disjoint union and use additivity. $\quad \blacksquare$

**5. Union Bound (Boole's Inequality):**

$$P\left(\bigcup_{i=1}^{n} A_i\right) \leq \sum_{i=1}^{n} P(A_i)$$

---

## 📐 Conditional Probability

### Definition

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

### Properties

**Theorem:** $P(\cdot|B)$ is a valid probability measure on $(\Omega, \mathcal{F})$.

**Proof:** Verify the three axioms:

1. $P(A|B) = P(A \cap B)/P(B) \geq 0$ ✓

2. $P(\Omega|B) = P(\Omega \cap B)/P(B) = P(B)/P(B) = 1$ ✓

3. Additivity follows from additivity of $P$. $\quad \blacksquare$

---

## 📐 Independence

### Definition

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

**Equivalent condition:**

$$P(A|B) = P(A) \quad \text{and} \quad P(B|A) = P(B)$$

### Mutual Independence

Events $A_1, \ldots, A_n$ are **mutually independent** if for every subset $S \subseteq \{1, \ldots, n\}$:

$$P\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i)$$

**Warning:** Pairwise independence does NOT imply mutual independence!

---

## 📐 Law of Total Probability

### Theorem

If $\{B_1, B_2, \ldots, B_n\}$ is a partition of $\Omega$ (disjoint with $\cup_i B_i = \Omega$):

$$P(A) = \sum_{i=1}^{n} P(A|B_i) P(B_i)$$

**Proof:**

$$A = A \cap \Omega = A \cap \left(\bigcup_i B_i\right) = \bigcup_i (A \cap B_i)$$

Since $A \cap B_i$ are disjoint:

$$P(A) = \sum_i P(A \cap B_i) = \sum_i P(A|B_i) P(B_i) \quad \blacksquare$$

---

## 📐 Random Variables (Preview)

A **random variable** is a measurable function $X: \Omega \to \mathbb{R}$.

"Measurable" means: for any Borel set $B \subseteq \mathbb{R}$:

$$\{ω \in \Omega : X(\omega) \in B\} \in \mathcal{F}$$

This ensures $P(X \in B)$ is well-defined.

---

## 💻 Code Examples

```python
import numpy as np
from scipy import stats

# Discrete sample space: die roll
omega = [1, 2, 3, 4, 5, 6]
p = np.ones(6) / 6  # Uniform probability

# Events as masks
event_even = np.array([x % 2 == 0 for x in omega])
event_greater_3 = np.array([x > 3 for x in omega])

# Probability of event
P_even = p[event_even].sum()  # = 0.5
P_greater_3 = p[event_greater_3].sum()  # = 0.5

# Intersection
event_both = event_even & event_greater_3  # {4, 6}
P_both = p[event_both].sum()  # = 1/3

# Verify inclusion-exclusion
P_union = P_even + P_greater_3 - P_both
print(f"P(even ∪ >3) = {P_union:.4f}")  # = 2/3

# Independence check
# P(A ∩ B) = P(A) * P(B)?
print(f"P(A)P(B) = {P_even * P_greater_3:.4f}")  # = 0.25
print(f"P(A ∩ B) = {P_both:.4f}")  # = 0.333
print("Not independent!")  # 0.25 ≠ 0.333

# Continuous: Gaussian
dist = stats.norm(0, 1)

# P(X > 0) via complement
P_positive = 1 - dist.cdf(0)  # = 0.5

# Monte Carlo probability estimation
def estimate_probability(sample_fn, event_fn, n_samples=10000):
    """
    P(A) ≈ (1/n) Σ 1[x_i ∈ A]
    By Law of Large Numbers, this converges to true P(A)
    """
    samples = sample_fn(n_samples)
    return event_fn(samples).mean()

# Example: P(X² + Y² < 1) for X,Y ~ N(0,1)
def sample_fn(n): 
    return np.random.randn(n, 2)

def event_fn(xy): 
    return (xy**2).sum(axis=1) < 1

prob = estimate_probability(sample_fn, event_fn)
print(f"P(X² + Y² < 1) ≈ {prob:.4f}")  # ≈ 0.393

# Conditional probability example
# P(sum = 7 | first die = 4)
def conditional_sum_7_given_first_4():
    # Sample space conditioned on first die = 4
    second_die = np.array([1, 2, 3, 4, 5, 6])
    sums = 4 + second_die
    return np.mean(sums == 7)  # P(sum=7 | die1=4) = 1/6

print(f"P(sum=7 | die1=4) = {conditional_sum_7_given_first_4():.4f}")

```

---

## 📊 Summary

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Sample Space $\Omega$ | All possible outcomes | Foundation |
| Event $A$ | Subset of $\Omega$ | Has probability |
| σ-algebra $\mathcal{F}$ | Collection of measurable sets | Closed under complements and unions |
| Probability $P$ | Measure on $(\Omega, \mathcal{F})$ | Satisfies Kolmogorov axioms |
| Independence | $P(A \cap B) = P(A)P(B)$ | Events don't affect each other |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Ross: Probability | [Book](https://www.elsevier.com/books/a-first-course-in-probability/ross/978-0-321-79477-2) |
| 🎥 | Stats 110 | [Harvard](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo) |
| 📖 | Billingsley: Probability | [Book](https://www.wiley.com/en-us/Probability+and+Measure%2C+Anniversary+Edition-p-9781118122372) |
| 🇨🇳 | 概率空间详解 | [知乎](https://zhuanlan.zhihu.com/p/26486223) |
| 🇨🇳 | 概率论基础 | [B站](https://www.bilibili.com/video/BV1R4411V7tZ) |

---

⬅️ [Back: Random Variables](../07_random_variables/) | ➡️ [Back: Probability](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
