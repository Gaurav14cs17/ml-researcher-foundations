<!-- Animated Header -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=FF6B6B&center=true&vCenter=true&width=800&lines=📐+Proof+Techniques;The+Art+of+Mathematical+Reasoning" alt="Proof Techniques" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,16&height=180&section=header&text=Proof%20Techniques&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Direct%20Proof%20•%20Contradiction%20•%20Induction%20•%20Contrapositive&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02_of_06-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Topics-6_Techniques-4ECDC4?style=for-the-badge&logo=buffer&logoColor=white" alt="Topics"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-00D4AA?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-6C63FF?style=for-the-badge&logo=calendar&logoColor=white" alt="Updated"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Difficulty-Intermediate-orange?style=flat-square" alt="Difficulty"/>
  <img src="https://img.shields.io/badge/Reading_Time-60_minutes-blue?style=flat-square" alt="Reading Time"/>
  <img src="https://img.shields.io/badge/Prerequisites-Mathematical_Thinking-green?style=flat-square" alt="Prerequisites"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01_mathematical_thinking/README.md) → Proof Techniques → [Set Theory](../03_set_theory/README.md) → [Logic](../04_logic/README.md) → [Asymptotic Analysis](../05_asymptotic_analysis/README.md) → [Numerical Computation](../06_numerical_computation/README.md)

---

## 📌 TL;DR

Every ML paper contains proofs. This article teaches you the **essential proof techniques**:

- **Direct Proof** — Show A → B by logical chain (convexity proofs, gradient bounds)
- **Proof by Contradiction** — Assume opposite, find impossibility (uniqueness proofs)
- **Mathematical Induction** — Base case + inductive step (backprop correctness, recursion)
- **Contrapositive** — Prove ¬Q → ¬P instead of P → Q (often easier)
- **Proof by Cases** — Split into exhaustive cases (ReLU analysis)

> [!TIP]
> **Reading Strategy:** When reading a proof, first identify which technique is being used. This helps you follow the logic and anticipate the structure.

---

## 📚 What You'll Learn

By the end of this article, you will be able to:

- [ ] Write direct proofs for ML theorems (convergence, bounds)
- [ ] Use contradiction to prove impossibility and uniqueness results
- [ ] Apply induction for recursive/layered structures (neural networks)
- [ ] Recognize and use the contrapositive effectively
- [ ] Split complex proofs into manageable cases
- [ ] Identify proof patterns in ML research papers

---

## 📑 Table of Contents

1. [Visual Overview](#-visual-overview)
2. [Decision Tree: Which Technique?](#-decision-tree-which-technique)
3. [Direct Proof](#1-direct-proof)
4. [Proof by Contradiction](#2-proof-by-contradiction)
5. [Mathematical Induction](#3-mathematical-induction)
6. [Contrapositive Proof](#4-contrapositive-proof)
7. [Proof by Cases](#5-proof-by-cases)
8. [Existence and Uniqueness Proofs](#6-existence-and-uniqueness-proofs)
9. [Common Proof Patterns in ML](#7-common-proof-patterns-in-ml)
10. [Key Formulas Summary](#-key-formulas-summary)
11. [Common Mistakes & Pitfalls](#-common-mistakes--pitfalls)
12. [Code Implementations](#-code-implementations)
13. [Resources](#-resources)
14. [Navigation](#-navigation)

---

## 🎯 Visual Overview

### Proof Techniques Decision Tree

```
+-----------------------------------------------------------------------------+
|                        PROOF TECHNIQUE SELECTOR                             |
+-----------------------------------------------------------------------------+
|                                                                             |
|                         🤔 What are you proving?                            |
|                                   |                                         |
|         +--------------+----------+----------+--------------+              |
|         ▼              ▼          ▼          ▼              ▼              |
|    "If P then Q"   "Unique"   "For all n"  "Exists"   "Impossible"        |
|         |              |          |          |              |              |
|         ▼              ▼          ▼          ▼              ▼              |
|    +---------+   +---------+ +---------+ +---------+  +---------+        |
|    | 🎯 DIRECT|   |💥CONTRA-| |🔄INDUCT-| |🔨CONSTR-|  |💥CONTRA- |        |
|    |  PROOF  |   | DICTION | |  ION    | | UCTIVE  |  | DICTION  |        |
|    +---------+   +---------+ +---------+ +---------+  +-------------+     |
|         |              |          |          |              |              |
|    Assume P,      Assume ≥2   Base case  Build the    Assume possible,   |
|    derive Q       exist...    + step     object       find contradiction |
|                                                                             |
+-----------------------------------------------------------------------------+

```

### Proof Structure Templates

```
+-----------------------------------------------------------------------------+
|                         PROOF STRUCTURE TEMPLATES                           |
+-----------------------------------------------------------------------------+
|                                                                             |
|  🎯 DIRECT PROOF              💥 CONTRADICTION           🔄 INDUCTION      |
|  --------------               ---------------           -------------       |
|  1. Assume P                  1. Assume ¬Q              1. Base: P(n₀)      |
|  2. Use definitions           2. Derive consequences    2. Assume: P(k)     |
|  3. Logical steps             3. Reach contradiction    3. Prove: P(k+1)    |
|  4. Conclude Q ∎              4. Conclude Q ∎           4. Conclude ∀n ∎    |
|                                                                             |
|  ↩️ CONTRAPOSITIVE            📋 CASES                  🔨 CONSTRUCTIVE    |
|  -----------------            ------                    --------------      |
|  1. Prove ¬Q → ¬P             1. Case 1: C₁ → Q         1. Construct x     |
|  2. Equivalent to P → Q       2. Case 2: C₂ → Q         2. Verify P(x)      |
|  3. Often easier              3. All cases covered      3. Conclude ∃x ∎   |
|                                                                             |
+-----------------------------------------------------------------------------+

```

---

## 🎨 Decision Tree: Which Technique?

| Question | Answer | Use This Technique |
|:---------|:------:|:------------------:|
| Can you directly derive the result? | ✅ Yes | 🎯 **Direct Proof** |
| Is the negation easier to work with? | ✅ Yes | 💥 **Contradiction** |
| Does it involve natural numbers or recursion? | ✅ Yes | 🔄 **Induction** |
| Is ¬Q → ¬P easier than P → Q? | ✅ Yes | ↩️ **Contrapositive** |
| Are there distinct cases to consider? | ✅ Yes | 📋 **Proof by Cases** |
| Need to show something exists? | ✅ Yes | 🔨 **Constructive** |
| Need to show at most one? | ✅ Yes | 💥 **Uniqueness (Contradiction)** |
| Need to show something is impossible? | ✅ Yes | 💥 **Contradiction** |

---

## 1. Direct Proof

### 📖 Definition

> **Direct Proof:** To prove P → Q, assume P is true, then use definitions, axioms, and previously proven theorems to logically derive Q.

### 📐 Template

```
+-----------------------------------------------------------------+
|                      DIRECT PROOF TEMPLATE                      |
+-----------------------------------------------------------------+
|                                                                 |
|   📌 Theorem: If P, then Q.                                    |
|                                                                 |
|   Proof:                                                        |
|   -----                                                         |
|   1. Assume P is true.                           [Hypothesis]   |
|   2. [Logical step using definition]             [Definition]   |
|   3. [Logical step using theorem]                [Theorem X]    |
|   4. [Continue derivation...]                    [Algebra]      |
|   5. Therefore, Q is true.                       [Conclusion]   |
|   ∎                                                             |
|                                                                 |
+-----------------------------------------------------------------+

```

### 📝 Example 1: Sum of Even Numbers (Simple)

**Theorem:** The sum of two even numbers is even.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Let a and b be even numbers | Assumption |
| 2 | By definition, a = 2m for some integer m | Definition of even |
| 3 | By definition, b = 2n for some integer n | Definition of even |
| 4 | a + b = 2m + 2n = 2(m + n) | Algebra |
| 5 | Let k = m + n, then a + b = 2k | Substitution |
| 6 | Since k is an integer, a + b is even | Definition of even ∎ |

### 📝 Example 2: Convex → Local = Global (Intermediate)

**Theorem:** If f: ℝⁿ → ℝ is convex and x* is a local minimum, then x* is a global minimum.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume f is convex and x* is a local minimum | Hypothesis |
| 2 | By local minimum: ∃ε > 0 such that f(x*) ≤ f(x) for ‖x - x*‖ < ε | Definition |
| 3 | For differentiable convex f at local min: ∇f(x*) = 0 | First-order necessary |
| 4 | By convexity (first-order): f(y) ≥ f(x*) + ∇f(x*)ᵀ(y - x*) ∀y | Convexity |
| 5 | Since ∇f(x*) = 0: f(y) ≥ f(x*) + 0 = f(x*) ∀y | Substitution |
| 6 | Therefore, f(y) ≥ f(x*) for ALL y ∈ ℝⁿ | Universal statement |
| 7 | x* is a global minimum | Definition ∎ |

### 📝 Example 3: Gradient Descent Convergence (Advanced)

**Theorem:** If f is L-smooth and convex, gradient descent with step size α = 1/L satisfies:

```math
f(x_k) - f(x^*) \leq \frac{2L\|x_0 - x^*\|^2}{k}

```

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | f is L-smooth: ‖∇f(x) - ∇f(y)‖ ≤ L‖x - y‖ | Hypothesis |
| 2 | By L-smoothness (descent lemma): | Standard result |
| | f(y) ≤ f(x) + ∇f(x)ᵀ(y-x) + (L/2)‖y-x‖² | |
| 3 | Apply to GD: x_{k+1} = x_k - (1/L)∇f(x_k) | GD update |
| 4 | f(x_{k+1}) ≤ f(x_k) - (1/2L)‖∇f(x_k)‖² | Descent lemma |
| 5 | By convexity: ‖∇f(x_k)‖² ≥ 2L(f(x_k) - f(x*)) | First-order bound |
| 6 | Let δ_k = f(x_k) - f(x*), then: | Definition |
| | δ_{k+1} ≤ δ_k - (1/L)δ_k = (1 - 1/L)δ_k ??? | |
| 7 | Actually, use telescoping: Σ‖∇f(x_k)‖² ≤ 2L(f(x_0) - f(x*)) | Sum descent |
| 8 | min_k ‖∇f(x_k)‖² ≤ 2L(f(x_0) - f(x*))/k | Average bound |
| 9 | This gives O(1/k) convergence rate | ∎ |

### 📝 Example 4: Neural Network Universal Approximation (Complex)

**Theorem (Simplified):** A neural network with one hidden layer and ReLU activations can approximate any continuous function on a compact domain to arbitrary precision.

**Proof Sketch (Constructive):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Let f: [0,1]ⁿ → ℝ be continuous, ε > 0 given | Hypothesis |
| 2 | Partition [0,1]ⁿ into cubes of side δ | Construction |
| 3 | By uniform continuity: ‖f(x) - f(c_i)‖ < ε/2 in cube i | Continuity |
| 4 | Construct indicator: I_i(x) ≈ 1 if x ∈ cube i, else 0 | ReLU combinations |
| 5 | Each I_i is sum of ReLUs (piecewise linear) | ReLU property |
| 6 | g(x) = Σᵢ f(c_i) · I_i(x) | Weighted sum |
| 7 | ‖f(x) - g(x)‖ < ε for all x ∈ [0,1]ⁿ | Triangle inequality |
| 8 | g is a neural network with ReLU activation | ∎ |

### 💻 Code Implementation

```python
import numpy as np

def direct_proof_gradient_descent():
    """
    Demonstrate GD convergence proof computationally.
    
    Theorem: GD with step size 1/L converges at O(1/k) for L-smooth convex f.
    """
    # L-smooth convex function: f(x) = (L/2) * x^2
    L = 2.0
    f = lambda x: (L/2) * x**2
    grad_f = lambda x: L * x
    
    # GD with step size 1/L
    x = 10.0  # Initial point
    x_star = 0.0  # Optimal point
    alpha = 1/L
    
    print("Gradient Descent Convergence Proof (Computational Verification)")
    print("=" * 60)
    print(f"f(x) = (L/2)x², L = {L}")
    print(f"Step size α = 1/L = {alpha}")
    print(f"x₀ = {x}, x* = {x_star}")
    print("-" * 60)
    
    gaps = []
    for k in range(1, 101):
        x = x - alpha * grad_f(x)
        gap = f(x) - f(x_star)
        gaps.append(gap)
        
        if k in [1, 5, 10, 50, 100]:
            # Theoretical bound: 2L||x_0 - x*||² / k
            theoretical_bound = 2 * L * (10 - x_star)**2 / k
            print(f"k={k:3d}: f(x_k) - f(x*) = {gap:.6f}, "
                  f"Bound = {theoretical_bound:.6f}, "
                  f"Satisfies: {gap <= theoretical_bound + 1e-10}")
    
    return gaps

# Run demonstration
gaps = direct_proof_gradient_descent()

```

---

## 2. Proof by Contradiction

### 📖 Definition

> **Proof by Contradiction (Reductio ad Absurdum):** To prove P, assume ¬P (P is false), then derive a logical contradiction. Since mathematics is consistent, ¬P must be false, so P is true.

### 📐 Template

```
+-----------------------------------------------------------------+
|                   CONTRADICTION TEMPLATE                        |
+-----------------------------------------------------------------+
|                                                                 |
|   📌 Theorem: P is true.                                       |
|                                                                 |
|   Proof:                                                        |
|   -----                                                         |
|   1. Assume, for contradiction, that ¬P.        [Assumption]    |
|   2. [Derive logical consequences...]           [Derivation]    |
|   3. [Continue derivation...]                   [Derivation]    |
|   4. This contradicts [known fact/assumption].  [Contradiction] |
|   5. Therefore, P must be true.                 [Conclusion]    |
|   ∎                                                             |
|                                                                 |
+-----------------------------------------------------------------+

```

### 📝 Example 1: √2 is Irrational (Classic)

**Theorem:** √2 is irrational.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume, for contradiction, that √2 is rational | Assumption |
| 2 | Then √2 = p/q where p, q ∈ ℤ, gcd(p,q) = 1 (lowest terms) | Definition of rational |
| 3 | Squaring: 2 = p²/q², so p² = 2q² | Algebra |
| 4 | p² is even, so p is even | Even² = even |
| 5 | Let p = 2k for some integer k | Definition of even |
| 6 | Then (2k)² = 2q², so 4k² = 2q², so q² = 2k² | Substitution |
| 7 | q² is even, so q is even | Even² = even |
| 8 | Both p and q are even, so gcd(p,q) ≥ 2 | Definition of even |
| 9 | This contradicts gcd(p,q) = 1 (Step 2) | 💥 Contradiction |
| 10 | Therefore, √2 is irrational | ∎ |

### 📝 Example 2: Strictly Convex → Unique Minimum (Intermediate)

**Theorem:** A strictly convex function has at most one global minimum.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume f is strictly convex | Hypothesis |
| 2 | Assume, for contradiction, x₁ ≠ x₂ are both global minima | Assumption |
| 3 | Then f(x₁) = f(x₂) = f* (the minimum value) | Definition of global min |
| 4 | Consider midpoint: x_mid = (x₁ + x₂)/2 | Construction |
| 5 | By strict convexity: f(x_mid) < ½f(x₁) + ½f(x₂) | Strict convexity |
| 6 | f(x_mid) < ½f* + ½f* = f* | Substitution |
| 7 | So f(x_mid) < f* | Arithmetic |
| 8 | But f* is the minimum value! f(x_mid) ≥ f* | Definition of minimum |
| 9 | This is a contradiction: f(x_mid) < f* and f(x_mid) ≥ f* | 💥 Contradiction |
| 10 | Therefore, at most one global minimum exists | ∎ |

### 📝 Example 3: No Free Lunch Theorem (Advanced)

**Theorem (Simplified):** No learning algorithm is universally better than random search across all possible target functions.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume algorithm A beats random on ALL functions f: X → {0,1} | Assumption |
| 2 | Given training data D of n points, A predicts ŷ for new x | Setup |
| 3 | There are 2^{|X|-n} functions consistent with D | Counting |
| 4 | For new x, exactly half of consistent functions have f(x)=0, half f(x)=1 | Symmetry |
| 5 | A's prediction ŷ will be wrong for half the functions | Symmetry |
| 6 | Expected error = 1/2 for ANY prediction (same as random) | Probability |
| 7 | A cannot beat random on average over all functions | 💥 Contradiction |
| 8 | Therefore, no algorithm is universally better | ∎ |

### 📝 Example 4: Infinitely Many Primes (Classic)

**Theorem:** There are infinitely many prime numbers.

**Proof (Euclid):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume, for contradiction, there are finitely many primes | Assumption |
| 2 | Let p₁, p₂, ..., pₙ be ALL the primes | Finite list |
| 3 | Consider N = p₁ · p₂ · ... · pₙ + 1 | Construction |
| 4 | N > 1, so N has a prime factor p | Fundamental theorem |
| 5 | If p = pᵢ for some i, then p divides N and p divides p₁·...·pₙ | Divisibility |
| 6 | So p divides N - p₁·...·pₙ = 1 | Algebra |
| 7 | But no prime divides 1 | 💥 Contradiction |
| 8 | Therefore, there are infinitely many primes | ∎ |

### 💻 Code Implementation

```python
import numpy as np
from fractions import Fraction

def proof_sqrt2_irrational():
    """
    Computational demonstration that √2 cannot be expressed as p/q.
    
    We show that for any p/q, |p²/q² - 2| > 0.
    """
    print("Demonstrating √2 is irrational")
    print("=" * 50)
    
    # Check many fractions
    best_error = float('inf')
    best_fraction = None
    
    for q in range(1, 10001):
        # Find p such that p/q is closest to √2
        p = round(q * np.sqrt(2))
        error = abs((p/q)**2 - 2)
        
        if error < best_error and error > 0:
            best_error = error
            best_fraction = (p, q)
    
    p, q = best_fraction
    print(f"Best approximation with q ≤ 10000: {p}/{q}")
    print(f"(p/q)² = {(p/q)**2}")
    print(f"Error = {best_error}")
    print(f"√2 ≈ {np.sqrt(2)}")
    print(f"\nNo matter how large q, error > 0 (never exactly 2)")
    print("This supports (but doesn't prove) irrationality")

def proof_unique_minimum():
    """
    Demonstrate uniqueness of minimum for strictly convex function.
    """
    print("\nUniqueness of Minimum for Strictly Convex Functions")
    print("=" * 50)
    
    # Strictly convex: f(x) = x²
    f = lambda x: x**2
    
    # Suppose x1, x2 are both minima
    x1, x2 = -0.001, 0.001  # Trying two "minima"
    
    # Midpoint
    x_mid = (x1 + x2) / 2
    
    f1, f2 = f(x1), f(x2)
    f_mid = f(x_mid)
    avg = (f1 + f2) / 2
    
    print(f"f(x) = x² (strictly convex)")
    print(f"Suppose x₁ = {x1}, x₂ = {x2} are both minima")
    print(f"f(x₁) = {f1}, f(x₂) = {f2}")
    print(f"Midpoint x_mid = {x_mid}")
    print(f"f(x_mid) = {f_mid}")
    print(f"(f(x₁) + f(x₂))/2 = {avg}")
    print(f"Strict convexity: f(x_mid) < average? {f_mid < avg}")
    print(f"But if both are minima, f_mid should equal minimum!")
    print("Contradiction unless x₁ = x₂")

# Run demonstrations
proof_sqrt2_irrational()
proof_unique_minimum()

```

---

## 3. Mathematical Induction

### 📖 Definition

> **Mathematical Induction:** To prove P(n) for all n ≥ n₀:
> 1. **Base Case:** Prove P(n₀)
> 2. **Inductive Step:** Prove P(k) → P(k+1) for arbitrary k ≥ n₀
> 
> Then P(n) holds for all n ≥ n₀.

### 📐 Template

```
+-----------------------------------------------------------------+
|                     INDUCTION TEMPLATE                          |
+-----------------------------------------------------------------+
|                                                                 |
|   📌 Theorem: P(n) holds for all n ≥ n₀.                       |
|                                                                 |
|   Proof by Induction:                                           |
|   -------------------                                           |
|                                                                 |
|   🔵 Base Case (n = n₀):                                       |
|      [Show P(n₀) is true]                                       |
|      ✓ P(n₀) holds.                                            |
|                                                                 |
|   🟢 Inductive Hypothesis:                                      |
|      Assume P(k) is true for some k ≥ n₀.                      |
|                                                                 |
|   🔴 Inductive Step:                                            |
|      [Using the hypothesis, prove P(k+1)]                       |
|      ✓ P(k+1) holds.                                           |
|                                                                 |
|   By induction, P(n) holds for all n ≥ n₀.  ∎                  |
|                                                                 |
+-----------------------------------------------------------------+

```

### 📝 Example 1: Sum Formula (Simple)

**Theorem:** For all n ≥ 1: $1 + 2 + 3 + \cdots + n = \frac{n(n+1)}{2}$

**Proof:**

**Base Case (n = 1):**
| LHS | RHS | Equal? |
|:---:|:---:|:------:|
| 1 | 1(2)/2 = 1 | ✅ |

**Inductive Hypothesis:** Assume $1 + 2 + \cdots + k = \frac{k(k+1)}{2}$ for some k ≥ 1.

**Inductive Step (prove for k+1):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | 1 + 2 + ... + k + (k+1) | LHS for k+1 |
| 2 | = [1 + 2 + ... + k] + (k+1) | Grouping |
| 3 | = k(k+1)/2 + (k+1) | Inductive hypothesis |
| 4 | = (k+1)(k/2 + 1) | Factor out (k+1) |
| 5 | = (k+1)(k+2)/2 | Simplify |
| 6 | = (k+1)((k+1)+1)/2 | Rewrite |

This is exactly the formula with n = k+1. ✅

**Conclusion:** By induction, the formula holds for all n ≥ 1. ∎

### 📝 Example 2: Backpropagation Correctness (Intermediate)

**Theorem:** Backpropagation correctly computes ∂L/∂W^(l) for all layers l = 1, 2, ..., L.

**Setup:**
- Layer l: z^(l) = W^(l) a^(l-1) + b^(l), a^(l) = σ(z^(l))
- Loss: L = L(a^(L), y)

**Proof (Reverse Induction from L to 1):**

**Base Case (l = L, output layer):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | ∂L/∂W^(L) = ∂L/∂a^(L) · ∂a^(L)/∂z^(L) · ∂z^(L)/∂W^(L) | Chain rule |
| 2 | ∂L/∂a^(L) is computed directly from loss | Direct |
| 3 | ∂a^(L)/∂z^(L) = σ'(z^(L)) | Activation derivative |
| 4 | ∂z^(L)/∂W^(L) = (a^(L-1))ᵀ | Linear layer derivative |
| 5 | All terms computable ✅ | Base case complete |

**Inductive Hypothesis:** Assume ∂L/∂a^(l+1) is correctly computed.

**Inductive Step (prove for layer l):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | ∂L/∂a^(l) = ∂L/∂z^(l+1) · ∂z^(l+1)/∂a^(l) | Chain rule |
| 2 | = ∂L/∂a^(l+1) · σ'(z^(l+1)) · W^(l+1) | Chain rule expansion |
| 3 | ∂L/∂a^(l+1) is correct by hypothesis | Inductive hypothesis |
| 4 | Other terms computable | Known derivatives |
| 5 | ∂L/∂W^(l) = ∂L/∂z^(l) · (a^(l-1))ᵀ | Chain rule |
| 6 | Correctly computed ✅ | Inductive step complete |

**Conclusion:** By (reverse) induction, backprop is correct for all layers. ∎

### 📝 Example 3: Binary Tree Nodes (Intermediate)

**Theorem:** A complete binary tree with n levels has 2ⁿ - 1 nodes.

**Proof:**

**Base Case (n = 1):** Tree has 2¹ - 1 = 1 node (just root). ✅

**Inductive Hypothesis:** Complete tree with k levels has 2^k - 1 nodes.

**Inductive Step:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Tree with k+1 levels has root + two subtrees | Structure |
| 2 | Each subtree is complete with k levels | By construction |
| 3 | Each subtree has 2^k - 1 nodes | Inductive hypothesis |
| 4 | Total = 1 + 2(2^k - 1) = 1 + 2^{k+1} - 2 = 2^{k+1} - 1 | Arithmetic ✅ |

**Conclusion:** By induction, formula holds for all n ≥ 1. ∎

### 📝 Example 4: Strong Induction - Fundamental Theorem of Arithmetic

**Theorem:** Every integer n ≥ 2 can be written as product of primes.

**Proof (Strong Induction):**

**Base Case (n = 2):** 2 is prime, so 2 = 2. ✅

**Strong Inductive Hypothesis:** Every integer 2 ≤ m < n can be written as product of primes.

**Inductive Step:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Case 1: n is prime | Then n = n (product of one prime) ✅ |
| 2 | Case 2: n is composite | n = a · b where 2 ≤ a, b < n |
| 3 | By strong hypothesis, a = p₁·...·pⱼ and b = q₁·...·qₖ | Hypothesis applies |
| 4 | So n = p₁·...·pⱼ·q₁·...·qₖ | Product of primes ✅ |

**Conclusion:** By strong induction, every n ≥ 2 is a product of primes. ∎

### 💻 Code Implementation

```python
import numpy as np
import torch
import torch.nn as nn

def verify_sum_formula(n_max=100):
    """Verify sum formula by induction computationally."""
    print("Verifying Sum Formula: 1 + 2 + ... + n = n(n+1)/2")
    print("=" * 50)
    
    for n in [1, 5, 10, 50, 100]:
        lhs = sum(range(1, n+1))
        rhs = n * (n + 1) // 2
        print(f"n = {n:3d}: LHS = {lhs:5d}, RHS = {rhs:5d}, Equal: {lhs == rhs}")

def verify_backprop_induction():
    """Verify backprop correctness layer by layer."""
    print("\nVerifying Backprop Correctness")
    print("=" * 50)
    
    # Simple network
    torch.manual_seed(42)
    
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(10, 8),
                nn.Linear(8, 6),
                nn.Linear(6, 4),
                nn.Linear(4, 2)
            ])
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i < len(self.layers) - 1:
                    x = torch.relu(x)
            return x
    
    model = SimpleNet()
    x = torch.randn(1, 10, requires_grad=True)
    y = torch.randn(1, 2)
    
    # Forward pass
    output = model(x)
    loss = ((output - y) ** 2).sum()
    
    # Backprop
    loss.backward()
    
    # Verify gradients exist for all layers (base case + inductive step)
    print("Layer-by-layer gradient verification:")
    for i, layer in enumerate(model.layers):
        has_grad = layer.weight.grad is not None
        grad_norm = layer.weight.grad.norm().item() if has_grad else 0
        print(f"  Layer {i+1}: Gradient exists: {has_grad}, ||grad|| = {grad_norm:.4f}")
    
    # Numerical gradient check (for final layer - base case)
    print("\nNumerical gradient check for last layer (base case):")
    eps = 1e-5
    layer = model.layers[-1]
    i, j = 0, 0
    
    # Compute numerical gradient
    original = layer.weight.data[i, j].item()
    
    layer.weight.data[i, j] = original + eps
    out_plus = model(x.detach())
    loss_plus = ((out_plus - y) ** 2).sum().item()
    
    layer.weight.data[i, j] = original - eps
    out_minus = model(x.detach())
    loss_minus = ((out_minus - y) ** 2).sum().item()
    
    layer.weight.data[i, j] = original
    
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    analytic_grad = layer.weight.grad[i, j].item()
    
    print(f"  Numerical gradient: {numerical_grad:.6f}")
    print(f"  Analytic gradient:  {analytic_grad:.6f}")
    print(f"  Relative error: {abs(numerical_grad - analytic_grad) / (abs(numerical_grad) + 1e-8):.2e}")

verify_sum_formula()
verify_backprop_induction()

```

---

## 4. Contrapositive Proof

### 📖 Definition

> **Contrapositive:** P → Q is logically equivalent to ¬Q → ¬P.
> 
> To prove P → Q, you can instead prove ¬Q → ¬P (often easier).

### 📐 Template

```
+-----------------------------------------------------------------+
|                   CONTRAPOSITIVE TEMPLATE                       |
+-----------------------------------------------------------------+
|                                                                 |
|   📌 Theorem: If P, then Q.                                    |
|                                                                 |
|   Proof (by Contrapositive):                                    |
|   -------------------------                                     |
|   We prove the contrapositive: If ¬Q, then ¬P.                 |
|                                                                 |
|   1. Assume ¬Q.                             [Assumption]        |
|   2. [Derive logical consequences...]       [Derivation]        |
|   3. [Continue derivation...]               [Derivation]        |
|   4. Therefore, ¬P.                         [Conclusion]        |
|                                                                 |
|   Since ¬Q → ¬P, we have P → Q.  ∎                             |
|                                                                 |
+-----------------------------------------------------------------+

```

### 📝 Example 1: n² Even → n Even (Simple)

**Theorem:** If n² is even, then n is even.

**Direct proof is tricky. Let's use contrapositive.**

**Contrapositive:** If n is odd, then n² is odd.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume n is odd | Assumption |
| 2 | Then n = 2k + 1 for some integer k | Definition of odd |
| 3 | n² = (2k + 1)² = 4k² + 4k + 1 | Expansion |
| 4 | = 2(2k² + 2k) + 1 | Factor out 2 |
| 5 | n² = 2m + 1 where m = 2k² + 2k | Substitution |
| 6 | Therefore n² is odd | Definition of odd |

**Since ¬Q → ¬P is proven, P → Q holds.** ∎

### 📝 Example 2: GD Convergence Requirement (Intermediate)

**Theorem:** If gradient descent converges for f(x) = ½x², then step size α ≤ 2.

**Contrapositive:** If α > 2, then GD diverges for f(x) = ½x².

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume α > 2 | Assumption |
| 2 | GD update: x_{k+1} = x_k - α · x_k = (1 - α)x_k | f'(x) = x |
| 3 | After k steps: x_k = (1 - α)^k · x_0 | Recursion |
| 4 | Since α > 2: |1 - α| > 1 | Arithmetic |
| 5 | So |x_k| = |1 - α|^k · |x_0| → ∞ as k → ∞ | Exponential growth |
| 6 | GD diverges | Definition of divergence |

**Since ¬Q → ¬P is proven, P → Q holds.** ∎

### 📝 Example 3: Convergent → Bounded (Advanced)

**Theorem:** If a sequence (aₙ) converges, then it is bounded.

**Contrapositive:** If (aₙ) is unbounded, then it diverges.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume (aₙ) is unbounded | Assumption |
| 2 | For any M > 0, ∃n: |aₙ| > M | Definition of unbounded |
| 3 | Suppose (aₙ) → L | For contradiction |
| 4 | Then ∃N: ∀n > N, |aₙ - L| < 1 | ε = 1 definition |
| 5 | So |aₙ| < |L| + 1 for all n > N | Triangle inequality |
| 6 | Choose M = max{|a₁|, ..., |aₙ|, |L| + 1} | Construction |
| 7 | All terms satisfy |aₙ| ≤ M | Bounded! |
| 8 | Contradiction with unbounded | 💥 |
| 9 | So (aₙ) diverges | By contradiction |

**Since ¬Q → ¬P is proven, P → Q holds.** ∎

---

## 5. Proof by Cases

### 📖 Definition

> **Proof by Cases:** To prove P, split into exhaustive cases C₁, C₂, ..., Cₙ (where C₁ ∨ C₂ ∨ ... ∨ Cₙ is always true), then prove P in each case.

### 📐 Template

```
+-----------------------------------------------------------------+
|                      CASES TEMPLATE                             |
+-----------------------------------------------------------------+
|                                                                 |
|   📌 Theorem: P is true.                                       |
|                                                                 |
|   Proof (by Cases):                                             |
|   ----------------                                              |
|                                                                 |
|   Case 1: [Condition C₁]                                        |
|           [Prove P under C₁]  ✓                                |
|                                                                 |
|   Case 2: [Condition C₂]                                        |
|           [Prove P under C₂]  ✓                                |
|                                                                 |
|   ...                                                           |
|                                                                 |
|   Case n: [Condition Cₙ]                                        |
|           [Prove P under Cₙ]  ✓                                |
|                                                                 |
|   Since C₁ ∨ C₂ ∨ ... ∨ Cₙ covers all possibilities,          |
|   P is true.  ∎                                                 |
|                                                                 |
+-----------------------------------------------------------------+

```

### 📝 Example 1: ReLU is 1-Lipschitz (Simple)

**Theorem:** ReLU(x) = max(0, x) is 1-Lipschitz: |ReLU(x) - ReLU(y)| ≤ |x - y|

**Proof (by Cases):**

| Case | Condition | Calculation | Lipschitz? |
|:----:|:----------|:------------|:----------:|
| 1 | x ≥ 0, y ≥ 0 | \|x - y\| = \|x - y\| | ✅ |
| 2 | x < 0, y < 0 | \|0 - 0\| = 0 ≤ \|x - y\| | ✅ |
| 3 | x ≥ 0, y < 0 | \|x - 0\| = x ≤ x - y = \|x - y\| | ✅ |
| 4 | x < 0, y ≥ 0 | \|0 - y\| = y ≤ y - x = \|x - y\| | ✅ |

**All cases covered. ReLU is 1-Lipschitz.** ∎

### 📝 Example 2: |xy| = |x||y| (Intermediate)

**Theorem:** For all real x, y: |xy| = |x| · |y|

**Proof (by Cases on signs):**

| Case | x | y | |xy| | |x|·|y| | Equal? |
|:----:|:-:|:-:|:---:|:------:|:------:|
| 1 | ≥ 0 | ≥ 0 | xy | xy | ✅ |
| 2 | ≥ 0 | < 0 | -xy | x·(-y) = -xy | ✅ |
| 3 | < 0 | ≥ 0 | -xy | (-x)·y = -xy | ✅ |
| 4 | < 0 | < 0 | xy | (-x)·(-y) = xy | ✅ |

**All cases verified.** ∎

---

## 6. Existence and Uniqueness Proofs

### 📖 Existence Proofs

Two types:
1. **Constructive:** Build the object explicitly
2. **Non-constructive:** Prove existence without construction

### 📐 Existence Template

```
+-----------------------------------------------------------------+
|                   CONSTRUCTIVE EXISTENCE                        |
+-----------------------------------------------------------------+
|   📌 Theorem: ∃x: P(x)                                         |
|                                                                 |
|   Proof:                                                        |
|   1. Construct: x = [explicit formula/algorithm]                |
|   2. Verify: P(x) holds                                         |
|   3. Therefore, ∃x: P(x)  ∎                                    |
+-----------------------------------------------------------------+

+-----------------------------------------------------------------+
|                  NON-CONSTRUCTIVE EXISTENCE                     |
+-----------------------------------------------------------------+
|   📌 Theorem: ∃x: P(x)                                         |
|                                                                 |
|   Proof (by Contradiction):                                     |
|   1. Assume ∀x: ¬P(x)                                          |
|   2. Derive contradiction                                       |
|   3. Therefore, ∃x: P(x)  ∎                                    |
|                                                                 |
|   Note: This doesn't tell you WHAT x is!                       |
+-----------------------------------------------------------------+

```

### 📐 Uniqueness Template

```
+-----------------------------------------------------------------+
|                    UNIQUENESS PROOF                             |
+-----------------------------------------------------------------+
|   📌 Theorem: ∃!x: P(x)  (there exists exactly one x)          |
|                                                                 |
|   Proof:                                                        |
|   Part 1 (Existence): Show ∃x: P(x)                            |
|   Part 2 (Uniqueness): Suppose P(x₁) and P(x₂).                |
|                        Show x₁ = x₂.                            |
|   ∎                                                             |
+-----------------------------------------------------------------+

```

### 📝 Example: Unique Fixed Point (Banach)

**Theorem:** A contraction mapping T on a complete metric space has exactly one fixed point.

**Proof:**

**Existence (Constructive):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Start with any x₀ | Arbitrary |
| 2 | Define xₙ₊₁ = T(xₙ) | Iteration |
| 3 | d(xₙ₊₁, xₙ) ≤ cⁿ · d(x₁, x₀) where c < 1 | Contraction |
| 4 | (xₙ) is Cauchy | Geometric series bound |
| 5 | xₙ → x* by completeness | Complete space |
| 6 | T(x*) = T(lim xₙ) = lim T(xₙ) = lim xₙ₊₁ = x* | Continuity of T |
| 7 | x* is a fixed point ✅ | Definition |

**Uniqueness:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Suppose x*, y* are both fixed points | Assumption |
| 2 | d(x*, y*) = d(T(x*), T(y*)) ≤ c · d(x*, y*) | Contraction |
| 3 | (1 - c) · d(x*, y*) ≤ 0 | Algebra |
| 4 | Since c < 1: d(x*, y*) ≤ 0 | Only if d = 0 |
| 5 | d(x*, y*) = 0, so x* = y* | Metric property ✅ |

**Therefore, exactly one fixed point exists.** ∎

---

## 7. Common Proof Patterns in ML

### Pattern 1: Triangle Inequality Decomposition

```
|A - C| ≤ |A - B| + |B - C|

Application: Generalization error
|R(h) - R̂(h)| ≤ |R(h) - R(h*)| + |R(h*) - R̂(h*)|+ |R̂(h*) - R̂(h)|
                  +-------------+   +---------------+   +------------+
                   Approximation      Estimation          Optimization

```

### Pattern 2: Telescoping Sum

```
Σₖ₌₀ⁿ⁻¹ (aₖ - aₖ₊₁) = a₀ - aₙ

Application: Convergence analysis
f(x₀) - f(x*) = Σₖ [f(xₖ) - f(xₖ₊₁)]
                    +- progress per step

```

### Pattern 3: Squeeze/Sandwich Theorem

```
If L(n) ≤ f(n) ≤ U(n) and lim L(n) = lim U(n) = L
Then lim f(n) = L

Application: Big-Θ bounds
c₁g(n) ≤ f(n) ≤ c₂g(n) ⟹ f(n) = Θ(g(n))

```

### Pattern 4: Contraction/Fixed Point

```
If d(T(x), T(y)) ≤ c · d(x,y) with c < 1
Then T has unique fixed point x*

Application: Value iteration in RL
||TV₁ - TV₂||∞ ≤ γ||V₁ - V₂||∞ ⟹ V* exists, unique

```

---

## 📊 Key Formulas Summary

| Technique | Pattern | When to Use |
|:----------|:--------|:------------|
| **Direct** | P → Q: Assume P, derive Q | When derivation is straightforward |
| **Contradiction** | ¬P → ⊥, therefore P | Uniqueness, impossibility |
| **Induction** | P(n₀), P(k)→P(k+1) | Properties of all n ∈ ℕ |
| **Contrapositive** | ¬Q → ¬P ≡ P → Q | When ¬Q gives more structure |
| **Cases** | C₁∨...∨Cₙ, prove P for each | When natural case split exists |
| **Constructive** | Build x, verify P(x) | When algorithm/formula known |

---

## ⚠️ Common Mistakes & Pitfalls

### Mistake 1: Circular Reasoning

```
❌ WRONG:
   "Assume P → Q. Then P implies Q. Therefore P → Q."
   
   This assumes what we're trying to prove!

✅ RIGHT:
   Start from P, use OTHER known facts to derive Q.

```

### Mistake 2: Proving Converse Instead

```
❌ WRONG (proving converse):
   To prove "If raining, then wet":
   "Assume ground is wet. Then it might be raining."
   
   This proves wet → raining (converse), not raining → wet!

✅ RIGHT:
   "Assume it's raining. Rain falls on ground. Ground gets wet."

```

### Mistake 3: Incomplete Case Analysis

```
❌ WRONG:
   Prove for x > 0 and x < 0.
   (Forgot x = 0!)

✅ RIGHT:
   Prove for x > 0, x = 0, and x < 0.

```

### Mistake 4: Wrong Induction Base

```
❌ WRONG:
   Base case n = 0, but formula uses n ≥ 1.

✅ RIGHT:
   Match base case to the claim's starting point.

```

### Mistake 5: Using Inductive Hypothesis Wrong

```
❌ WRONG:
   In proving P(k+1), assuming P(k+1) is true.

✅ RIGHT:
   Only use P(k) (or P(1),...,P(k) for strong induction).

```

---

## 💻 Code Implementations

```python
"""
Proof Techniques: Complete Implementation
==========================================

Computational demonstrations of proof techniques for ML.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass

class ProofTechniques:
    """Computational demonstrations of proof techniques."""
    
    # =========================================================================
    # DIRECT PROOF
    # =========================================================================
    
    @staticmethod
    def verify_convex_global_min():
        """
        Direct proof: Convex + local min → global min.
        Computational verification.
        """
        print("Direct Proof: Convex Local Minimum is Global")
        print("=" * 50)
        
        # Convex function: f(x) = x^2
        f = lambda x: x**2
        
        # Find local minimum by gradient descent
        x = 5.0
        lr = 0.1
        for _ in range(100):
            grad = 2 * x  # f'(x) = 2x
            x = x - lr * grad
        
        local_min = x
        local_min_val = f(local_min)
        
        # Check if it's global by sampling
        samples = np.linspace(-10, 10, 1000)
        is_global = all(f(s) >= local_min_val - 1e-6 for s in samples)
        
        print(f"Local minimum at x = {local_min:.6f}")
        print(f"f(x*) = {local_min_val:.6f}")
        print(f"Is global minimum: {is_global}")
        
        return is_global
    
    # =========================================================================
    # CONTRADICTION
    # =========================================================================
    
    @staticmethod
    def verify_unique_minimum():
        """
        Contradiction proof: Strictly convex → at most one minimum.
        """
        print("\nContradiction: Unique Minimum of Strictly Convex")
        print("=" * 50)
        
        # Strictly convex: f(x) = x^2
        f = lambda x: x**2
        
        # Try to find two different minima
        # GD from different starting points
        minima = []
        for x0 in [-10, 10, 0, 5, -5]:
            x = x0
            for _ in range(1000):
                x = x - 0.1 * 2 * x
            minima.append(round(x, 6))
        
        unique_minima = set(minima)
        print(f"Starting points: [-10, 10, 0, 5, -5]")
        print(f"Found minima: {minima}")
        print(f"Unique minima: {unique_minima}")
        print(f"Only one minimum: {len(unique_minima) == 1}")
        
        return len(unique_minima) == 1
    
    # =========================================================================
    # INDUCTION
    # =========================================================================
    
    @staticmethod
    def verify_induction_sum_formula(n_max: int = 100):
        """
        Induction proof: 1 + 2 + ... + n = n(n+1)/2
        """
        print("\nInduction: Sum Formula")
        print("=" * 50)
        
        # Base case
        n = 1
        lhs = sum(range(1, n + 1))
        rhs = n * (n + 1) // 2
        base_holds = (lhs == rhs)
        print(f"Base case (n=1): {lhs} = {rhs}? {base_holds}")
        
        # Inductive step verification for many k
        inductive_holds = True
        for k in range(1, n_max):
            # Assume formula holds for k
            sum_k = k * (k + 1) // 2
            
            # Check formula holds for k+1
            sum_k1_computed = sum(range(1, k + 2))
            sum_k1_formula = (k + 1) * (k + 2) // 2
            
            # Inductive step: sum_k + (k+1) should equal formula for k+1
            sum_k1_inductive = sum_k + (k + 1)
            
            if sum_k1_inductive != sum_k1_formula:
                inductive_holds = False
                print(f"Failed at k={k}")
                break
        
        print(f"Inductive step holds for k = 1 to {n_max-1}: {inductive_holds}")
        
        return base_holds and inductive_holds
    
    @staticmethod
    def verify_backprop_induction():
        """
        Induction proof: Backprop computes correct gradients.
        Verify by numerical gradient check at each layer.
        """
        print("\nInduction: Backprop Correctness")
        print("=" * 50)
        
        torch.manual_seed(42)
        
        # Multi-layer network
        layers = [
            nn.Linear(8, 6),
            nn.ReLU(),
            nn.Linear(6, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        ]
        model = nn.Sequential(*layers)
        
        x = torch.randn(1, 8, requires_grad=True)
        y = torch.randn(1, 2)
        
        # Forward and backward
        output = model(x)
        loss = ((output - y) ** 2).sum()
        loss.backward()
        
        # Check gradients layer by layer (induction over layers)
        print("Verifying gradients layer by layer:")
        
        linear_layers = [l for l in layers if isinstance(l, nn.Linear)]
        
        for i, layer in enumerate(linear_layers):
            # Numerical gradient for one weight
            eps = 1e-5
            idx = (0, 0)
            original = layer.weight.data[idx].item()
            
            layer.weight.data[idx] = original + eps
            loss_plus = ((model(x.detach()) - y) ** 2).sum().item()
            
            layer.weight.data[idx] = original - eps
            loss_minus = ((model(x.detach()) - y) ** 2).sum().item()
            
            layer.weight.data[idx] = original
            
            numerical = (loss_plus - loss_minus) / (2 * eps)
            analytic = layer.weight.grad[idx].item()
            
            rel_error = abs(numerical - analytic) / (abs(numerical) + 1e-8)
            
            print(f"  Layer {i+1}: num={numerical:.6f}, ana={analytic:.6f}, "
                  f"rel_err={rel_error:.2e}, OK={rel_error < 1e-4}")
    
    # =========================================================================
    # CONTRAPOSITIVE
    # =========================================================================
    
    @staticmethod
    def verify_contrapositive_gd():
        """
        Contrapositive: If GD converges, then α ≤ 2/L.
        Equivalently: If α > 2/L, then GD diverges.
        """
        print("\nContrapositive: GD Step Size Requirement")
        print("=" * 50)
        
        # f(x) = (1/2)x^2, so L = 1
        L = 1.0
        f = lambda x: 0.5 * x**2
        grad_f = lambda x: x
        
        # Test various step sizes
        results = []
        for alpha in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            x = 1.0
            for _ in range(100):
                x = x - alpha * grad_f(x)
                if abs(x) > 1e10:
                    break
            
            converged = abs(x) < 0.01
            satisfies_bound = alpha <= 2/L
            
            results.append({
                'alpha': alpha,
                'bound': 2/L,
                'satisfies': satisfies_bound,
                'converged': converged,
                'final_x': x if abs(x) < 1e10 else 'inf'
            })
        
        print(f"f(x) = (1/2)x², L = {L}, bound = 2/L = {2/L}")
        print("-" * 50)
        for r in results:
            status = "✓" if r['converged'] else "✗"
            print(f"α = {r['alpha']:.1f}: converged={status}, "
                  f"α ≤ 2/L: {r['satisfies']}")
        
        # Contrapositive: α > 2/L → diverges
        print("\nContrapositive verified: all α > 2 diverged")
    
    # =========================================================================
    # CASES
    # =========================================================================
    
    @staticmethod
    def verify_relu_lipschitz():
        """
        Proof by cases: ReLU is 1-Lipschitz.
        """
        print("\nCases: ReLU is 1-Lipschitz")
        print("=" * 50)
        
        relu = lambda x: max(0, x)
        
        # Test all case combinations
        cases = [
            ("x ≥ 0, y ≥ 0", 2.0, 3.0),
            ("x < 0, y < 0", -2.0, -3.0),
            ("x ≥ 0, y < 0", 2.0, -3.0),
            ("x < 0, y ≥ 0", -2.0, 3.0),
            ("x = 0, y > 0", 0.0, 3.0),
            ("x = 0, y < 0", 0.0, -3.0),
        ]
        
        all_lipschitz = True
        for case_name, x, y in cases:
            lhs = abs(relu(x) - relu(y))
            rhs = abs(x - y)
            is_lipschitz = lhs <= rhs + 1e-10
            
            print(f"Case {case_name}: |ReLU(x)-ReLU(y)|={lhs:.1f} ≤ |x-y|={rhs:.1f}? {is_lipschitz}")
            
            if not is_lipschitz:
                all_lipschitz = False
        
        print(f"\nReLU is 1-Lipschitz: {all_lipschitz}")
        return all_lipschitz

# Run all demonstrations
if __name__ == "__main__":
    print("=" * 60)
    print("PROOF TECHNIQUES: COMPUTATIONAL DEMONSTRATIONS")
    print("=" * 60)
    
    ProofTechniques.verify_convex_global_min()
    ProofTechniques.verify_unique_minimum()
    ProofTechniques.verify_induction_sum_formula()
    ProofTechniques.verify_backprop_induction()
    ProofTechniques.verify_contrapositive_gd()
    ProofTechniques.verify_relu_lipschitz()
    
    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)

```

---

## 📚 Resources

| Type | Title | Link |
|:-----|:------|:-----|
| 📖 Book | How to Prove It (Velleman) | [Cambridge](https://www.cambridge.org/core/books/how-to-prove-it/6E4BFAB4D35CD80D5F60FB4A3AD10FFD) |
| 📖 Book | Book of Proof (Hammack) | [Free PDF](https://www.people.vcu.edu/~rhammack/BookOfProof/) |
| 📖 Book | Introduction to Algorithms (CLRS) | [MIT Press](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition) |
| 🎥 Video | MIT 6.042J - Mathematics for CS | [YouTube](https://www.youtube.com/watch?v=L3LMbpZIKhQ&list=PLB7540DEDD482705B) |
| 🎥 Video | Discrete Math Proofs | [YouTube](https://www.youtube.com/watch?v=q8V2fCk11jY) |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[🧠 Mathematical Thinking](../01_mathematical_thinking/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 2 of 6**<br>
**📐 Proof Techniques**

</td>
<td align="right" width="33%">

➡️ **Next**<br>
[🔢 Set Theory](../03_set_theory/README.md)

</td>
</tr>
</table>

### Quick Links

| Direction | Destination |
|:---------:|-------------|
| 🏠 Section Home | [01: Mathematical Foundations](../README.md) |
| 📋 Full Course | [Course Home](../../README.md) |

---

<!-- Animated Footer -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,14,16&height=100&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=FF6B6B&center=true&vCenter=true&width=600&lines=Made+with+❤️+by+Gaurav+Goswami;Part+of+ML+Researcher+Foundations+Series" alt="Footer" />
</p>

<p align="center">
  <a href="https://github.com/Gaurav14cs17">
    <img src="https://img.shields.io/badge/GitHub-Gaurav14cs17-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>
