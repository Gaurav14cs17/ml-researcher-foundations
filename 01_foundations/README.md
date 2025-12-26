<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Mathematical%20Foundations&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-01-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Topics-6-4ECDC4?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Proofs-Complete-success?style=flat-square" alt="Proofs"/>
  <img src="https://img.shields.io/badge/Examples-50+-informational?style=flat-square" alt="Examples"/>
  <img src="https://img.shields.io/badge/Code-Python%20%7C%20PyTorch-orange?style=flat-square" alt="Code"/>
  <img src="https://img.shields.io/badge/ML_Applications-✓-brightgreen?style=flat-square" alt="ML"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 TL;DR

> **Mathematical foundations provide the rigorous thinking patterns essential for ML research.**

| Foundation | Purpose | ML Application |
|------------|---------|----------------|
| 📐 **Mathematical Thinking** | Abstraction, logical reasoning | Understanding paper proofs |
| 🔢 **Proof Techniques** | Verify claims rigorously | Proving convergence guarantees |
| 📊 **Set Theory** | Foundation for probability | Sample spaces, σ-algebras |
| 🧠 **Logic** | Formal reasoning | Specifications, constraints |
| ⏱️ **Asymptotic Analysis** | Algorithm efficiency | Model scalability |
| 💻 **Numerical Computation** | Floating-point stability | Training stability |

---

## 📚 Table of Contents

| # | Topic | Key Concepts | Time | Link |
|:-:|-------|--------------|:----:|:----:|
| 1 | [Mathematical Thinking](#-1-mathematical-thinking) | Abstraction, Necessary vs Sufficient | 3h | [📁](./01_mathematical_thinking/) |
| 2 | [Proof Techniques](#-2-proof-techniques) | Direct, Contradiction, Induction | 4h | [📁](./02_proof_techniques/) |
| 3 | [Set Theory](#-3-set-theory) | Sets, Functions, Relations, σ-algebras | 3h | [📁](./03_set_theory/) |
| 4 | [Logic](#-4-logic) | Propositional, Predicate, Inference | 3h | [📁](./04_logic/) |
| 5 | [Asymptotic Analysis](#-5-asymptotic-analysis) | Big-O, Ω, Θ, little-o Notation | 3h | [📁](./05_asymptotic_analysis/) |
| 6 | [Numerical Computation](#-6-numerical-computation) | Floating Point, Stability, Mixed Precision | 3h | [📁](./06_numerical_computation/) |

**Total: ~19 hours**

---

## 🗺️ Visual Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          MATHEMATICAL FOUNDATIONS                                │
│                     Building Blocks for ML Research                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│     ┌──────────────────┐         ┌──────────────────┐                           │
│     │  📐 Mathematical │         │  🔢 Proof        │                           │
│     │     Thinking     │────────▶│   Techniques     │                           │
│     │  ┈┈┈┈┈┈┈┈┈┈┈┈┈  │         │  ┈┈┈┈┈┈┈┈┈┈┈┈┈  │                           │
│     │  • Abstraction   │         │  • Direct Proof  │                           │
│     │  • Nec. vs Suff. │         │  • Contradiction │                           │
│     │  • Definitions   │         │  • Induction     │                           │
│     └────────┬─────────┘         └────────┬─────────┘                           │
│              │                            │                                      │
│              ▼                            ▼                                      │
│     ┌──────────────────┐         ┌──────────────────┐                           │
│     │  📊 Set Theory   │         │  🧠 Logic        │                           │
│     │  ┈┈┈┈┈┈┈┈┈┈┈┈┈  │         │  ┈┈┈┈┈┈┈┈┈┈┈┈┈  │                           │
│     │  • Operations    │◀───────▶│  • Propositional │                           │
│     │  • Functions     │         │  • Predicate     │                           │
│     │  • σ-algebras    │         │  • Inference     │                           │
│     └────────┬─────────┘         └────────┬─────────┘                           │
│              │                            │                                      │
│              ▼                            ▼                                      │
│     ┌──────────────────┐         ┌──────────────────┐                           │
│     │  ⏱️ Asymptotic   │         │  💻 Numerical    │                           │
│     │    Analysis      │         │   Computation    │                           │
│     │  ┈┈┈┈┈┈┈┈┈┈┈┈┈  │         │  ┈┈┈┈┈┈┈┈┈┈┈┈┈  │                           │
│     │  • Big-O, Ω, Θ   │         │  • IEEE 754      │                           │
│     │  • Complexity    │         │  • Stability     │                           │
│     │  • Scalability   │         │  • Mixed Prec.   │                           │
│     └────────┬─────────┘         └────────┬─────────┘                           │
│              │                            │                                      │
│              └────────────┬───────────────┘                                      │
│                           ▼                                                      │
│              ┌──────────────────────────────┐                                    │
│              │     🤖 MACHINE LEARNING      │                                    │
│              │  ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈  │                                    │
│              │  Linear Algebra • Calculus   │                                    │
│              │  Probability • Optimization  │                                    │
│              └──────────────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 1. Mathematical Thinking

> **📁 [Detailed Notes →](./01_mathematical_thinking/)**

### Core Concepts

#### 1.1 Abstraction Levels

**Definition:** Abstraction is the process of extracting essential features while ignoring irrelevant details.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ABSTRACTION LEVELS IN ML                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 4   ┌─────────────────────────────────────┐              │
│  (Highest) │  "Train a classifier"               │  ← User      │
│            └─────────────────────────────────────┘              │
│                           │                                      │
│  Level 3   ┌─────────────────────────────────────┐              │
│            │  "Minimize cross-entropy loss"      │  ← ML Eng    │
│            └─────────────────────────────────────┘              │
│                           │                                      │
│  Level 2   ┌─────────────────────────────────────┐              │
│            │  θ ← θ - η∇L(θ)                     │  ← Researcher│
│            └─────────────────────────────────────┘              │
│                           │                                      │
│  Level 1   ┌─────────────────────────────────────┐              │
│            │  for w in weights: w -= lr * dL/dw  │  ← Implement │
│            └─────────────────────────────────────┘              │
│                           │                                      │
│  Level 0   ┌─────────────────────────────────────┐              │
│  (Lowest)  │  Binary floating-point operations   │  ← Hardware  │
│            └─────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

**PyTorch Example:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════════════
# HIGH ABSTRACTION (PyTorch)
# ═══════════════════════════════════════════════════════════════
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
loss = F.cross_entropy(model(x), y)
loss.backward()  # Automatic differentiation!

# ═══════════════════════════════════════════════════════════════
# LOW ABSTRACTION (NumPy - manual everything)
# ═══════════════════════════════════════════════════════════════
import numpy as np

def forward(x, W1, b1, W2, b2):
    z1 = x @ W1 + b1           # Linear
    a1 = np.maximum(0, z1)      # ReLU
    z2 = a1 @ W2 + b2           # Linear
    return z2, z1, a1

def backward(x, y, z2, z1, a1, W1, W2):
    # Softmax + cross-entropy gradient
    exp_z2 = np.exp(z2 - z2.max(axis=1, keepdims=True))
    softmax = exp_z2 / exp_z2.sum(axis=1, keepdims=True)
    
    dz2 = softmax.copy()
    dz2[range(len(y)), y] -= 1
    dz2 /= len(y)
    
    dW2 = a1.T @ dz2
    db2 = dz2.sum(axis=0)
    
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0)  # ReLU derivative
    
    dW1 = x.T @ dz1
    db1 = dz1.sum(axis=0)
    
    return dW1, db1, dW2, db2
```

#### 1.2 Necessary vs Sufficient Conditions

**Formal Definitions:**

| Condition | Notation | Meaning | Example |
|-----------|----------|---------|---------|
| **Necessary** | Q → P | P is required for Q | "Fuel is necessary to drive" |
| **Sufficient** | P → Q | P guarantees Q | "Winning lottery is sufficient to be rich" |
| **Nec. & Suff.** | P ↔ Q | P if and only if Q | "Triangle has 3 sides ↔ Triangle" |

**ML Examples with Analysis:**

| Statement | Nec? | Suff? | Explanation |
|-----------|:----:|:-----:|-------------|
| Convexity → Global optimum | ❌ | ✅ | Non-convex can have global optimum too |
| ∇f(x*) = 0 for minimum | ✅ | ❌ | Could be saddle point or maximum |
| Full rank for unique solution | ✅ | ✅ | Exactly characterizes unique solution |
| Large dataset for good model | ❌ | ❌ | Neither necessary nor sufficient |
| Lipschitz continuity for convergence | ✅ | ❌ | Required but not enough alone |

**Complete Proof: Gradient = 0 is Necessary but Not Sufficient**

```
══════════════════════════════════════════════════════════════════
THEOREM: For differentiable f, ∇f(x*) = 0 is NECESSARY for x* 
         to be a local minimum.
══════════════════════════════════════════════════════════════════

PROOF (by contradiction):
────────────────────────────────────────────────────────────────

1. ASSUME: x* is a local minimum AND ∇f(x*) ≠ 0

2. CONSTRUCT: Let d = -∇f(x*) (negative gradient direction)
   Since ∇f(x*) ≠ 0, we have d ≠ 0

3. TAYLOR EXPANSION: For small ε > 0:
   f(x* + εd) = f(x*) + ε⟨∇f(x*), d⟩ + O(ε²)
              = f(x*) + ε⟨∇f(x*), -∇f(x*)⟩ + O(ε²)
              = f(x*) - ε‖∇f(x*)‖² + O(ε²)

4. ANALYZE: For sufficiently small ε:
   The term -ε‖∇f(x*)‖² < 0 dominates O(ε²)
   Therefore: f(x* + εd) < f(x*)

5. CONTRADICTION: This means we found a point x* + εd 
   with lower function value, contradicting that x* 
   is a local minimum.

6. CONCLUSION: ∇f(x*) = 0 is necessary for local minimum ∎

══════════════════════════════════════════════════════════════════
COUNTEREXAMPLE: ∇f(x*) = 0 is NOT SUFFICIENT
══════════════════════════════════════════════════════════════════

Consider f(x) = x³

∇f(x) = 3x² 
∇f(0) = 0 ✓ (satisfies necessary condition)

But f(x) = x³ has:
- f(-ε) < 0 = f(0) for ε > 0
- f(+ε) > 0 = f(0) for ε > 0

Therefore x = 0 is an INFLECTION POINT, not a minimum!
The gradient being zero is not sufficient. ∎
```

---

## 🔢 2. Proof Techniques

> **📁 [Detailed Notes →](./02_proof_techniques/)**

### Decision Tree for Choosing Proof Technique

```
                    ┌─────────────────────────┐
                    │   What to prove?        │
                    └───────────┬─────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ P → Q         │     │ ∀n ∈ ℕ: P(n)   │     │ ¬∃ or unique    │
│ (implication) │     │ (for all n)    │     │ (impossibility) │
└───────┬───────┘     └────────┬────────┘     └────────┬────────┘
        │                      │                       │
        ▼                      ▼                       ▼
┌───────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ DIRECT PROOF  │     │   INDUCTION     │     │ CONTRADICTION   │
│ Assume P,     │     │ Base + Step     │     │ Assume opposite │
│ derive Q      │     │                 │     │ find conflict   │
└───────────────┘     └─────────────────┘     └─────────────────┘
```

### 2.1 Direct Proof

**Structure:** Assume P → Apply logical steps → Conclude Q

**Example: Gradient Descent Convergence for Convex Functions**

```
══════════════════════════════════════════════════════════════════
THEOREM (Convergence Rate for Smooth Convex Functions):
For L-smooth convex f, gradient descent with step size η = 1/L:
    f(x_k) - f(x*) ≤ ‖x_0 - x*‖²L / (2k)
══════════════════════════════════════════════════════════════════

PROOF:
────────────────────────────────────────────────────────────────

Step 1: L-smoothness inequality
   f(y) ≤ f(x) + ⟨∇f(x), y-x⟩ + (L/2)‖y-x‖²

Step 2: Apply to GD update x_{k+1} = x_k - (1/L)∇f(x_k)
   Let y = x_{k+1} = x_k - (1/L)∇f(x_k)
   
   f(x_{k+1}) ≤ f(x_k) + ⟨∇f(x_k), -(1/L)∇f(x_k)⟩ 
                       + (L/2)‖(1/L)∇f(x_k)‖²
             
             = f(x_k) - (1/L)‖∇f(x_k)‖² + (1/2L)‖∇f(x_k)‖²
             
             = f(x_k) - (1/2L)‖∇f(x_k)‖²

Step 3: Use convexity f(x*) ≥ f(x_k) + ⟨∇f(x_k), x* - x_k⟩
   Rearranging: f(x_k) - f(x*) ≤ ⟨∇f(x_k), x_k - x*⟩
   
   By Cauchy-Schwarz: ≤ ‖∇f(x_k)‖ · ‖x_k - x*‖

Step 4: Combine and telescope
   After k iterations of summing and bounding:
   
   f(x_k) - f(x*) ≤ L‖x_0 - x*‖² / (2k)  ∎
```

### 2.2 Proof by Contradiction

**Structure:** Assume ¬Q → Derive logical contradiction → Conclude Q

**Example: √2 is Irrational**

```
══════════════════════════════════════════════════════════════════
THEOREM: √2 is irrational
══════════════════════════════════════════════════════════════════

PROOF (by contradiction):
────────────────────────────────────────────────────────────────

1. ASSUME (for contradiction): √2 is rational
   Then √2 = p/q where p, q ∈ ℤ, q ≠ 0, gcd(p,q) = 1

2. SQUARE both sides:
   2 = p²/q²
   p² = 2q²

3. ANALYZE p:
   p² is even (since p² = 2q²)
   ⟹ p is even (since odd² = odd)
   Let p = 2m for some integer m

4. SUBSTITUTE:
   (2m)² = 2q²
   4m² = 2q²
   q² = 2m²

5. ANALYZE q:
   q² is even (since q² = 2m²)
   ⟹ q is even

6. CONTRADICTION:
   Both p and q are even
   ⟹ gcd(p,q) ≥ 2
   But we assumed gcd(p,q) = 1  ⚡

7. CONCLUSION: √2 is irrational ∎
```

### 2.3 Mathematical Induction

**Structure:** Base case P(1) + [P(k) → P(k+1)] ⟹ ∀n ≥ 1: P(n)

**Example: Backpropagation Correctness**

```
══════════════════════════════════════════════════════════════════
THEOREM: Backpropagation correctly computes ∂L/∂w for all layers
══════════════════════════════════════════════════════════════════

PROOF (by strong induction on layer depth, from output to input):
────────────────────────────────────────────────────────────────

Setup: L-layer network with activations a_l = σ(z_l), z_l = W_l a_{l-1} + b_l

BASE CASE (Layer L - output layer):
   ∂L/∂W_L = ∂L/∂a_L · ∂a_L/∂z_L · ∂z_L/∂W_L
           = δ_L · a_{L-1}^T
   
   where δ_L = ∂L/∂z_L = ∂L/∂a_L ⊙ σ'(z_L)
   
   This is exactly what backprop computes ✓

INDUCTIVE HYPOTHESIS:
   Assume backprop correctly computes gradients for layers l+1, ..., L

INDUCTIVE STEP (Layer l):
   By chain rule:
   ∂L/∂W_l = ∂L/∂z_l · ∂z_l/∂W_l
           = δ_l · a_{l-1}^T
   
   where δ_l = ∂L/∂z_l = (∂L/∂z_{l+1}) · (∂z_{l+1}/∂a_l) · (∂a_l/∂z_l)
             = (W_{l+1}^T δ_{l+1}) ⊙ σ'(z_l)
   
   By IH, δ_{l+1} is correctly computed
   ⟹ δ_l is correctly computed
   ⟹ ∂L/∂W_l is correctly computed ✓

CONCLUSION: By induction, backprop is correct for all layers ∎
```

**Code Pattern - Induction in Algorithms:**

```python
# Recursive algorithms mirror induction proofs!

def factorial(n: int) -> int:
    """
    Correctness proof by induction:
    - Base: factorial(0) = 1 = 0! ✓
    - Step: factorial(k+1) = (k+1) * factorial(k) 
                           = (k+1) * k!  [by IH]
                           = (k+1)! ✓
    """
    if n == 0:
        return 1  # Base case
    return n * factorial(n - 1)  # Inductive step

def merge_sort(arr: list) -> list:
    """
    Correctness by strong induction on len(arr):
    - Base: len(arr) ≤ 1 → already sorted ✓
    - Step: Assume works for all arrays < k elements
            For array of k elements:
            - Split into two arrays < k elements
            - By IH, both halves sort correctly
            - Merge preserves sorted order ✓
    """
    if len(arr) <= 1:
        return arr  # Base case
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])    # IH applies
    right = merge_sort(arr[mid:])   # IH applies
    return merge(left, right)       # Correct merge
```

---

## 📊 3. Set Theory

> **📁 [Detailed Notes →](./03_set_theory/)**

### 3.1 Set Operations

| Operation | Notation | Definition | Python |
|-----------|----------|------------|--------|
| Union | A ∪ B | {x : x ∈ A or x ∈ B} | `A \| B` |
| Intersection | A ∩ B | {x : x ∈ A and x ∈ B} | `A & B` |
| Difference | A \ B | {x : x ∈ A and x ∉ B} | `A - B` |
| Symmetric Diff | A △ B | (A \ B) ∪ (B \ A) | `A ^ B` |
| Complement | Aᶜ | {x ∈ U : x ∉ A} | `U - A` |
| Power Set | P(A) | {S : S ⊆ A} | `itertools.combinations` |
| Cartesian Product | A × B | {(a,b) : a ∈ A, b ∈ B} | `itertools.product` |

### 3.2 De Morgan's Laws - Complete Proof

```
══════════════════════════════════════════════════════════════════
THEOREM (De Morgan's Laws):
   (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
   (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
══════════════════════════════════════════════════════════════════

PROOF of (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ:
────────────────────────────────────────────────────────────────

(⊆) Let x ∈ (A ∪ B)ᶜ
    ⟹ x ∉ (A ∪ B)           [definition of complement]
    ⟹ ¬(x ∈ A ∨ x ∈ B)      [definition of union]
    ⟹ (x ∉ A) ∧ (x ∉ B)     [De Morgan's law for logic]
    ⟹ x ∈ Aᶜ ∧ x ∈ Bᶜ       [definition of complement]
    ⟹ x ∈ Aᶜ ∩ Bᶜ           [definition of intersection]

(⊇) Let x ∈ Aᶜ ∩ Bᶜ
    ⟹ x ∈ Aᶜ ∧ x ∈ Bᶜ       [definition of intersection]
    ⟹ x ∉ A ∧ x ∉ B         [definition of complement]
    ⟹ ¬(x ∈ A ∨ x ∈ B)      [De Morgan's law for logic]
    ⟹ x ∉ (A ∪ B)           [definition of union]
    ⟹ x ∈ (A ∪ B)ᶜ          [definition of complement]

Both inclusions proven ⟹ (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ ∎
```

### 3.3 Functions - Formal Definitions

| Property | Definition | Test |
|----------|------------|------|
| **Injective** (1-1) | f(a) = f(b) ⟹ a = b | Different inputs → different outputs |
| **Surjective** (onto) | ∀y ∈ Y, ∃x ∈ X: f(x) = y | Every y has a preimage |
| **Bijective** | Injective ∧ Surjective | 1-1 correspondence |

**ML Application - Activation Functions:**

```python
import numpy as np
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
# SIGMOID: (0, 1) - NOT surjective onto ℝ
# ═══════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# Range is (0, 1), not all of ℝ
# NOT injective on extended domain if we consider limits

# ═══════════════════════════════════════════════════════════════
# ReLU: [0, ∞) - NOT injective
# ═══════════════════════════════════════════════════════════════
def relu(x):
    return np.maximum(0, x)
# f(-1) = f(-2) = f(-100) = 0
# Many inputs map to same output!

# ═══════════════════════════════════════════════════════════════
# LEAKY ReLU: ℝ → ℝ - BIJECTIVE!
# ═══════════════════════════════════════════════════════════════
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
# Invertible: different inputs always give different outputs
# Covers all of ℝ as output
```

### 3.4 σ-Algebras (Critical for Probability Theory)

```
══════════════════════════════════════════════════════════════════
DEFINITION: A σ-algebra F on set Ω is a collection of subsets such that:
   1. Ω ∈ F                    (contains the whole space)
   2. A ∈ F ⟹ Aᶜ ∈ F           (closed under complement)  
   3. A₁, A₂, ... ∈ F ⟹ ⋃Aᵢ ∈ F  (closed under countable union)
══════════════════════════════════════════════════════════════════

ML APPLICATION: Probability Space (Ω, F, P)
────────────────────────────────────────────────────────────────

Example: Coin flip
   Ω = {H, T}                      # Sample space
   F = {∅, {H}, {T}, {H,T}}        # σ-algebra (all subsets here)
   P: F → [0,1]                    # Probability measure

Example: Image classification  
   Ω = {all possible 28×28 images} # Sample space
   F = Borel σ-algebra on ℝ^784   # Generated by open sets
   P = data distribution           # Unknown, we sample from it
```

---

## 🧠 4. Logic

> **📁 [Detailed Notes →](./04_logic/)**

### 4.1 Propositional Logic

**Truth Tables:**

| P | Q | P ∧ Q | P ∨ Q | P → Q | P ↔ Q | ¬P |
|:-:|:-:|:-----:|:-----:|:-----:|:-----:|:--:|
| T | T |   T   |   T   |   T   |   T   | F  |
| T | F |   F   |   T   |   F   |   F   | F  |
| F | T |   F   |   T   |   T   |   F   | T  |
| F | F |   F   |   F   |   T   |   T   | T  |

**Key Logical Equivalences:**

```
═══════════════════════════════════════════════════════════════
IMPLICATION EQUIVALENCES:
   P → Q  ≡  ¬P ∨ Q           (definition of implication)
   P → Q  ≡  ¬Q → ¬P          (contrapositive - EQUIVALENT!)
   P → Q  ≢  Q → P            (converse - NOT equivalent!)
   P → Q  ≢  ¬P → ¬Q          (inverse - NOT equivalent!)

DE MORGAN'S LAWS (Logic):
   ¬(P ∧ Q) ≡ ¬P ∨ ¬Q
   ¬(P ∨ Q) ≡ ¬P ∧ ¬Q

DISTRIBUTION:
   P ∧ (Q ∨ R) ≡ (P ∧ Q) ∨ (P ∧ R)
   P ∨ (Q ∧ R) ≡ (P ∨ Q) ∧ (P ∨ R)
═══════════════════════════════════════════════════════════════
```

### 4.2 Predicate Logic & Quantifiers

**Quantifier Negation Rules:**

| Original | Negation |
|----------|----------|
| ∀x P(x) | ∃x ¬P(x) |
| ∃x P(x) | ∀x ¬P(x) |
| ∀x∃y P(x,y) | ∃x∀y ¬P(x,y) |

**ML Example - Convergence Statements:**

```
═══════════════════════════════════════════════════════════════
STATEMENT: "SGD converges for all convex functions"

FORMAL: ∀f: [Convex(f) → Converges(SGD, f)]

NEGATION: "There exists a convex function where SGD doesn't converge"
          ∃f: [Convex(f) ∧ ¬Converges(SGD, f)]

═══════════════════════════════════════════════════════════════
STATEMENT: "For every ε > 0, there exists N such that for all n > N, |aₙ - L| < ε"
           (Definition of limit)

FORMAL: ∀ε > 0, ∃N ∈ ℕ, ∀n > N: |aₙ - L| < ε

NEGATION: ∃ε > 0, ∀N ∈ ℕ, ∃n > N: |aₙ - L| ≥ ε
          "Sequence does NOT converge to L"
═══════════════════════════════════════════════════════════════
```

### 4.3 Rules of Inference

| Rule | Form | Name |
|------|------|------|
| P, P → Q ⊢ Q | If P and P implies Q, then Q | Modus Ponens |
| ¬Q, P → Q ⊢ ¬P | If not Q and P implies Q, then not P | Modus Tollens |
| P → Q, Q → R ⊢ P → R | Chain implications | Hypothetical Syllogism |
| P ∨ Q, ¬P ⊢ Q | If P or Q and not P, then Q | Disjunctive Syllogism |

---

## ⏱️ 5. Asymptotic Analysis

> **📁 [Detailed Notes →](./05_asymptotic_analysis/)**

### 5.1 Formal Definitions

```
═══════════════════════════════════════════════════════════════
BIG-O (Upper Bound):
   f(n) = O(g(n)) ⟺ ∃c > 0, n₀: ∀n ≥ n₀, f(n) ≤ c·g(n)
   
BIG-OMEGA (Lower Bound):
   f(n) = Ω(g(n)) ⟺ ∃c > 0, n₀: ∀n ≥ n₀, f(n) ≥ c·g(n)
   
BIG-THETA (Tight Bound):
   f(n) = Θ(g(n)) ⟺ f(n) = O(g(n)) ∧ f(n) = Ω(g(n))
   
LITTLE-O (Strict Upper Bound):
   f(n) = o(g(n)) ⟺ lim_{n→∞} f(n)/g(n) = 0
═══════════════════════════════════════════════════════════════
```

### 5.2 Complexity Hierarchy

```
Fastest ←──────────────────────────────────────────────→ Slowest

O(1) < O(log n) < O(√n) < O(n) < O(n log n) < O(n²) < O(n³) < O(2ⁿ) < O(n!)
 │        │         │       │        │          │        │        │       │
 │        │         │       │        │          │        │        │       └ Brute force
 │        │         │       │        │          │        │        └ Subset enumeration
 │        │         │       │        │          │        └ Matrix multiplication
 │        │         │       │        │          └ Naive attention O(n²d)
 │        │         │       │        └ Merge sort, FFT
 │        │         │       └ Linear scan
 │        │         └ Meet in the middle
 │        └ Binary search, tree operations
 └ Hash table lookup
```

### 5.3 ML Model Complexities

| Model | Training | Inference | Space |
|-------|----------|-----------|-------|
| Linear Regression | O(nd² + d³) | O(d) | O(d²) |
| k-NN | O(1) | O(nd) | O(nd) |
| Decision Tree | O(nd log n) | O(log n) | O(nodes) |
| Random Forest | O(k·nd log n) | O(k log n) | O(k·nodes) |
| Naive Bayes | O(nd) | O(d) | O(d) |
| SVM (kernel) | O(n²d) - O(n³) | O(sv·d) | O(n²) |
| Transformer | O(n²d) | O(n²d) | O(n² + nd) |
| Flash Attention | O(n²d) | O(n²d) | O(n) memory! |

**Proof: 3n² + 2n + 1 = O(n²)**

```
══════════════════════════════════════════════════════════════════
CLAIM: 3n² + 2n + 1 = O(n²)
══════════════════════════════════════════════════════════════════

PROOF:
We need to find c > 0 and n₀ such that:
   3n² + 2n + 1 ≤ c·n² for all n ≥ n₀

For n ≥ 1:
   • 2n ≤ 2n²    (since n ≤ n² for n ≥ 1)
   • 1 ≤ n²      (since 1 ≤ n² for n ≥ 1)

Therefore:
   3n² + 2n + 1 ≤ 3n² + 2n² + n²
                = 6n²

Choose c = 6, n₀ = 1

Verification: For all n ≥ 1:
   3n² + 2n + 1 ≤ 6n² ✓

Therefore, 3n² + 2n + 1 = O(n²) ∎
```

---

## 💻 6. Numerical Computation

> **📁 [Detailed Notes →](./06_numerical_computation/)**

### 6.1 IEEE 754 Floating Point

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IEEE 754 FLOATING POINT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  32-bit (float32):                                                          │
│  ┌───┬──────────┬───────────────────────────────────┐                       │
│  │ S │ Exponent │           Mantissa                │                       │
│  │ 1 │    8     │             23                    │                       │
│  └───┴──────────┴───────────────────────────────────┘                       │
│                                                                              │
│  64-bit (float64):                                                          │
│  ┌───┬─────────────┬────────────────────────────────────────────────────┐   │
│  │ S │  Exponent   │                    Mantissa                        │   │
│  │ 1 │     11      │                      52                            │   │
│  └───┴─────────────┴────────────────────────────────────────────────────┘   │
│                                                                              │
│  16-bit (float16 / half):                                                   │
│  ┌───┬───────┬────────────────┐                                             │
│  │ S │  Exp  │    Mantissa    │                                             │
│  │ 1 │   5   │       10       │                                             │
│  └───┴───────┴────────────────┘                                             │
│                                                                              │
│  Value = (-1)^S × 2^(Exp - bias) × (1 + Mantissa/2^bits)                    │
│                                                                              │
│  Bias: float16=15, float32=127, float64=1023                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Numbers:**

| Type | Min Positive | Max | Epsilon | Decimal Digits |
|------|-------------|-----|---------|----------------|
| float16 | 6.1e-5 | 6.5e4 | 9.77e-4 | ~3 |
| float32 | 1.2e-38 | 3.4e38 | 1.19e-7 | ~7 |
| float64 | 2.2e-308 | 1.8e308 | 2.22e-16 | ~15 |

### 6.2 Numerical Stability Issues & Solutions

**Problem 1: Softmax Overflow**

```python
import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════
# PROBLEM: Overflow in naive softmax
# ═══════════════════════════════════════════════════════════════
def softmax_unstable(x):
    exp_x = np.exp(x)  # Can overflow for large x!
    return exp_x / np.sum(exp_x)

x = np.array([1000, 1001, 1002])
print(softmax_unstable(x))  # [nan, nan, nan] - BROKEN!

# ═══════════════════════════════════════════════════════════════
# SOLUTION: Subtract max (mathematically equivalent)
# ═══════════════════════════════════════════════════════════════
def softmax_stable(x):
    x_shifted = x - np.max(x)  # Shift to prevent overflow
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

print(softmax_stable(x))  # [0.09, 0.24, 0.67] - Correct!

# PROOF OF EQUIVALENCE:
# exp(x - max) / Σexp(x - max) = exp(x)/exp(max) / (Σexp(x)/exp(max))
#                              = exp(x) / Σexp(x) ✓
```

**Problem 2: Log-Sum-Exp Underflow**

```python
# ═══════════════════════════════════════════════════════════════
# PROBLEM: Underflow in naive log-sum-exp
# ═══════════════════════════════════════════════════════════════
def logsumexp_unstable(x):
    return np.log(np.sum(np.exp(x)))

x = np.array([-1000, -1001, -1002])
print(logsumexp_unstable(x))  # -inf (underflow!)

# ═══════════════════════════════════════════════════════════════
# SOLUTION: Factor out max
# ═══════════════════════════════════════════════════════════════
def logsumexp_stable(x):
    c = np.max(x)
    return c + np.log(np.sum(np.exp(x - c)))

print(logsumexp_stable(x))  # -999.59 - Correct!

# PROOF:
# log(Σexp(xᵢ)) = log(Σexp(xᵢ - c + c))
#               = log(exp(c) · Σexp(xᵢ - c))
#               = c + log(Σexp(xᵢ - c)) ✓
```

**Problem 3: Catastrophic Cancellation**

```python
# ═══════════════════════════════════════════════════════════════
# PROBLEM: Loss of precision in variance calculation
# ═══════════════════════════════════════════════════════════════
def variance_unstable(x):
    """Two-pass formula: Var = E[X²] - E[X]²"""
    n = len(x)
    mean_sq = np.sum(x**2) / n
    sq_mean = (np.sum(x) / n)**2
    return mean_sq - sq_mean  # Subtracting similar large numbers!

# ═══════════════════════════════════════════════════════════════
# SOLUTION: Welford's online algorithm
# ═══════════════════════════════════════════════════════════════
def variance_welford(x):
    """Single-pass numerically stable algorithm"""
    n = 0
    mean = 0
    M2 = 0
    
    for xi in x:
        n += 1
        delta = xi - mean
        mean += delta / n
        delta2 = xi - mean
        M2 += delta * delta2
    
    return M2 / n if n > 0 else 0
```

### 6.3 Mixed Precision Training

```python
import torch
from torch.cuda.amp import autocast, GradScaler

# ═══════════════════════════════════════════════════════════════
# MIXED PRECISION TRAINING
# Forward: FP16 (fast, memory efficient)
# Master weights & gradients: FP32 (precision)
# ═══════════════════════════════════════════════════════════════

model = MyModel().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = GradScaler()  # Scales gradients to prevent FP16 underflow

for batch_idx, (data, target) in enumerate(dataloader):
    data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(data)
        loss = F.cross_entropy(output, target)
    
    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    
    # Unscale gradients, clip, then step
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

---

## 📝 Key Formulas Summary

| Category | Formula | Description |
|----------|---------|-------------|
| **Induction** | P(1) ∧ [P(k)→P(k+1)] ⟹ ∀n P(n) | Mathematical induction principle |
| **De Morgan (Sets)** | (A∪B)ᶜ = Aᶜ∩Bᶜ | Complement of union |
| **De Morgan (Logic)** | ¬(P∧Q) ≡ ¬P∨¬Q | Negation of conjunction |
| **Contrapositive** | (P→Q) ≡ (¬Q→¬P) | Equivalent form of implication |
| **Big-O** | f = O(g) ⟺ ∃c,n₀: f(n) ≤ cg(n) ∀n≥n₀ | Asymptotic upper bound |
| **Big-Θ** | f = Θ(g) ⟺ f = O(g) ∧ f = Ω(g) | Asymptotic tight bound |
| **Softmax** | σ(x)ᵢ = exp(xᵢ)/Σⱼexp(xⱼ) | Probability distribution |
| **Z-score** | z = (x-μ)/σ | Standardization |
| **Condition Number** | κ(A) = ‖A‖·‖A⁻¹‖ | Sensitivity to perturbation |
| **Machine Epsilon** | ε = min{x > 0 : 1 + x ≠ 1} | Smallest representable difference |

---

## ⚠️ Common Mistakes

| Mistake | Correct Understanding |
|---------|----------------------|
| "∇f=0 means minimum" | Only necessary, not sufficient (could be saddle point) |
| "P→Q means P causes Q" | Implication ≠ causation |
| "Converse equals original" | P→Q ≢ Q→P |
| "O(n²) is always slow" | Depends on constants and n; O(n²) with small c beats O(n) with huge c |
| "float == for equality" | Use `abs(a-b) < ε` for floating point comparison |
| "More data = better model" | Diminishing returns; quality and diversity matter |
| "FP16 always works" | Need gradient scaling; some ops need FP32 |

---

## 📚 Resources

### Books
| Title | Author | Topic |
|-------|--------|-------|
| How to Prove It | Velleman | Proof techniques |
| Discrete Mathematics and Its Applications | Rosen | Logic, sets, proofs |
| Introduction to Algorithms (CLRS) | Cormen et al. | Complexity analysis |
| Numerical Linear Algebra | Trefethen & Bau | Numerical stability |

### Online Courses
| Course | Platform | Link |
|--------|----------|------|
| MIT 6.042J Mathematics for CS | MIT OCW | [Link](https://ocw.mit.edu/6-042J) |
| Introduction to Proofs | Coursera | [Link](https://www.coursera.org/learn/proofs) |
| 3Blue1Brown Essence Series | YouTube | [Link](https://youtube.com/c/3blue1brown) |

---

## 🧭 Navigation

### Section Navigation (Within This Series)

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[🏠 Main Course](../README.md)

</td>
<td align="center" width="34%">

📍 **Current: Section 1**<br>
**📐 Mathematical Foundations**

</td>
<td align="right" width="33%">

➡️ **Next Section**<br>
[📊 02: Mathematics](../02_mathematics/README.md)

</td>
</tr>
</table>

### Sub-Topics in This Section

| # | Topic | Direct Link |
|:-:|-------|:-----------:|
| 1 | Mathematical Thinking | [📁 Open](./01_mathematical_thinking/README.md) |
| 2 | Proof Techniques | [📁 Open](./02_proof_techniques/README.md) |
| 3 | Set Theory | [📁 Open](./03_set_theory/README.md) |
| 4 | Logic | [📁 Open](./04_logic/README.md) |
| 5 | Asymptotic Analysis | [📁 Open](./05_asymptotic_analysis/README.md) |
| 6 | Numerical Computation | [📁 Open](./06_numerical_computation/README.md) |

### Full Course Navigation

| Section | Topic | Link |
|:-------:|-------|:----:|
| 01 | **Mathematical Foundations** ← You are here | — |
| 02 | Mathematics (Linear Algebra, Calculus) | [Go →](../02_mathematics/README.md) |
| 03 | Probability & Statistics | [Go →](../03_probability_statistics/README.md) |
| 04 | Optimization | [Go →](../04_optimization/README.md) |
| 05 | ML Theory | [Go →](../05_ml_theory/README.md) |
| 06 | Deep Learning | [Go →](../06_deep_learning/README.md) |
| 07 | Reinforcement Learning | [Go →](../07_reinforcement_learning/README.md) |
| 08 | Model Compression | [Go →](../08_model_compression/README.md) |
| 09 | Efficient ML | [Go →](../09_efficient_ml/README.md) |

---

## 📁 Section Structure

```
01_foundations/
├── README.md                      ← You are here (Overview)
│
├── 01_mathematical_thinking/
│   ├── README.md                  # Abstraction, necessary vs sufficient
│   └── images/
│
├── 02_proof_techniques/
│   ├── README.md                  # Direct, contradiction, induction
│   └── images/
│
├── 03_set_theory/
│   ├── README.md                  # Sets, functions, relations, σ-algebras
│   └── images/
│
├── 04_logic/
│   ├── README.md                  # Propositional, predicate, inference
│   └── images/
│
├── 05_asymptotic_analysis/
│   ├── README.md                  # Big-O, Ω, Θ, little-o analysis
│   └── images/
│
├── 06_numerical_computation/
│   ├── README.md                  # Floating point, stability, mixed precision
│   └── images/
│
└── images/                        # Shared images
```

---

## ✅ Learning Checklist

- [ ] Can identify necessary vs sufficient conditions in ML contexts
- [ ] Can construct direct proofs, proofs by contradiction, and induction
- [ ] Understand set operations and their notation
- [ ] Can negate quantified statements correctly
- [ ] Can analyze algorithm complexity using Big-O notation
- [ ] Understand floating-point representation and common pitfalls
- [ ] Can implement numerically stable versions of common operations

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>
</p>
