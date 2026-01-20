<!-- Animated Header -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=FFD93D&center=true&vCenter=true&width=800&lines=🔀+Mathematical+Logic;The+Language+of+Reasoning" alt="Logic" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,3,4&height=180&section=header&text=Mathematical%20Logic&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Propositional%20•%20Predicate%20•%20Inference%20•%20Boolean%20Functions&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04_of_06-FFD93D?style=for-the-badge&logo=bookstack&logoColor=black" alt="Section"/>
  <img src="https://img.shields.io/badge/Topics-8_Concepts-FF6B6B?style=for-the-badge&logo=buffer&logoColor=white" alt="Topics"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-00D4AA?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-6C63FF?style=for-the-badge&logo=calendar&logoColor=white" alt="Updated"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Difficulty-Intermediate-orange?style=flat-square" alt="Difficulty"/>
  <img src="https://img.shields.io/badge/Reading_Time-55_minutes-blue?style=flat-square" alt="Reading Time"/>
  <img src="https://img.shields.io/badge/Prerequisites-Set_Theory-green?style=flat-square" alt="Prerequisites"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01_mathematical_thinking/README.md) → [Proof Techniques](../02_proof_techniques/README.md) → [Set Theory](../03_set_theory/README.md) → Logic → [Asymptotic Analysis](../05_asymptotic_analysis/README.md) → [Numerical Computation](../06_numerical_computation/README.md)

---

## 📌 TL;DR

Logic is the **foundation of all mathematical reasoning** and computation. This article covers:

- **Propositional Logic** — AND, OR, NOT, IMPLIES with truth tables
- **Predicate Logic** — ∀ (for all) and ∃ (exists) quantifiers
- **Rules of Inference** — Modus ponens, modus tollens, syllogisms
- **Boolean Functions** — How neural networks compute logic
- **De Morgan's Laws** — Negation rules for complex expressions

> [!WARNING]
> **Quantifier order matters!** `∀x ∃y P(x,y)` ≠ `∃y ∀x P(x,y)`. This is a common source of confusion in ML papers.

---

## 📚 What You'll Learn

By the end of this article, you will be able to:

- [ ] Read and construct truth tables
- [ ] Understand quantifiers (∀, ∃) in ML theorems
- [ ] Apply De Morgan's laws for negation
- [ ] Use inference rules in proofs
- [ ] Connect boolean logic to neural networks
- [ ] Recognize logical patterns in ML papers

---

## 📑 Table of Contents

1. [Visual Overview](#-visual-overview)
2. [Propositional Logic](#1-propositional-logic)
3. [Logical Equivalences](#2-logical-equivalences)
4. [Predicate Logic](#3-predicate-logic)
5. [Rules of Inference](#4-rules-of-inference)
6. [Boolean Functions & Neural Networks](#5-boolean-functions--neural-networks)
7. [Logic in ML Papers](#6-logic-in-ml-papers)
8. [Key Formulas Summary](#-key-formulas-summary)
9. [Common Mistakes & Pitfalls](#-common-mistakes--pitfalls)
10. [Code Implementations](#-code-implementations)
11. [Resources](#-resources)
12. [Navigation](#-navigation)

---

## 🎯 Visual Overview

### Logical Connectives

```
+-----------------------------------------------------------------------------+
|                         LOGICAL CONNECTIVES                                 |
+-----------------------------------------------------------------------------+
|                                                                             |
|   ∧ AND          ∨ OR           ¬ NOT          → IMPLIES       ↔ IFF      |
|   -----          ----           -----          ---------       -----       |
|   P Q P∧Q        P Q P∨Q        P ¬P           P Q P→Q         P Q P↔Q     |
|   T T  T         T T  T         T  F           T T  T          T T  T      |
|   T F  F         T F  T         F  T           T F  F          T F  F      |
|   F T  F         F T  T                        F T  T          F T  F      |
|   F F  F         F F  F                        F F  T          F F  T      |
|                                                                             |
|   "Both true"    "At least      "Opposite"     "If P then Q"   "P iff Q"   |
|                   one true"                     ≡ ¬P ∨ Q                   |
|                                                                             |
+-----------------------------------------------------------------------------+
```

### Quantifiers

```
+-----------------------------------------------------------------------------+
|                              QUANTIFIERS                                    |
+-----------------------------------------------------------------------------+
|                                                                             |
|   ∀ UNIVERSAL                           ∃ EXISTENTIAL                      |
|   ------------                          --------------                      |
|   "For all x, P(x)"                     "There exists x such that P(x)"    |
|                                                                             |
|   ∀x ∈ ℕ: x ≥ 0                         ∃x ∈ ℕ: x > 100                    |
|   "Every natural number is ≥ 0"         "Some natural number is > 100"     |
|                                                                             |
|   Negation:                             Negation:                           |
|   ¬(∀x: P(x)) ≡ ∃x: ¬P(x)              ¬(∃x: P(x)) ≡ ∀x: ¬P(x)            |
|   "Not all" = "Some don't"              "None exist" = "All don't"         |
|                                                                             |
|   ⚠️ ORDER MATTERS!                                                        |
|   ∀x ∃y: P(x,y) ≠ ∃y ∀x: P(x,y)                                           |
|   "For each x, exists y"  vs  "One y works for all x"                      |
|                                                                             |
+-----------------------------------------------------------------------------+
```

---

## 1. Propositional Logic

### 📖 Definition

> **Propositional Logic** deals with propositions (statements that are either true or false) and logical connectives.

### 📖 Connectives and Truth Tables

| Connective | Symbol | Name | Read As |
|:-----------|:------:|:-----|:--------|
| Negation | ¬P | NOT | "not P" |
| Conjunction | P ∧ Q | AND | "P and Q" |
| Disjunction | P ∨ Q | OR | "P or Q" |
| Implication | P → Q | IMPLIES | "if P then Q" |
| Biconditional | P ↔ Q | IFF | "P if and only if Q" |
| Exclusive Or | P ⊕ Q | XOR | "P or Q but not both" |

### 📐 Complete Truth Tables

**Basic Connectives:**

| P | Q | ¬P | P ∧ Q | P ∨ Q | P → Q | P ↔ Q | P ⊕ Q |
|:-:|:-:|:--:|:-----:|:-----:|:-----:|:-----:|:-----:|
| T | T | F | T | T | T | T | F |
| T | F | F | F | T | F | F | T |
| F | T | T | F | T | T | F | T |
| F | F | T | F | F | T | T | F |

### 📐 Proof: P → Q ≡ ¬P ∨ Q

| P | Q | P → Q | ¬P | ¬P ∨ Q | Equal? |
|:-:|:-:|:-----:|:--:|:------:|:------:|
| T | T | T | F | T | ✅ |
| T | F | F | F | F | ✅ |
| F | T | T | T | T | ✅ |
| F | F | T | T | T | ✅ |

**All rows match, therefore P → Q ≡ ¬P ∨ Q** ∎

### 📝 Examples

#### Example 1: Understanding Implication (Simple)

```
P = "It is raining"
Q = "The ground is wet"

P → Q = "If it is raining, then the ground is wet"

When is P → Q FALSE?
Only when P is TRUE and Q is FALSE.
(Raining, but ground NOT wet → claim is false)

Key insight: P → Q is TRUE when P is FALSE!
(If it's not raining, the implication is "vacuously true")
```

#### Example 2: ML Theorem as Logic (Intermediate)

```
Theorem: "If f is convex, then every local minimum is global"

P = "f is convex"
Q = "every local minimum is global"

P → Q (the theorem)

Contrapositive (logically equivalent):
¬Q → ¬P
"If some local minimum is not global, then f is not convex"
```

### 💻 Code Implementation

```python
import numpy as np
from itertools import product

def truth_table(connective, name):
    """Generate truth table for binary connective."""
    print(f"\n{name} Truth Table:")
    print("P | Q | Result")
    print("-" * 15)
    for P, Q in product([True, False], repeat=2):
        result = connective(P, Q)
        print(f"{'T' if P else 'F'} | {'T' if Q else 'F'} | {'T' if result else 'F'}")

# Define connectives
AND = lambda P, Q: P and Q
OR = lambda P, Q: P or Q
IMPLIES = lambda P, Q: not P or Q
IFF = lambda P, Q: P == Q
XOR = lambda P, Q: P != Q

truth_table(AND, "AND (∧)")
truth_table(OR, "OR (∨)")
truth_table(IMPLIES, "IMPLIES (→)")
truth_table(IFF, "IFF (↔)")
truth_table(XOR, "XOR (⊕)")

# Verify P → Q ≡ ¬P ∨ Q
print("\nVerifying P → Q ≡ ¬P ∨ Q:")
for P, Q in product([True, False], repeat=2):
    lhs = IMPLIES(P, Q)
    rhs = (not P) or Q
    print(f"P={P}, Q={Q}: P→Q = {lhs}, ¬P∨Q = {rhs}, Equal: {lhs == rhs}")
```

---

## 2. Logical Equivalences

### 📖 De Morgan's Laws

| Law | Formula | In Words |
|:----|:--------|:---------|
| **Negation of AND** | ¬(P ∧ Q) ≡ ¬P ∨ ¬Q | "Not both" = "At least one false" |
| **Negation of OR** | ¬(P ∨ Q) ≡ ¬P ∧ ¬Q | "Not either" = "Both false" |

### 📐 Proof: De Morgan's Law ¬(P ∧ Q) ≡ ¬P ∨ ¬Q

| P | Q | P ∧ Q | ¬(P ∧ Q) | ¬P | ¬Q | ¬P ∨ ¬Q | Equal? |
|:-:|:-:|:-----:|:--------:|:--:|:--:|:-------:|:------:|
| T | T | T | F | F | F | F | ✅ |
| T | F | F | T | F | T | T | ✅ |
| F | T | F | T | T | F | T | ✅ |
| F | F | F | T | T | T | T | ✅ |

**All rows match, therefore ¬(P ∧ Q) ≡ ¬P ∨ ¬Q** ∎

### 📖 Important Equivalences

| Name | Equivalence |
|:-----|:------------|
| **Implication** | P → Q ≡ ¬P ∨ Q |
| **Contrapositive** | P → Q ≡ ¬Q → ¬P |
| **Biconditional** | P ↔ Q ≡ (P → Q) ∧ (Q → P) |
| **Negation of Implication** | ¬(P → Q) ≡ P ∧ ¬Q |
| **Double Negation** | ¬¬P ≡ P |
| **Idempotence** | P ∧ P ≡ P, P ∨ P ≡ P |
| **Absorption** | P ∧ (P ∨ Q) ≡ P |
| **Distributive** | P ∧ (Q ∨ R) ≡ (P ∧ Q) ∨ (P ∧ R) |

### 📝 ML Application: Negating Complex Conditions

```
Original: "All models with VC dimension < n generalize well"
∀M: (VC(M) < n) → Generalizes(M)

Negation: ¬[∀M: (VC(M) < n) → Generalizes(M)]
        = ∃M: ¬[(VC(M) < n) → Generalizes(M)]
        = ∃M: (VC(M) < n) ∧ ¬Generalizes(M)

"There exists a model with VC < n that doesn't generalize"
```

---

## 3. Predicate Logic

### 📖 Definition

> **Predicate Logic** (First-Order Logic) extends propositional logic with:
> - **Predicates:** P(x) — statements about objects
> - **Quantifiers:** ∀ (for all), ∃ (there exists)
> - **Variables:** x, y, z that range over a domain

### 📖 Quantifier Definitions

| Quantifier | Symbol | Meaning | Example |
|:-----------|:------:|:--------|:--------|
| **Universal** | ∀x: P(x) | P(x) is true for ALL x | ∀x ∈ ℝ: x² ≥ 0 |
| **Existential** | ∃x: P(x) | P(x) is true for SOME x | ∃x ∈ ℕ: x > 100 |
| **Unique** | ∃!x: P(x) | P(x) is true for EXACTLY ONE x | ∃!x: x + 1 = 2 |

### 📖 Quantifier Negation Rules

| Original | Negation | Intuition |
|:---------|:---------|:----------|
| ∀x: P(x) | ∃x: ¬P(x) | "Not all" = "Some don't" |
| ∃x: P(x) | ∀x: ¬P(x) | "None exist" = "All don't" |

### 📐 Proof: Negation Rule ¬(∀x: P(x)) ≡ ∃x: ¬P(x)

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | ∀x: P(x) means P(x) is true for every x | Definition |
| 2 | ¬(∀x: P(x)) means "it's false that P(x) holds for all x" | Negation |
| 3 | This means there is some x where P(x) fails | Logical meaning |
| 4 | Which is exactly ∃x: ¬P(x) | Definition |
| 5 | Therefore ¬(∀x: P(x)) ≡ ∃x: ¬P(x) | ∎ |

### 📝 Examples

#### Example 1: ML Theorem with Quantifiers (Intermediate)

```
Universal Approximation Theorem:

∀ε > 0, ∀f continuous, ∃NN: ‖NN - f‖ < ε

Translation:
"For any error tolerance ε > 0,
 For any continuous function f,
 There exists a neural network NN,
 Such that NN approximates f within ε"
```

#### Example 2: Quantifier Order Matters! (Critical)

```
Statement 1: ∀x ∃y: P(x, y)
"For each x, there exists some y (depending on x)"
Example: "Everyone has a mother"
(Each person has THEIR OWN mother)

Statement 2: ∃y ∀x: P(x, y)
"There exists one y that works for all x"
Example: "There is someone who is everyone's mother"
(ONE person is mother of ALL - much stronger!)

These are NOT equivalent!
Statement 2 implies Statement 1, but not vice versa.
```

#### Example 3: Limit Definition (Advanced)

```
lim(x→a) f(x) = L  is defined as:

∀ε > 0, ∃δ > 0: (0 < |x - a| < δ) → (|f(x) - L| < ε)

Reading: 
"For any ε (how close we want),
 there exists δ (how close x must be to a),
 such that if x is within δ of a (but not equal),
 then f(x) is within ε of L"
```

### 💻 Code Implementation

```python
import numpy as np

def demonstrate_quantifiers():
    """Demonstrate universal and existential quantifiers."""
    
    domain = list(range(1, 11))  # {1, 2, ..., 10}
    
    # ∀x ∈ domain: x > 0
    P1 = lambda x: x > 0
    all_positive = all(P1(x) for x in domain)
    print(f"∀x ∈ {{1,...,10}}: x > 0 → {all_positive}")
    
    # ∃x ∈ domain: x > 5
    P2 = lambda x: x > 5
    exists_greater_5 = any(P2(x) for x in domain)
    print(f"∃x ∈ {{1,...,10}}: x > 5 → {exists_greater_5}")
    
    # Negation: ¬(∀x: P(x)) ≡ ∃x: ¬P(x)
    P3 = lambda x: x < 5
    all_less_5 = all(P3(x) for x in domain)
    exists_not_less_5 = any(not P3(x) for x in domain)
    print(f"∀x: x < 5 → {all_less_5}")
    print(f"¬(∀x: x < 5) ≡ ∃x: x ≥ 5 → {exists_not_less_5}")

def demonstrate_quantifier_order():
    """Show that quantifier order matters."""
    
    # P(x, y) = "y is greater than x"
    P = lambda x, y: y > x
    
    domain = [1, 2, 3]
    
    # ∀x ∃y: y > x
    # "For each x, there exists y greater than x"
    statement1 = all(any(P(x, y) for y in domain) for x in domain)
    
    # ∃y ∀x: y > x  
    # "There exists y greater than all x"
    statement2 = any(all(P(x, y) for x in domain) for y in domain)
    
    print(f"\nDomain: {domain}")
    print(f"∀x ∃y: y > x → {statement1}")  # True (for each x, take y = x+1)
    print(f"∃y ∀x: y > x → {statement2}")  # False (no single y > all x in domain)
    print(f"Order matters: {statement1 != statement2}")

demonstrate_quantifiers()
demonstrate_quantifier_order()
```

---

## 4. Rules of Inference

### 📖 Definition

> **Rules of Inference** are valid argument forms that allow us to derive conclusions from premises.

### 📖 Major Rules

| Rule | Pattern | Name |
|:-----|:--------|:-----|
| **Modus Ponens** | P → Q, P ⊢ Q | "Affirming the antecedent" |
| **Modus Tollens** | P → Q, ¬Q ⊢ ¬P | "Denying the consequent" |
| **Hypothetical Syllogism** | P → Q, Q → R ⊢ P → R | "Chain rule" |
| **Disjunctive Syllogism** | P ∨ Q, ¬P ⊢ Q | "Elimination" |
| **Conjunction** | P, Q ⊢ P ∧ Q | "Introduction" |
| **Simplification** | P ∧ Q ⊢ P | "Elimination" |
| **Addition** | P ⊢ P ∨ Q | "Introduction" |

### 📐 Proof: Modus Ponens is Valid

**To show:** {P → Q, P} ⊨ Q (whenever premises are true, conclusion is true)

| P | Q | P → Q | Premises True? | Q |
|:-:|:-:|:-----:|:--------------:|:-:|
| T | T | T | ✅ (P=T, P→Q=T) | T ✅ |
| T | F | F | ❌ (P→Q=F) | - |
| F | T | T | ❌ (P=F) | - |
| F | F | T | ❌ (P=F) | - |

**The only row where both premises are true, Q is also true.** ∎

### 📐 Proof: Modus Tollens is Valid

**To show:** {P → Q, ¬Q} ⊨ ¬P

| P | Q | P → Q | ¬Q | Premises True? | ¬P |
|:-:|:-:|:-----:|:--:|:--------------:|:--:|
| T | T | T | F | ❌ | - |
| T | F | F | T | ❌ | - |
| F | T | T | F | ❌ | - |
| F | F | T | T | ✅ | T ✅ |

**The only row where both premises are true, ¬P is also true.** ∎

### 📝 ML Examples

| Rule | ML Application |
|:-----|:---------------|
| **Modus Ponens** | "If convex, then global min. f is convex. ∴ f has global min." |
| **Modus Tollens** | "If LR too high, diverges. Didn't diverge. ∴ LR not too high." |
| **Hypothetical Syllogism** | "If data clean → good model. If good model → deploy. ∴ If data clean → deploy." |
| **Disjunctive Syllogism** | "Underfitting OR overfitting. Not underfitting. ∴ Overfitting." |

### 💻 Code Implementation

```python
def modus_ponens(p_implies_q: bool, p: bool) -> bool:
    """
    Modus Ponens: P → Q, P ⊢ Q
    
    If we know P → Q is true and P is true, then Q must be true.
    """
    if p_implies_q and p:
        return True  # Q is definitely true
    return None  # Cannot conclude

def modus_tollens(p_implies_q: bool, not_q: bool) -> bool:
    """
    Modus Tollens: P → Q, ¬Q ⊢ ¬P
    
    If we know P → Q is true and Q is false, then P must be false.
    """
    if p_implies_q and not_q:
        return True  # ¬P is definitely true (P is false)
    return None

def hypothetical_syllogism(p_implies_q: bool, q_implies_r: bool) -> bool:
    """
    Hypothetical Syllogism: P → Q, Q → R ⊢ P → R
    """
    if p_implies_q and q_implies_r:
        return True  # P → R is valid
    return None

# ML Example
print("ML Example: Gradient Descent Convergence")
print("=" * 50)

# P = "Learning rate is small enough"
# Q = "GD converges"
p_implies_q = True  # Theorem: small LR → convergence

# Scenario 1: Small LR (P is true)
p = True
q = modus_ponens(p_implies_q, p)
print(f"Small LR ({p}) + theorem → Converges: {q}")

# Scenario 2: Didn't converge (¬Q is true)
not_q = True
not_p = modus_tollens(p_implies_q, not_q)
print(f"Didn't converge ({not_q}) + theorem → LR not small: {not_p}")
```

---

## 5. Boolean Functions & Neural Networks

### 📖 Boolean Functions as Neural Networks

Every boolean function can be computed by a neural network!

```
AND Gate:                  OR Gate:                   NOT Gate:
x₁ --+                     x₁ --+                     x ----------
     +--▶ σ(x₁+x₂-1.5)          +--▶ σ(x₁+x₂-0.5)           ↓
x₂ --+                     x₂ --+                     σ(-x+0.5)

w = [1, 1]                 w = [1, 1]                 w = [-1]
b = -1.5                   b = -0.5                   b = 0.5

σ(z) = 1 if z ≥ 0 else 0  (step function)
```

### 📐 Proof: AND Gate Works

| x₁ | x₂ | z = x₁ + x₂ - 1.5 | σ(z) | x₁ ∧ x₂ |
|:--:|:--:|:-----------------:|:----:|:-------:|
| 0 | 0 | -1.5 | 0 | 0 ✅ |
| 0 | 1 | -0.5 | 0 | 0 ✅ |
| 1 | 0 | -0.5 | 0 | 0 ✅ |
| 1 | 1 | 0.5 | 1 | 1 ✅ |

**Neural network computes AND correctly!** ∎

### 📖 XOR Problem (Famous Limitation)

```
XOR(x₁, x₂) = (x₁ ∧ ¬x₂) ∨ (¬x₁ ∧ x₂)

Truth table:
x₁  x₂  XOR
0   0    0
0   1    1
1   0    1
1   1    0

XOR is NOT linearly separable!
Single perceptron CANNOT learn XOR.
This was the famous Minsky-Papert criticism (1969).

Solution: Use 2 layers!
Layer 1: Compute OR and NAND
Layer 2: Compute AND of Layer 1 outputs

XOR = (x₁ ∨ x₂) ∧ ¬(x₁ ∧ x₂)
    = OR(x₁, x₂) ∧ NAND(x₁, x₂)
```

### 📐 Proof: XOR Requires 2 Layers

**Theorem:** XOR cannot be computed by a single-layer perceptron.

**Proof:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Single perceptron computes: σ(w₁x₁ + w₂x₂ + b) | Definition |
| 2 | Output is determined by sign of w₁x₁ + w₂x₂ + b | Step function |
| 3 | Decision boundary is a LINE: w₁x₁ + w₂x₂ + b = 0 | Geometry |
| 4 | XOR outputs: (0,0)→0, (0,1)→1, (1,0)→1, (1,1)→0 | Truth table |
| 5 | Class 0: {(0,0), (1,1)}, Class 1: {(0,1), (1,0)} | Grouping |
| 6 | No line separates {(0,0), (1,1)} from {(0,1), (1,0)} | Geometry |
| 7 | XOR is not linearly separable | Definition |
| 8 | Single perceptron cannot compute XOR | ∎ |

### 💻 Code Implementation

```python
import numpy as np

def step(z):
    """Step activation function."""
    return (z >= 0).astype(float)

def create_gate(w, b, name):
    """Create a logic gate as neural network."""
    def gate(*inputs):
        x = np.array(inputs)
        z = np.dot(w, x) + b
        return step(z)
    gate.__name__ = name
    return gate

# Create gates
AND = create_gate([1, 1], -1.5, "AND")
OR = create_gate([1, 1], -0.5, "OR")
NAND = create_gate([-1, -1], 1.5, "NAND")
NOT = lambda x: step(-x + 0.5)

# XOR using 2 layers
def XOR(x1, x2):
    """XOR = (x1 OR x2) AND (x1 NAND x2)"""
    layer1_or = OR(x1, x2)
    layer1_nand = NAND(x1, x2)
    output = AND(layer1_or, layer1_nand)
    return output

# Test XOR
print("XOR Truth Table (2-layer network):")
print("x1 | x2 | XOR")
print("-" * 15)
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    result = XOR(x1, x2)
    print(f" {x1} |  {x2} |  {int(result)}")

# Verify cannot learn with single layer
print("\nAttempting single-layer XOR (will fail):")
# No w1, w2, b can satisfy all 4 conditions simultaneously
```

---

## 6. Logic in ML Papers

### Reading ML Theorems

When you see a theorem in an ML paper:

```
Theorem: Let f: ℝⁿ → ℝ be L-Lipschitz and convex. 
Then gradient descent with step size α ≤ 1/L satisfies:
    f(x_T) - f(x*) ≤ O(1/T)
```

**Parse it as:**

```
∀f: (Lipschitz(f, L) ∧ Convex(f)) →
    (∀α ≤ 1/L: Convergence(GD(f, α), O(1/T)))

Assumptions (Antecedents):
- f is L-Lipschitz
- f is convex  
- α ≤ 1/L

Conclusion (Consequent):
- f(x_T) - f(x*) ≤ O(1/T)
```

### Common Logical Patterns in ML

| Pattern | Example | Logical Form |
|:--------|:--------|:-------------|
| **Sufficient condition** | "Convexity guarantees global opt" | Convex → GlobalOpt |
| **Necessary condition** | "Convergence requires differentiability" | Converge → Diff |
| **Universal bound** | "For all inputs, loss bounded" | ∀x: Loss(x) ≤ M |
| **Existence** | "Optimal solution exists" | ∃θ*: Optimal(θ*) |
| **Uniqueness** | "Unique optimum" | ∃!θ*: Optimal(θ*) |

---

## 📊 Key Formulas Summary

| Concept | Formula | Notes |
|:--------|:--------|:------|
| **Implication** | P → Q ≡ ¬P ∨ Q | Definition |
| **Contrapositive** | P → Q ≡ ¬Q → ¬P | Equivalent |
| **De Morgan 1** | ¬(P ∧ Q) ≡ ¬P ∨ ¬Q | Negate AND |
| **De Morgan 2** | ¬(P ∨ Q) ≡ ¬P ∧ ¬Q | Negate OR |
| **Quantifier Negation** | ¬(∀x: P) ≡ ∃x: ¬P | "Not all" = "Some don't" |
| **Quantifier Negation** | ¬(∃x: P) ≡ ∀x: ¬P | "None" = "All don't" |
| **Modus Ponens** | P → Q, P ⊢ Q | Affirm antecedent |
| **Modus Tollens** | P → Q, ¬Q ⊢ ¬P | Deny consequent |

---

## ⚠️ Common Mistakes & Pitfalls

### Mistake 1: Affirming the Consequent

```
❌ FALLACY:
   P → Q (If rain, then wet)
   Q     (Ground is wet)
   ∴ P   (Therefore it rained)  ← INVALID!
   
   The ground could be wet from sprinklers!

✅ VALID (Modus Ponens):
   P → Q, P ⊢ Q
```

### Mistake 2: Denying the Antecedent

```
❌ FALLACY:
   P → Q (If rain, then wet)
   ¬P    (Not raining)
   ∴ ¬Q  (Therefore not wet)  ← INVALID!
   
   Ground could be wet from other reasons!

✅ VALID (Modus Tollens):
   P → Q, ¬Q ⊢ ¬P
```

### Mistake 3: Quantifier Order

```
❌ WRONG: ∀x ∃y = ∃y ∀x
   These are NOT equivalent!

✅ RIGHT:
   ∀x ∃y: "For each x, there's a (possibly different) y"
   ∃y ∀x: "One y works for ALL x" (much stronger)
```

### Mistake 4: Misreading Implication

```
❌ WRONG: P → Q means P causes Q

✅ RIGHT: P → Q means "if P is true, Q is true"
   - Can be true even when P is false (vacuously)
   - Doesn't imply causation, just logical relation
```

---

## 💻 Code Implementations

```python
"""
Mathematical Logic: Complete Implementation
============================================

Boolean logic, quantifiers, and inference for ML.
"""

import numpy as np
from itertools import product
from typing import Callable, List, Any

class PropositionalLogic:
    """Propositional logic operations and truth tables."""
    
    @staticmethod
    def NOT(P: bool) -> bool:
        return not P
    
    @staticmethod
    def AND(P: bool, Q: bool) -> bool:
        return P and Q
    
    @staticmethod
    def OR(P: bool, Q: bool) -> bool:
        return P or Q
    
    @staticmethod
    def IMPLIES(P: bool, Q: bool) -> bool:
        """P → Q ≡ ¬P ∨ Q"""
        return (not P) or Q
    
    @staticmethod
    def IFF(P: bool, Q: bool) -> bool:
        """P ↔ Q"""
        return P == Q
    
    @staticmethod
    def XOR(P: bool, Q: bool) -> bool:
        return P != Q
    
    @staticmethod
    def truth_table(formula: Callable, n_vars: int = 2):
        """Generate truth table for formula."""
        table = []
        for values in product([False, True], repeat=n_vars):
            result = formula(*values)
            table.append((*values, result))
        return table
    
    @staticmethod
    def is_tautology(formula: Callable, n_vars: int = 2) -> bool:
        """Check if formula is always true."""
        return all(formula(*vals) for vals in product([False, True], repeat=n_vars))
    
    @staticmethod
    def is_contradiction(formula: Callable, n_vars: int = 2) -> bool:
        """Check if formula is always false."""
        return not any(formula(*vals) for vals in product([False, True], repeat=n_vars))
    
    @staticmethod
    def are_equivalent(f1: Callable, f2: Callable, n_vars: int = 2) -> bool:
        """Check if two formulas are logically equivalent."""
        for vals in product([False, True], repeat=n_vars):
            if f1(*vals) != f2(*vals):
                return False
        return True

class PredicateLogic:
    """Predicate logic with quantifiers."""
    
    @staticmethod
    def forall(predicate: Callable[[Any], bool], domain: List[Any]) -> bool:
        """∀x ∈ domain: predicate(x)"""
        return all(predicate(x) for x in domain)
    
    @staticmethod
    def exists(predicate: Callable[[Any], bool], domain: List[Any]) -> bool:
        """∃x ∈ domain: predicate(x)"""
        return any(predicate(x) for x in domain)
    
    @staticmethod
    def exists_unique(predicate: Callable[[Any], bool], domain: List[Any]) -> bool:
        """∃!x ∈ domain: predicate(x)"""
        count = sum(1 for x in domain if predicate(x))
        return count == 1
    
    @staticmethod
    def forall_exists(predicate: Callable[[Any, Any], bool], 
                      domain_x: List[Any], domain_y: List[Any]) -> bool:
        """∀x ∃y: predicate(x, y)"""
        return all(any(predicate(x, y) for y in domain_y) for x in domain_x)
    
    @staticmethod
    def exists_forall(predicate: Callable[[Any, Any], bool],
                      domain_x: List[Any], domain_y: List[Any]) -> bool:
        """∃y ∀x: predicate(x, y)"""
        return any(all(predicate(x, y) for x in domain_x) for y in domain_y)

class InferenceRules:
    """Rules of inference."""
    
    @staticmethod
    def modus_ponens(p_implies_q: bool, p: bool):
        """P → Q, P ⊢ Q"""
        if p_implies_q and p:
            return True  # Q is true
        return None  # Cannot conclude
    
    @staticmethod
    def modus_tollens(p_implies_q: bool, not_q: bool):
        """P → Q, ¬Q ⊢ ¬P"""
        if p_implies_q and not_q:
            return True  # ¬P is true
        return None
    
    @staticmethod
    def hypothetical_syllogism(p_implies_q: bool, q_implies_r: bool):
        """P → Q, Q → R ⊢ P → R"""
        if p_implies_q and q_implies_r:
            return True  # P → R is valid
        return None
    
    @staticmethod
    def disjunctive_syllogism(p_or_q: bool, not_p: bool):
        """P ∨ Q, ¬P ⊢ Q"""
        if p_or_q and not_p:
            return True  # Q is true
        return None

class NeuralLogicGates:
    """Boolean logic gates as neural networks."""
    
    @staticmethod
    def step(z):
        """Step activation."""
        return 1.0 if z >= 0 else 0.0
    
    @staticmethod
    def AND(x1, x2):
        """AND gate: w=[1,1], b=-1.5"""
        return NeuralLogicGates.step(x1 + x2 - 1.5)
    
    @staticmethod
    def OR(x1, x2):
        """OR gate: w=[1,1], b=-0.5"""
        return NeuralLogicGates.step(x1 + x2 - 0.5)
    
    @staticmethod
    def NOT(x):
        """NOT gate: w=-1, b=0.5"""
        return NeuralLogicGates.step(-x + 0.5)
    
    @staticmethod
    def NAND(x1, x2):
        """NAND gate: w=[-1,-1], b=1.5"""
        return NeuralLogicGates.step(-x1 - x2 + 1.5)
    
    @staticmethod
    def XOR(x1, x2):
        """XOR: 2-layer network (OR ∧ NAND)"""
        layer1_or = NeuralLogicGates.OR(x1, x2)
        layer1_nand = NeuralLogicGates.NAND(x1, x2)
        return NeuralLogicGates.AND(layer1_or, layer1_nand)

# =============================================================================
# DEMONSTRATIONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MATHEMATICAL LOGIC: DEMONSTRATIONS")
    print("=" * 60)
    
    # 1. Verify P → Q ≡ ¬P ∨ Q
    print("\n1. IMPLICATION EQUIVALENCE")
    print("-" * 40)
    
    impl = lambda P, Q: PropositionalLogic.IMPLIES(P, Q)
    equiv = lambda P, Q: PropositionalLogic.OR(not P, Q)
    
    print(f"P → Q ≡ ¬P ∨ Q: {PropositionalLogic.are_equivalent(impl, equiv)}")
    
    # 2. De Morgan's Laws
    print("\n2. DE MORGAN'S LAWS")
    print("-" * 40)
    
    demorgan1_lhs = lambda P, Q: not (P and Q)
    demorgan1_rhs = lambda P, Q: (not P) or (not Q)
    print(f"¬(P ∧ Q) ≡ ¬P ∨ ¬Q: {PropositionalLogic.are_equivalent(demorgan1_lhs, demorgan1_rhs)}")
    
    # 3. Quantifier examples
    print("\n3. QUANTIFIERS")
    print("-" * 40)
    
    domain = list(range(1, 11))
    
    print(f"∀x ∈ [1,10]: x > 0: {PredicateLogic.forall(lambda x: x > 0, domain)}")
    print(f"∃x ∈ [1,10]: x > 5: {PredicateLogic.exists(lambda x: x > 5, domain)}")
    print(f"∃!x ∈ [1,10]: x = 5: {PredicateLogic.exists_unique(lambda x: x == 5, domain)}")
    
    # 4. Quantifier order matters
    print("\n4. QUANTIFIER ORDER")
    print("-" * 40)
    
    domain_xy = [1, 2, 3]
    P = lambda x, y: y > x
    
    forall_exists = PredicateLogic.forall_exists(P, domain_xy, domain_xy)
    exists_forall = PredicateLogic.exists_forall(P, domain_xy, domain_xy)
    
    print(f"∀x ∃y: y > x: {forall_exists}")
    print(f"∃y ∀x: y > x: {exists_forall}")
    print(f"Order matters: {forall_exists != exists_forall}")
    
    # 5. Neural logic gates
    print("\n5. NEURAL LOGIC GATES")
    print("-" * 40)
    
    print("XOR via 2-layer network:")
    for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        result = NeuralLogicGates.XOR(x1, x2)
        expected = x1 != x2
        print(f"XOR({x1}, {x2}) = {int(result)}, Expected: {int(expected)}, ✓" if result == expected else "✗")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 60)
```

---

## 📚 Resources

| Type | Title | Link |
|:-----|:------|:-----|
| 📖 Book | Logic in Computer Science (Huth & Ryan) | [Cambridge](https://www.cambridge.org/core/books/logic-in-computer-science/A60C1BEC4FE5C271231A63F03880F8D0) |
| 📖 Book | Introduction to Logic (Tarski) | Classic text |
| 📖 Free | Open Logic Project | [openlogicproject.org](https://openlogicproject.org/) |
| 🎥 Video | Discrete Math - Logic | [YouTube](https://www.youtube.com/watch?v=itrXYg41-V0) |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[🔢 Set Theory](../03_set_theory/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 4 of 6**<br>
**🔀 Mathematical Logic**

</td>
<td align="right" width="33%">

➡️ **Next**<br>
[⏱️ Asymptotic Analysis](../05_asymptotic_analysis/README.md)

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
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,3,4&height=100&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=FFD93D&center=true&vCenter=true&width=600&lines=Made+with+❤️+by+Gaurav+Goswami;Part+of+ML+Researcher+Foundations+Series" alt="Footer" />
</p>

<p align="center">
  <a href="https://github.com/Gaurav14cs17">
    <img src="https://img.shields.io/badge/GitHub-Gaurav14cs17-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>
