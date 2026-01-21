<!-- Animated Header -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=4ECDC4&center=true&vCenter=true&width=800&lines=🔢+Set+Theory;The+Language+of+Mathematics" alt="Set Theory" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,25,26&height=180&section=header&text=Set%20Theory&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Sets%20•%20Functions%20•%20Relations%20•%20Cardinality%20•%20σ-Algebras&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03_of_06-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Topics-7_Concepts-FF6B6B?style=for-the-badge&logo=buffer&logoColor=white" alt="Topics"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-00D4AA?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-6C63FF?style=for-the-badge&logo=calendar&logoColor=white" alt="Updated"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Difficulty-Intermediate-orange?style=flat-square" alt="Difficulty"/>
  <img src="https://img.shields.io/badge/Reading_Time-50_minutes-blue?style=flat-square" alt="Reading Time"/>
  <img src="https://img.shields.io/badge/Prerequisites-Mathematical_Thinking-green?style=flat-square" alt="Prerequisites"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01_mathematical_thinking/README.md) → [Proof Techniques](../02_proof_techniques/README.md) → Set Theory → [Logic](../04_logic/README.md) → [Asymptotic Analysis](../05_asymptotic_analysis/README.md) → [Numerical Computation](../06_numerical_computation/README.md)

---

## 📌 TL;DR

Set theory is the **foundation of probability and ML**. This article covers:

- **Set Operations** — Union, intersection, difference, complement (data operations)

- **Functions** — Injective, surjective, bijective (normalizing flows, encoders)

- **Relations** — Equivalence relations and partitions (clustering)

- **Cardinality** — Finite, countable, uncountable (discrete vs continuous)

- **σ-Algebra** — Foundation for probability spaces (measure theory)

> [!NOTE]
> **Why This Matters:** Every probability distribution is defined on a σ-algebra. Understanding sets is essential for probabilistic ML, data manipulation, and theoretical analysis.

---

## 📚 What You'll Learn

By the end of this article, you will be able to:

- [ ] Perform set operations and apply De Morgan's laws

- [ ] Classify functions (injective, surjective, bijective)

- [ ] Understand equivalence relations and partitions

- [ ] Distinguish countable from uncountable sets

- [ ] Define and work with σ-algebras

- [ ] Apply set theory to data operations (SQL, Pandas)

---

## 📑 Table of Contents

1. [Visual Overview](#-visual-overview)

2. [Basic Set Operations](#1-basic-set-operations)

3. [Set Laws and Identities](#2-set-laws-and-identities)

4. [Functions](#3-functions)

5. [Relations](#4-relations)

6. [Cardinality](#5-cardinality)

7. [σ-Algebras and Probability Spaces](#6-σ-algebras-and-probability-spaces)

8. [Key Formulas Summary](#-key-formulas-summary)

9. [Common Mistakes & Pitfalls](#-common-mistakes--pitfalls)

10. [Code Implementations](#-code-implementations)

11. [ML Applications](#-ml-applications)

12. [Resources](#-resources)

13. [Navigation](#-navigation)

---

## 🎯 Visual Overview

### Set Operations Venn Diagram

```
+-----------------------------------------------------------------------------+
|                           SET OPERATIONS                                    |
+-----------------------------------------------------------------------------+
|                                                                             |
|   UNION (A ∪ B)           INTERSECTION (A ∩ B)       DIFFERENCE (A \ B)    |
|   +-----+-----+           +-----+-----+              +-----+-----+         |
|   |▓▓▓▓▓|▓▓▓▓▓|           |     |▓▓▓▓▓|              |▓▓▓▓▓|     |         |
|   |▓▓A▓▓|▓▓B▓▓|           |  A  |▓▓B▓▓|              |▓▓A▓▓|  B  |         |
|   |▓▓▓▓▓|▓▓▓▓▓|           |     |▓▓▓▓▓|              |▓▓▓▓▓|     |         |
|   +-----+-----+           +-----+-----+              +-----+-----+         |
|   "A or B"                "A and B"                  "A but not B"          |
|                                                                             |
|   COMPLEMENT (Aᶜ)         SYMMETRIC DIFF (A △ B)    CARTESIAN (A × B)     |
|   +-------------+         +-----+-----+              A = {1,2}             |
|   |▓▓▓▓▓|     |▓|         |▓▓▓▓▓|▓▓▓▓▓|              B = {a,b}             |
|   |▓▓▓▓▓|  A  |▓|         |▓▓A▓▓|▓▓B▓▓|              A×B = {(1,a),(1,b),   |
|   |▓▓▓▓▓|     |▓|         |▓▓▓▓▓|▓▓▓▓▓|                     (2,a),(2,b)}   |
|   +-------------+         +-----+-----+              |A×B| = |A|·|B| = 4   |
|   "Everything not A"      "(A or B) but not both"                          |
|                                                                             |
+-----------------------------------------------------------------------------+

```

### Function Types Visualization

```
+-----------------------------------------------------------------------------+
|                            FUNCTION TYPES                                   |
+-----------------------------------------------------------------------------+
|                                                                             |
|  💉 INJECTIVE (1-to-1)     🎯 SURJECTIVE (Onto)      🔄 BIJECTIVE (Both)   |
|  +-----------------+      +-----------------+      +-----------------+     |
|  |  A      B       |      |  A      B       |      |  A      B       |     |
|  |  a ----▶ 1      |      |  a --+--▶ 1     |      |  a ◀---▶ 1      |     |
|  |  b ----▶ 2      |      |  b --+          |      |  b ◀---▶ 2      |     |
|  |  c ----▶ 3      |      |  c ----▶ 2      |      |  c ◀---▶ 3      |     |
|  |         4       |      |                 |      |                 |     |
|  +-----------------+      +-----------------+      +-----------------+     |
|  f(a)=f(b) ⟹ a=b         Every y has preimage     Both: invertible!       |
|  "No collisions"          "Covers all of B"        f⁻¹ exists              |
|                                                                             |
|  ML Example:              ML Example:              ML Example:              |
|  • Encoders               • Surjective mappings    • Normalizing Flows     |
|  • Embeddings             • Full class coverage    • Autoencoders (ideal)  |
|                                                                             |
+-----------------------------------------------------------------------------+

```

---

## 1. Basic Set Operations

### 📖 Definitions

| Operation | Symbol | Definition | Set-Builder |
|:----------|:------:|:-----------|:------------|
| **Union** | A ∪ B | Elements in A OR B | {x : x ∈ A ∨ x ∈ B} |
| **Intersection** | A ∩ B | Elements in A AND B | {x : x ∈ A ∧ x ∈ B} |
| **Difference** | A \ B | Elements in A but NOT B | {x : x ∈ A ∧ x ∉ B} |
| **Complement** | Aᶜ | Elements NOT in A | {x : x ∉ A} = U \ A |
| **Symmetric Diff** | A △ B | In A or B but not both | (A \ B) ∪ (B \ A) |
| **Cartesian Product** | A × B | All ordered pairs | {(a,b) : a ∈ A, b ∈ B} |
| **Power Set** | P(A) | All subsets of A | {S : S ⊆ A} |

### 📝 Examples

#### Example 1: Basic Operations (Simple)

```
Let A = {1, 2, 3, 4}
Let B = {3, 4, 5, 6}

A ∪ B = {1, 2, 3, 4, 5, 6}      (Union: everything)
A ∩ B = {3, 4}                  (Intersection: common elements)
A \ B = {1, 2}                  (Difference: in A, not in B)
B \ A = {5, 6}                  (Difference: in B, not in A)
A △ B = {1, 2, 5, 6}            (Symmetric difference)

```

#### Example 2: Power Set (Intermediate)

```
Let A = {1, 2}

P(A) = {∅, {1}, {2}, {1,2}}

|P(A)| = 2^|A| = 2² = 4

```

**Proof that |P(A)| = 2^|A|:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Each element x ∈ A is either in subset S or not | Binary choice |
| 2 | For |A| = n elements, there are 2 choices per element | Independence |
| 3 | Total subsets = 2 × 2 × ... × 2 (n times) = 2ⁿ | Multiplication principle |
| 4 | Therefore \|P(A)\| = 2^{\|A\|} | ∎ |

#### Example 3: Cartesian Product (Intermediate)

```
Let X = {red, blue}
Let Y = {1, 2, 3}

X × Y = {(red,1), (red,2), (red,3), (blue,1), (blue,2), (blue,3)}

|X × Y| = |X| · |Y| = 2 · 3 = 6

```

**ML Application:** Feature space is Cartesian product of feature domains!

```
Features: Height × Weight × Age = ℝ × ℝ × ℝ = ℝ³

```

### 💻 Code Implementation

```python
# Python set operations
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

print(f"A = {A}")
print(f"B = {B}")
print(f"A ∪ B = {A | B}")           # Union
print(f"A ∩ B = {A & B}")           # Intersection
print(f"A \\ B = {A - B}")          # Difference
print(f"A △ B = {A ^ B}")           # Symmetric difference
print(f"A ⊆ B: {A <= B}")           # Subset
print(f"A ⊂ B: {A < B}")            # Proper subset

# Cartesian product
from itertools import product
X = {'red', 'blue'}
Y = {1, 2, 3}
cartesian = set(product(X, Y))
print(f"X × Y = {cartesian}")

# Power set
from itertools import combinations
def power_set(s):
    s = list(s)
    return [set(c) for i in range(len(s)+1) for c in combinations(s, i)]

print(f"P({{1,2}}) = {power_set({1,2})}")

```

---

## 2. Set Laws and Identities

### 📖 De Morgan's Laws

| Law | Formula | In Words |
|:----|:--------|:---------|
| **Union Complement** | (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ | Not (A or B) = (not A) and (not B) |
| **Intersection Complement** | (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ | Not (A and B) = (not A) or (not B) |
| **Generalized Union** | (⋃ᵢ Aᵢ)ᶜ = ⋂ᵢ Aᵢᶜ | |
| **Generalized Intersection** | (⋂ᵢ Aᵢ)ᶜ = ⋃ᵢ Aᵢᶜ | |

### 📐 Proof: De Morgan's Law (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ

**Proof (show mutual inclusion):**

**Part 1: (A ∪ B)ᶜ ⊆ Aᶜ ∩ Bᶜ**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Let x ∈ (A ∪ B)ᶜ | Assumption |
| 2 | x ∉ A ∪ B | Definition of complement |
| 3 | x ∉ A AND x ∉ B | Negation of union |
| 4 | x ∈ Aᶜ AND x ∈ Bᶜ | Definition of complement |
| 5 | x ∈ Aᶜ ∩ Bᶜ | Definition of intersection ✓ |

**Part 2: Aᶜ ∩ Bᶜ ⊆ (A ∪ B)ᶜ**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Let x ∈ Aᶜ ∩ Bᶜ | Assumption |
| 2 | x ∈ Aᶜ AND x ∈ Bᶜ | Definition of intersection |
| 3 | x ∉ A AND x ∉ B | Definition of complement |
| 4 | x ∉ A ∪ B | Negation of union |
| 5 | x ∈ (A ∪ B)ᶜ | Definition of complement ✓ |

**By mutual inclusion, (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ** ∎

### 📖 Distributive Laws

| Law | Formula |
|:----|:--------|
| **∩ over ∪** | A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) |
| **∪ over ∩** | A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) |

### 📖 Inclusion-Exclusion Principle

**For 2 sets:**

$$|A \cup B| = |A| + |B| - |A \cap B|$$

**For 3 sets:**

$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|$$

**General form:**

$$\left|\bigcup_{i=1}^n A_i\right| = \sum_{i}|A_i| - \sum_{i<j}|A_i \cap A_j| + \sum_{i<j<k}|A_i \cap A_j \cap A_k| - \cdots$$

### 📐 Proof: Inclusion-Exclusion for 2 Sets

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | A ∪ B = A ∪ (B \ A) | Disjoint decomposition |
| 2 | \|A ∪ B\| = \|A\| + \|B \ A\| | Disjoint sets |
| 3 | B = (B ∩ A) ∪ (B \ A) | Partition of B |
| 4 | \|B\| = \|B ∩ A\| + \|B \ A\| | Disjoint sets |
| 5 | \|B \ A\| = \|B\| - \|B ∩ A\| | From Step 4 |
| 6 | \|A ∪ B\| = \|A\| + \|B\| - \|A ∩ B\| | Substituting Step 5 into Step 2 ∎ |

---

## 3. Functions

### 📖 Definition

> A **function** f: A → B assigns to each element a ∈ A exactly one element f(a) ∈ B.
> - **Domain:** A (input set)
> - **Codomain:** B (possible output set)
> - **Range/Image:** f(A) = {f(a) : a ∈ A} ⊆ B (actual outputs)

### 📖 Function Types

| Type | Definition | ML Application |
|:-----|:-----------|:---------------|
| **Injective (1-to-1)** | f(a) = f(b) ⟹ a = b | Encoders, embeddings |
| **Surjective (Onto)** | ∀b ∈ B, ∃a ∈ A: f(a) = b | Full class coverage |
| **Bijective** | Injective AND Surjective | Normalizing flows |

### 📐 Proofs for Function Types

#### Proof: f(x) = 2x is Injective on ℝ

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume f(a) = f(b) | Hypothesis |
| 2 | 2a = 2b | Definition of f |
| 3 | a = b | Divide by 2 |
| 4 | f is injective | Definition ∎ |

#### Proof: f(x) = x² is NOT Injective on ℝ

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | f(1) = 1² = 1 | Calculation |
| 2 | f(-1) = (-1)² = 1 | Calculation |
| 3 | f(1) = f(-1) but 1 ≠ -1 | Observation |
| 4 | f is not injective | Counterexample ∎ |

#### Proof: exp: ℝ → ℝ⁺ is Bijective

**Injective:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume eᵃ = eᵇ | Hypothesis |
| 2 | ln(eᵃ) = ln(eᵇ) | Apply ln (monotonic) |
| 3 | a = b | ln(eˣ) = x |
| 4 | exp is injective | ∎ |

**Surjective:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Let y ∈ ℝ⁺ (any positive real) | Arbitrary |
| 2 | Let a = ln(y) | Construction |
| 3 | exp(a) = exp(ln(y)) = y | Inverse property |
| 4 | exp is surjective | ∎ |

**Therefore exp: ℝ → ℝ⁺ is bijective with inverse ln: ℝ⁺ → ℝ.**

### 📝 ML Examples

| Model Component | Function Type | Why |
|:----------------|:--------------|:----|
| **Encoder (VAE)** | Injective (ideally) | No information loss |
| **Classifier** | Surjective (ideally) | Cover all classes |
| **Normalizing Flow** | Bijective | Invertible transformation |
| **ReLU** | Neither | ReLU(-1) = ReLU(-2) = 0 |
| **Softmax** | Neither | Many inputs → same output |

### 💻 Code Implementation

```python
import numpy as np

def is_injective(f, domain_samples, tolerance=1e-10):
    """Check if function appears injective (no collisions)."""
    outputs = [f(x) for x in domain_samples]
    unique_outputs = len(set([round(o, 10) for o in outputs]))
    return unique_outputs == len(outputs)

def is_surjective(f, domain_samples, codomain_samples, tolerance=1e-6):
    """Check if function covers codomain (approximately)."""
    outputs = set([round(f(x), 6) for x in domain_samples])
    codomain = set([round(y, 6) for y in codomain_samples])
    # Check if each codomain point is approximately reached
    covered = sum(1 for y in codomain if any(abs(o - y) < tolerance for o in outputs))
    return covered == len(codomain)

# Examples
f_injective = lambda x: 2*x  # Injective
f_not_injective = lambda x: x**2  # Not injective (f(1) = f(-1))

domain = np.linspace(-10, 10, 1000)

print(f"f(x) = 2x injective: {is_injective(f_injective, domain)}")
print(f"f(x) = x² injective: {is_injective(f_not_injective, domain)}")

```

---

## 4. Relations

### 📖 Definition

> A **relation** R on set A is a subset of A × A. We write aRb if (a, b) ∈ R.

### 📖 Relation Properties

| Property | Definition | Example |
|:---------|:-----------|:--------|
| **Reflexive** | ∀a: aRa | a ≤ a |
| **Symmetric** | aRb ⟹ bRa | a = b |
| **Antisymmetric** | aRb ∧ bRa ⟹ a = b | a ≤ b |
| **Transitive** | aRb ∧ bRc ⟹ aRc | a < b < c ⟹ a < c |

### 📖 Equivalence Relations

> **Equivalence Relation:** Reflexive + Symmetric + Transitive

**Key Theorem:** Every equivalence relation on A creates a **partition** of A into equivalence classes.

### 📐 Proof: Equivalence Relation → Partition

**Definition:** Equivalence class [a] = {b ∈ A : aRb}

**Proof that equivalence classes partition A:**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | **Non-empty:** a ∈ [a] | Reflexivity: aRa |
| 2 | **Cover A:** ∀a ∈ A, a ∈ [a] | From Step 1 |
| 3 | **Disjoint:** Suppose [a] ∩ [b] ≠ ∅ | Assumption |
| 4 | Let c ∈ [a] ∩ [b] | Exists by Step 3 |
| 5 | aRc and bRc | Definition of [a], [b] |
| 6 | cRb | Symmetry |
| 7 | aRb | Transitivity: aRc, cRb |
| 8 | [a] = [b] | Elements equivalent |
| 9 | Contrapositive: [a] ≠ [b] ⟹ [a] ∩ [b] = ∅ | Disjoint |
| 10 | Equivalence classes partition A | ∎ |

### 📝 ML Examples

| Relation | Properties | ML Application |
|:---------|:-----------|:---------------|
| **Same cluster** | Equivalence | K-means clustering |
| **Same latent code** | Equivalence | VAE representations |
| **Within ε distance** | Reflexive, Symmetric (not transitive!) | ε-neighborhoods |
| **Preference order** | Antisymmetric, Transitive | Ranking |

### 💻 Code Implementation

```python
def check_equivalence(relation, domain):
    """Check if relation is an equivalence relation."""
    
    # Reflexive: ∀a: aRa
    reflexive = all(relation(a, a) for a in domain)
    
    # Symmetric: aRb ⟹ bRa
    symmetric = all(
        not relation(a, b) or relation(b, a)
        for a in domain for b in domain
    )
    
    # Transitive: aRb ∧ bRc ⟹ aRc
    transitive = all(
        not (relation(a, b) and relation(b, c)) or relation(a, c)
        for a in domain for b in domain for c in domain
    )
    
    return {
        'reflexive': reflexive,
        'symmetric': symmetric,
        'transitive': transitive,
        'is_equivalence': reflexive and symmetric and transitive
    }

# Example: Same parity (both even or both odd)
same_parity = lambda a, b: (a % 2) == (b % 2)
domain = range(10)

result = check_equivalence(same_parity, domain)
print(f"Same parity relation: {result}")
# Output: is_equivalence: True

```

---

## 5. Cardinality

### 📖 Definition

> The **cardinality** of a set A, denoted |A|, is the "size" of A.
> 
> Two sets have the same cardinality if there exists a bijection between them.

### 📖 Cardinality Types

| Type | Definition | Examples |
|:-----|:-----------|:---------|
| **Finite** | \|A\| = n for some n ∈ ℕ | {1,2,3}, ∅ |
| **Countably Infinite** | \|A\| = \|ℕ\| = ℵ₀ | ℕ, ℤ, ℚ |
| **Uncountably Infinite** | \|A\| > \|ℕ\| | ℝ, [0,1], P(ℕ) |

### 📐 Proof: ℤ is Countable

**Construct bijection f: ℕ → ℤ:**

```
n:    0  1  2  3  4  5  6  ...
f(n): 0  1 -1  2 -2  3 -3  ...

```

**Formula:**

$$f(n) = \begin{cases} n/2 & \text{if } n \text{ even} \\ -(n+1)/2 & \text{if } n \text{ odd} \end{cases}$$

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | f is well-defined | Formula gives unique output |
| 2 | f is injective | Different n give different f(n) |
| 3 | f is surjective | Every integer appears |
| 4 | f is a bijection | Injective + Surjective |
| 5 | \|ℤ\| = \|ℕ\| | Definition of equal cardinality ∎ |

### 📐 Proof: ℝ is Uncountable (Cantor's Diagonal Argument)

**Theorem:** There is no bijection between ℕ and ℝ (or even [0,1]).

**Proof (by contradiction):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Assume [0,1] is countable | Hypothesis |
| 2 | List all reals: r₁, r₂, r₃, ... | Assumption |
| 3 | Write in decimal: rᵢ = 0.dᵢ₁dᵢ₂dᵢ₃... | Decimal expansion |
| 4 | Construct x = 0.b₁b₂b₃... where bᵢ ≠ dᵢᵢ | Diagonal construction |
| 5 | x differs from rᵢ at position i | By construction |
| 6 | x ∉ {r₁, r₂, r₃, ...} | x ≠ rᵢ for all i |
| 7 | But x ∈ [0,1]! | 💥 Contradiction |
| 8 | [0,1] is uncountable | ∎ |

**Diagonal visualization:**

```
r₁ = 0.[d₁₁]d₁₂ d₁₃ d₁₄ ...
r₂ = 0. d₂₁[d₂₂]d₂₃ d₂₄ ...
r₃ = 0. d₃₁ d₃₂[d₃₃]d₃₄ ...
r₄ = 0. d₄₁ d₄₂ d₄₃[d₄₄]...
...
x  = 0. b₁  b₂  b₃  b₄ ...   where bᵢ ≠ dᵢᵢ

```

### 📝 ML Implications

| Set | Cardinality | ML Implication |
|:----|:------------|:---------------|
| **Training data** | Finite | Can enumerate |
| **NN parameters** | Finite | Can optimize |
| **All functions ℝⁿ → ℝ** | Uncountable | Cannot enumerate |
| **Continuous distributions** | Uncountable | Need density, not pmf |

---

## 6. σ-Algebras and Probability Spaces

### 📖 Definition: σ-Algebra

> A collection F ⊆ P(Ω) is a **σ-algebra** on Ω if:
> 1. **Contains Ω:** Ω ∈ F
> 2. **Closed under complement:** A ∈ F ⟹ Aᶜ ∈ F
> 3. **Closed under countable union:** A₁, A₂, ... ∈ F ⟹ ⋃ᵢAᵢ ∈ F

### 📖 Probability Space

> A **probability space** is a triple (Ω, F, P) where:
> - **Ω:** Sample space (set of all outcomes)
> - **F:** σ-algebra of events (measurable sets)
> - **P:** Probability measure P: F → [0,1]

### 📐 Proof: σ-Algebra is Closed Under Countable Intersection

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Let A₁, A₂, ... ∈ F | Assumption |
| 2 | Aᵢᶜ ∈ F for all i | Closed under complement |
| 3 | ⋃ᵢ Aᵢᶜ ∈ F | Closed under countable union |
| 4 | (⋃ᵢ Aᵢᶜ)ᶜ ∈ F | Closed under complement |
| 5 | ⋂ᵢ Aᵢ = (⋃ᵢ Aᵢᶜ)ᶜ | De Morgan's law |
| 6 | ⋂ᵢ Aᵢ ∈ F | From Steps 4, 5 ∎ |

### 📝 Example: Coin Flip Probability Space

```
Ω = {H, T}                          (Sample space)
F = {∅, {H}, {T}, {H,T}} = P(Ω)     (σ-algebra: all subsets)
P({H}) = 0.5, P({T}) = 0.5          (Fair coin)

Verify F is σ-algebra:

1. Ω = {H,T} ∈ F  ✓

2. {H}ᶜ = {T} ∈ F, {T}ᶜ = {H} ∈ F, etc.  ✓

3. {H} ∪ {T} = {H,T} ∈ F  ✓

```

### 📝 Example: Borel σ-Algebra

> The **Borel σ-algebra** on ℝ is the smallest σ-algebra containing all open intervals.
> 
> It includes: open sets, closed sets, countable unions/intersections, and more.

**ML Use:** Continuous random variables are measurable with respect to Borel σ-algebra.

### 💻 Code Implementation

```python
import numpy as np
from itertools import combinations, chain

def power_set(s):
    """Generate power set of s."""
    s = list(s)
    return [frozenset(c) for i in range(len(s)+1) 
            for c in combinations(s, i)]

def is_sigma_algebra(F, omega):
    """Check if F is a σ-algebra on omega."""
    omega = frozenset(omega)
    F = [frozenset(s) for s in F]
    
    # 1. Contains omega
    if omega not in F:
        return False, "Missing omega"
    
    # 2. Closed under complement
    for A in F:
        complement = omega - A
        if complement not in F:
            return False, f"Missing complement of {set(A)}"
    
    # 3. Closed under finite union (for finite case)
    for A in F:
        for B in F:
            if A.union(B) not in F:
                return False, f"Missing union of {set(A)} and {set(B)}"
    
    return True, "Valid σ-algebra"

# Example: Full power set is always a σ-algebra
omega = {1, 2, 3}
F = power_set(omega)
result, msg = is_sigma_algebra(F, omega)
print(f"P(Ω) is σ-algebra: {result} - {msg}")

# Example: Not a σ-algebra (missing complement)
omega = {1, 2}
F_bad = [set(), {1}, {1, 2}]  # Missing {2}
result, msg = is_sigma_algebra(F_bad, omega)
print(f"{{∅, {{1}}, Ω}} is σ-algebra: {result} - {msg}")

```

---

## 📊 Key Formulas Summary

| Concept | Formula | Notes |
|:--------|:--------|:------|
| **Union** | A ∪ B = {x : x ∈ A ∨ x ∈ B} | OR |
| **Intersection** | A ∩ B = {x : x ∈ A ∧ x ∈ B} | AND |
| **Complement** | Aᶜ = {x : x ∉ A} | NOT |
| **De Morgan 1** | (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ | Flip operation |
| **De Morgan 2** | (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ | Flip operation |
| **Power Set Size** | \|P(A)\| = 2^{\|A\|} | Exponential |
| **Cartesian Size** | \|A × B\| = \|A\| · \|B\| | Multiplicative |
| **Inclusion-Exclusion** | \|A ∪ B\| = \|A\| + \|B\| - \|A ∩ B\| | Avoid double counting |

---

## ⚠️ Common Mistakes & Pitfalls

### Mistake 1: Confusing ⊆ and ∈

```
❌ WRONG: {1} ∈ {1, 2, 3}
   {1} is a SET, not an element of {1,2,3}

✅ RIGHT: 1 ∈ {1, 2, 3}
          {1} ⊆ {1, 2, 3}

```

### Mistake 2: Forgetting Empty Set in Power Set

```
❌ WRONG: P({1,2}) = {{1}, {2}, {1,2}}
   Missing ∅!

✅ RIGHT: P({1,2}) = {∅, {1}, {2}, {1,2}}

```

### Mistake 3: Thinking Surjective ⟹ Injective

```
❌ WRONG: f is onto, so f is one-to-one

✅ RIGHT: f(x) = x² is surjective on ℝ → [0,∞)
          but NOT injective (f(1) = f(-1) = 1)

```

### Mistake 4: Confusing Cardinality

```
❌ WRONG: ℤ has more elements than ℕ

✅ RIGHT: |ℤ| = |ℕ| = ℵ₀
          Both are countably infinite!

```

---

## 💻 Code Implementations

```python
"""
Set Theory: Complete Implementation
====================================

Comprehensive set operations and verifications for ML.
"""

import numpy as np
from typing import Set, Callable, List, Tuple, FrozenSet
from itertools import combinations, product
from dataclasses import dataclass

class SetTheory:
    """Set theory operations and verifications."""
    
    # =========================================================================
    # SET OPERATIONS
    # =========================================================================
    
    @staticmethod
    def union(A: set, B: set) -> set:
        """A ∪ B"""
        return A | B
    
    @staticmethod
    def intersection(A: set, B: set) -> set:
        """A ∩ B"""
        return A & B
    
    @staticmethod
    def difference(A: set, B: set) -> set:
        """A \ B"""
        return A - B
    
    @staticmethod
    def symmetric_difference(A: set, B: set) -> set:
        """A △ B"""
        return A ^ B
    
    @staticmethod
    def complement(A: set, U: set) -> set:
        """Aᶜ relative to universal set U"""
        return U - A
    
    @staticmethod
    def cartesian_product(A: set, B: set) -> set:
        """A × B"""
        return set(product(A, B))
    
    @staticmethod
    def power_set(A: set) -> set:
        """P(A) - all subsets of A"""
        A_list = list(A)
        result = []
        for i in range(len(A_list) + 1):
            for subset in combinations(A_list, i):
                result.append(frozenset(subset))
        return set(result)
    
    # =========================================================================
    # LAWS VERIFICATION
    # =========================================================================
    
    @staticmethod
    def verify_de_morgan(A: set, B: set, U: set) -> dict:
        """Verify De Morgan's laws."""
        # Law 1: (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
        lhs1 = U - (A | B)
        rhs1 = (U - A) & (U - B)
        law1 = lhs1 == rhs1
        
        # Law 2: (A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
        lhs2 = U - (A & B)
        rhs2 = (U - A) | (U - B)
        law2 = lhs2 == rhs2
        
        return {
            'law1_holds': law1,
            'law2_holds': law2,
            'both_hold': law1 and law2
        }
    
    @staticmethod
    def verify_inclusion_exclusion(A: set, B: set) -> dict:
        """Verify inclusion-exclusion principle."""
        lhs = len(A | B)
        rhs = len(A) + len(B) - len(A & B)
        
        return {
            '|A∪B|': lhs,
            '|A|+|B|-|A∩B|': rhs,
            'holds': lhs == rhs
        }
    
    # =========================================================================
    # FUNCTION PROPERTIES
    # =========================================================================
    
    @staticmethod
    def is_function(R: set, A: set, B: set) -> bool:
        """Check if R ⊆ A × B is a function from A to B."""
        # Every element of A must appear exactly once as first component
        first_components = [pair[0] for pair in R]
        return set(first_components) == A and len(first_components) == len(A)
    
    @staticmethod
    def is_injective(f: Callable, domain: list) -> bool:
        """Check if f is injective on domain."""
        outputs = [f(x) for x in domain]
        return len(outputs) == len(set(outputs))
    
    @staticmethod
    def is_surjective(f: Callable, domain: list, codomain: list, tol=1e-6) -> bool:
        """Check if f is surjective (onto codomain)."""
        outputs = {round(f(x), 10) for x in domain}
        codomain_rounded = {round(y, 10) for y in codomain}
        return codomain_rounded.issubset(outputs)
    
    @staticmethod
    def is_bijective(f: Callable, domain: list, codomain: list) -> bool:
        """Check if f is bijective."""
        return (SetTheory.is_injective(f, domain) and 
                SetTheory.is_surjective(f, domain, codomain))
    
    # =========================================================================
    # RELATIONS
    # =========================================================================
    
    @staticmethod
    def check_relation_properties(R: Callable[[any, any], bool], domain: list) -> dict:
        """Check reflexive, symmetric, transitive properties."""
        reflexive = all(R(a, a) for a in domain)
        
        symmetric = all(
            not R(a, b) or R(b, a)
            for a in domain for b in domain
        )
        
        transitive = all(
            not (R(a, b) and R(b, c)) or R(a, c)
            for a in domain for b in domain for c in domain
        )
        
        antisymmetric = all(
            not (R(a, b) and R(b, a)) or (a == b)
            for a in domain for b in domain
        )
        
        return {
            'reflexive': reflexive,
            'symmetric': symmetric,
            'antisymmetric': antisymmetric,
            'transitive': transitive,
            'is_equivalence': reflexive and symmetric and transitive,
            'is_partial_order': reflexive and antisymmetric and transitive
        }
    
    @staticmethod
    def equivalence_classes(R: Callable[[any, any], bool], domain: list) -> List[set]:
        """Compute equivalence classes for equivalence relation R."""
        classes = []
        remaining = set(domain)
        
        while remaining:
            a = next(iter(remaining))
            eq_class = {b for b in domain if R(a, b)}
            classes.append(eq_class)
            remaining -= eq_class
        
        return classes
    
    # =========================================================================
    # CARDINALITY
    # =========================================================================
    
    @staticmethod
    def demonstrate_countability():
        """Demonstrate bijection between ℕ and ℤ."""
        def f(n):
            """Bijection ℕ → ℤ"""
            if n % 2 == 0:
                return n // 2
            else:
                return -(n + 1) // 2
        
        # Show first 10 values
        mapping = [(n, f(n)) for n in range(10)]
        print("Bijection ℕ → ℤ:")
        print("n:   ", [m[0] for m in mapping])
        print("f(n):", [m[1] for m in mapping])
        
        # Verify bijection
        outputs = [f(n) for n in range(100)]
        is_injective = len(outputs) == len(set(outputs))
        covers_range = set(outputs) == set(range(-50, 50))
        
        print(f"Injective (first 100): {is_injective}")
        print(f"Covers [-50, 49]: {covers_range}")

# =============================================================================
# DEMONSTRATIONS
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SET THEORY: DEMONSTRATIONS")
    print("=" * 60)
    
    # Set operations
    print("\n1. SET OPERATIONS")
    print("-" * 40)
    
    A = {1, 2, 3, 4}
    B = {3, 4, 5, 6}
    U = {1, 2, 3, 4, 5, 6, 7, 8}
    
    print(f"A = {A}")
    print(f"B = {B}")
    print(f"A ∪ B = {SetTheory.union(A, B)}")
    print(f"A ∩ B = {SetTheory.intersection(A, B)}")
    print(f"A \\ B = {SetTheory.difference(A, B)}")
    print(f"A △ B = {SetTheory.symmetric_difference(A, B)}")
    
    # De Morgan's laws
    print("\n2. DE MORGAN'S LAWS")
    print("-" * 40)
    result = SetTheory.verify_de_morgan(A, B, U)
    print(f"Law 1 (∪ complement): {result['law1_holds']}")
    print(f"Law 2 (∩ complement): {result['law2_holds']}")
    
    # Function properties
    print("\n3. FUNCTION PROPERTIES")
    print("-" * 40)
    
    domain = list(range(-10, 11))
    
    f1 = lambda x: 2*x  # Injective
    f2 = lambda x: x**2  # Not injective
    
    print(f"f(x) = 2x injective: {SetTheory.is_injective(f1, domain)}")
    print(f"f(x) = x² injective: {SetTheory.is_injective(f2, domain)}")
    
    # Equivalence relations
    print("\n4. EQUIVALENCE RELATIONS")
    print("-" * 40)
    
    same_parity = lambda a, b: (a % 2) == (b % 2)
    domain = list(range(10))
    
    props = SetTheory.check_relation_properties(same_parity, domain)
    print(f"Same parity relation properties: {props}")
    
    classes = SetTheory.equivalence_classes(same_parity, domain)
    print(f"Equivalence classes: {classes}")
    
    # Countability
    print("\n5. COUNTABILITY DEMONSTRATION")
    print("-" * 40)
    SetTheory.demonstrate_countability()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATIONS COMPLETE")
    print("=" * 60)

```

---

## 🤖 ML Applications

| Set Theory Concept | ML Application | Example |
|:-------------------|:---------------|:--------|
| **Set operations** | Data filtering, SQL joins | `df1.merge(df2, how='inner')` |
| **Cartesian product** | Grid search, attention | Hyperparameter combinations |
| **Bijection** | Normalizing flows | Invertible transformations |
| **Equivalence relation** | Clustering | Points in same cluster |
| **σ-algebra** | Probability spaces | Random variable definition |
| **Cardinality** | Discrete vs continuous | PMF vs PDF |
| **Power set** | Feature selection | All feature subsets |

---

## 📚 Resources

| Type | Title | Link |
|:-----|:------|:-----|
| 📖 Book | Naive Set Theory (Halmos) | Classic introduction |
| 📖 Book | Set Theory and Logic (Stoll) | Comprehensive |
| 📖 Book | A Transition to Advanced Mathematics | [Amazon](https://www.amazon.com/Transition-Advanced-Mathematics-Douglas-Smith/dp/0495562025) |
| 🎥 Video | MIT OCW - Set Theory | [MIT](https://ocw.mit.edu) |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[📐 Proof Techniques](../02_proof_techniques/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 3 of 6**<br>
**🔢 Set Theory**

</td>
<td align="right" width="33%">

➡️ **Next**<br>
[🔀 Logic](../04_logic/README.md)

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
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=24,25,26&height=100&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=4ECDC4&center=true&vCenter=true&width=600&lines=Made+with+❤️+by+Gaurav+Goswami;Part+of+ML+Researcher+Foundations+Series" alt="Footer" />
</p>

<p align="center">
  <a href="https://github.com/Gaurav14cs17">
    <img src="https://img.shields.io/badge/GitHub-Gaurav14cs17-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>
