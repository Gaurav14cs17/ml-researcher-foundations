<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FFE66D&height=120&section=header&text=🔀%20Logic&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-4_of_6-FFE66D?style=for-the-badge&logo=bookstack&logoColor=black" alt="Section"/>
  <img src="https://img.shields.io/badge/Reading-18_min-00C853?style=for-the-badge&logo=clock&logoColor=white" alt="Reading Time"/>
  <img src="https://img.shields.io/badge/Level-Intermediate-FF9800?style=for-the-badge&logo=signal&logoColor=white" alt="Difficulty"/>
</p>

<p align="center">
  <i>Formal reasoning for reading and writing proofs</i>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**✍️ Author:** [Gaurav Goswami](https://github.com/Gaurav14cs17)  
**📅 Published:** December 2024  
**🏷️ Tags:** `logic` `propositional-logic` `predicate-logic` `quantifiers` `boolean`

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01-mathematical-thinking/README.md) → [Proof Techniques](../02-proof-techniques/README.md) → [Set Theory](../03-set-theory/README.md) → Logic → [Asymptotic Analysis](../05-asymptotic-analysis/README.md) → [Numerical Computation](../06-numerical-computation/README.md)

---

## 📌 TL;DR

Logic is the foundation of all mathematical reasoning. This article covers:
- **Propositional Logic** — AND, OR, NOT, IMPLIES truth tables
- **Predicate Logic** — ∀ (for all) and ∃ (exists) quantifiers
- **Rules of Inference** — Modus ponens, modus tollens
- **Boolean Functions** — How neural networks compute logic

> [!WARNING]
> Quantifier order matters! `∀x ∃y P(x,y)` ≠ `∃y ∀x P(x,y)`. This is a common source of confusion in ML papers.

---

## 📚 What You'll Learn

- [ ] Read and construct truth tables
- [ ] Understand quantifiers (∀, ∃) in ML theorems
- [ ] Apply De Morgan's laws for negation
- [ ] Use inference rules in proofs
- [ ] Connect boolean logic to neural networks

---

## 📑 Table of Contents

- [Visual Overview](#-visual-overview)
- [Logic Symbols Quick Reference](#-logic-symbols-quick-reference)
- [Propositional Logic](#1-propositional-logic-complete-truth-tables)
- [Predicate Logic and Quantifiers](#3-predicate-logic-and-quantifiers)
- [Rules of Inference](#4-rules-of-inference)
- [Boolean Functions and Neural Networks](#7-boolean-functions-and-neural-networks)
- [Resources](#-references)
- [Navigation](#-navigation)

---

## 🎯 Visual Overview

<img src="./images/logic-operators.svg" width="100%">

*Caption: This comprehensive diagram shows truth tables for logical operators (AND, OR, NOT, IMPLIES), logic gates used in neural networks, and quantifiers (∀ for all, ∃ exists). Understanding these is crucial for reading ML research papers and formal specifications.*

---

## 📐 Mathematical Foundations

### Propositional Logic

**Truth Table for Implication (→):**

| P | Q | P → Q | ¬P ∨ Q |
|:---:|:---:|:---------:|:---------------:|
| T | T | ✅ T | ✅ T |
| T | F | ❌ F | ❌ F |
| F | T | ✅ T | ✅ T |
| F | F | ✅ T | ✅ T |

> [!TIP]
> P → Q is equivalent to ¬P ∨ Q — they have the same truth table!

### Predicate Logic

```
┌────────────────────────┐    ┌────────────────────────┐
│      ∀ FOR ALL         │    │       ∃ EXISTS         │
│                        │    │                        │
│    ∀x P(x)             │    │    ∃x P(x)             │
│    Every x satisfies P │    │    Some x satisfies P  │
└────────────────────────┘    └────────────────────────┘
```

**Example:**
```
∀x (NeuralNet(x) → HasWeights(x))
```
*"Every neural network has weights"*

### Proof Techniques Overview

| Technique | Pattern | Use When |
|:---------:|:--------|:---------|
| 🎯 **Direct** | P → Q | Can derive directly |
| ⚡ **Contradiction** | Assume ¬Q, find ⊥ | Hard to prove directly |
| 🔄 **Induction** | P(0) + P(k) → P(k+1) | Natural numbers |

---

## 📂 Topics in This Folder

| File | Topic | Application |
|------|-------|-------------|

---

## 🎯 Logic Symbols Quick Reference

| Symbol | Name | Meaning | Example |
|:------:|------|---------|:-----:|
| ∧ | AND | Both true | P ∧ Q |
| ∨ | OR | At least one true | P ∨ Q |
| ¬ | NOT | Negation | ¬P |
| → | IMPLIES | If...then | P → Q |
| ↔ | IFF | If and only if | P ↔ Q |
| ∀ | FOR ALL | Universal | ∀x P(x) |
| ∃ | EXISTS | Existential | ∃x P(x) |

**Quick Examples:**

```
∀x ∈ ℝ: x² ≥ 0
∃x ∈ ℕ: x > 100
```

---

## 🌍 ML Applications

| Logic Concept | ML Application |
|---------------|----------------|
| Boolean AND/OR | Attention masking |
| Predicate logic | Type systems, specs |
| ∀x P(x) | "For all inputs, model outputs..." |
| ∃x P(x) | "There exists an adversarial example..." |

---

## 💻 Negation Rules (Important!)

```
🔄 De Morgan's Laws              📊 Quantifier Negation
┌────────────────────────┐      ┌────────────────────────┐
│ ¬(P ∧ Q)  =  ¬P ∨ ¬Q   │      │ ¬∀x P(x)  =  ∃x ¬P(x)  │
│ ¬(P ∨ Q)  =  ¬P ∧ ¬Q   │      │ ¬∃x P(x)  =  ∀x ¬P(x)  │
└────────────────────────┘      └────────────────────────┘
```

| Original | Negation | Meaning |
|:--------:|:--------:|:--------|
| ¬(P ∧ Q) | ¬P ∨ ¬Q | Not both → at least one false |
| ¬(P ∨ Q) | ¬P ∧ ¬Q | Not either → both false |
| ¬∀x P(x) | ∃x ¬P(x) | Not all → some don't |
| ¬∃x P(x) | ∀x ¬P(x) | None exist → all don't |

> [!TIP]
> **Example:** "Not all neural networks overfit"  
> = ¬∀x Overfit(x)  
> = ∃x ¬Overfit(x)  
> = "There exists a neural network that doesn't overfit" ✅

---

## 📐 DETAILED MATHEMATICAL THEORY

### 1. Propositional Logic: Complete Truth Tables

**Basic Connectives:**

```
🔌 Logic Gates
┌─────────────────────────────────────────────────────────┐
│   ∧ AND    ∨ OR    ¬ NOT    → IMPLIES    ↔ IFF        │
└─────────────────────────────────────────────────────────┘
```

<table>
<tr>
<td>

**AND (∧)**

| P | Q | P∧Q |
|:-:|:-:|:---:|
| T | T | ✅ T |
| T | F | ❌ F |
| F | T | ❌ F |
| F | F | ❌ F |

</td>
<td>

**OR (∨)**

| P | Q | P∨Q |
|:-:|:-:|:---:|
| T | T | ✅ T |
| T | F | ✅ T |
| F | T | ✅ T |
| F | F | ❌ F |

</td>
<td>

**NOT (¬)**

| P | ¬P |
|:-:|:--:|
| T | ❌ F |
| F | ✅ T |

</td>
</tr>
</table>

<table>
<tr>
<td>

**IMPLIES (→)**

| P | Q | P→Q |
|:-:|:-:|:---:|
| T | T | ✅ T |
| T | F | ❌ F |
| F | T | ✅ T |
| F | F | ✅ T |

</td>
<td>

**IFF (↔)**

| P | Q | P↔Q |
|:-:|:-:|:---:|
| T | T | ✅ T |
| T | F | ❌ F |
| F | T | ❌ F |
| F | F | ✅ T |

</td>
</tr>
</table>

> [!TIP]
> **Key Equivalences:**
> - P → Q ≡ ¬P ∨ Q ("If P then Q" = "Not P or Q")
> - P ↔ Q ≡ (P → Q) ∧ (Q → P)

---

**Exclusive OR (XOR):**

| P | Q | P ⊕ Q | Meaning |
|:-:|:-:|:-----:|:--------|
| T | T | ❌ F | Both true → false |
| T | F | ✅ T | Different → true |
| F | T | ✅ T | Different → true |
| F | F | ❌ F | Both false → false |

```
P ⊕ Q ≡ (P ∨ Q) ∧ ¬(P ∧ Q) ≡ (P ∧ ¬Q) ∨ (¬P ∧ Q)
```

> [!NOTE]
> **ML Application:** Parity functions, feature interactions

---

### 2. Logical Equivalences (Proof Techniques)

```
📜 Key Logical Laws
┌──────────────────────────────────────────────────────────────┐
│  🔄 De Morgan    ──▶  ¬(P∧Q) ≡ ¬P∨¬Q                        │
│  📐 Distributive ──▶  P∧(Q∨R) ≡ (P∧Q)∨(P∧R)                 │
│  ➡️ Implication  ──▶  P→Q ≡ ¬P∨Q                            │
└──────────────────────────────────────────────────────────────┘
```

**De Morgan's Laws:**

| Original | Equivalent | In Words |
|:--------:|:----------:|:---------|
| ¬(P ∧ Q) | ¬P ∨ ¬Q | Not both → at least one false |
| ¬(P ∨ Q) | ¬P ∧ ¬Q | Not either → both false |

> **Example:** "Not (raining AND cold)" = "Not raining OR not cold"

---

**Distributive Laws:**

```
P ∧ (Q ∨ R) ≡ (P ∧ Q) ∨ (P ∧ R)
P ∨ (Q ∧ R) ≡ (P ∨ Q) ∧ (P ∨ R)
```

> [!TIP]
> **ML Use:** Simplifying boolean expressions in code

---

**Implication Laws:**

| Law | Formula | Use Case |
|:---:|:-------:|:---------|
| Definition | P → Q ≡ ¬P ∨ Q | Convert to OR |
| Contrapositive | P → Q ≡ ¬Q → ¬P | Easier proofs |
| Negation | ¬(P → Q) ≡ P ∧ ¬Q | Disprove implication |

> [!TIP]
> **Contrapositive Power:** To prove P → Q, prove ¬Q → ¬P instead (often easier!)

---

**Tautologies & Contradictions:**

| Type | Examples | Property |
|:----:|:---------|:--------:|
| ✅ **Tautology** | P ∨ ¬P (Excluded Middle) | Always TRUE |
| ✅ **Tautology** | ¬(P ∧ ¬P) (Non-Contradiction) | Always TRUE |
| ❌ **Contradiction** | P ∧ ¬P | Always FALSE |

---

### 3. Predicate Logic and Quantifiers

```
🔢 Quantifiers
┌────────────────────────────────────────────────┐
│                                                │
│   ∀ Universal     ◀──Negation──▶   ∃ Existential   │
│   "For all"                        "There exists"  │
│                                                │
└────────────────────────────────────────────────┘
```

**Universal Quantifier (∀):**

| Symbol | Meaning | ML Example |
|:------:|:--------|:-----------|
| ∀x P(x) | "For all x, P(x) is true" | ∀x ∈ Train: Loss(x) < ε |

**Negation:** ¬(∀x P(x)) ≡ ∃x ¬P(x)

---

**Existential Quantifier (∃):**

| Symbol | Meaning | ML Example |
|:------:|:--------|:-----------|
| ∃x P(x) | "There exists x with P(x)" | ∃x: Adversarial(x) ∧ Fools(x) |

**Negation:** ¬(∃x P(x)) ≡ ∀x ¬P(x)

---

**⚠️ Quantifier Order Matters!**

<table>
<tr>
<td width="50%">

`∀x ∃y: P(x, y)`

*"For each x, there exists a y"*

**Example:** Everyone likes *some* food ✅
<br/>(different food per person)

</td>
<td width="50%">

`∃y ∀x: P(x, y)`

*"There exists one y for all x"*

**Example:** There's a food *everyone* likes ❓
<br/>(single universal food)

</td>
</tr>
</table>

> [!WARNING]
> **No Free Lunch Theorem:**
> - ∀data ∃model: Fits ✅ (trivially true)
> - ∃model ∀data: Fits ❌ (impossible!)

---

**Multiple Quantifiers - The Limit Definition:**

```
∀ε > 0, ∃N: ∀n > N, |aₙ - L| < ε
```

> [!NOTE]
> **ML Application:** Convergence proofs for optimization algorithms
> ∀ε > 0, ∃T: ∀t > T, ‖θₜ - θ*‖ < ε

---

### 4. Rules of Inference

```
⚖️ Core Inference Rules
┌────────────────────────────────────────────────────────────────────┐
│  Modus Ponens   Modus Tollens   Hypothetical Syll.   Disjunctive S.│
└────────────────────────────────────────────────────────────────────┘
```

<table>
<tr>
<th>Rule</th>
<th>Premises → Conclusion</th>
<th>ML Example</th>
</tr>
<tr>
<td><b>🔵 Modus Ponens</b></td>
<td>P → Q, P<br/>∴ Q</td>
<td>If overfits → loss ↑. Overfits. ∴ Loss ↑ ✅</td>
</tr>
<tr>
<td><b>🔴 Modus Tollens</b></td>
<td>P → Q, ¬Q<br/>∴ ¬P</td>
<td>If LR high → diverges. No diverge. ∴ LR ok ✅</td>
</tr>
<tr>
<td><b>🟣 Hypothetical Syllogism</b></td>
<td>P → Q, Q → R<br/>∴ P → R</td>
<td>Data → Generalization → Accuracy ✅</td>
</tr>
<tr>
<td><b>🟡 Disjunctive Syllogism</b></td>
<td>P ∨ Q, ¬P<br/>∴ Q</td>
<td>Underfits OR Overfits. Not underfit. ∴ Overfits ✅</td>
</tr>
</table>

---

**Quantifier Rules:**

| Rule | Pattern | ML Example |
|:----:|:--------|:-----------|
| **Universal Instantiation** | ∀x P(x) ⟹ P(c) | All NNs need gradients → ResNet needs gradients ✅ |
| **Universal Generalization** | P(c) for arbitrary c ⟹ ∀x P(x) | Backprop works for any net → Works for all ✅ |

---

### 5. Proof Techniques Using Logic

```
📐 Proof Techniques
┌──────────────────────────────────────────────────────────────────────┐
│  ➡️ Direct Proof      ──▶  Assume P, derive Q                       │
│  💥 Contradiction     ──▶  Assume ¬Q, find contradiction            │
│  ↩️ Contrapositive    ──▶  Prove ¬Q → ¬P                            │
└──────────────────────────────────────────────────────────────────────┘
```

<details>
<summary><b>➡️ Direct Proof (P → Q)</b></summary>

**Steps:**
1. Assume P is true
2. Using logic rules, derive Q
3. Conclude P → Q

**Example: Convex functions have global minima**

```
Given: f convex, x* local min
By convexity: f(y) ≥ f(x*) + ∇f(x*)ᵀ(y-x*) for all y
At local min: ∇f(x*) = 0
Therefore: f(y) ≥ f(x*) for all y
⟹ x* is global min ✓
```

</details>

<details>
<summary><b>💥 Proof by Contradiction (P → Q)</b></summary>

**Steps:**
1. Assume P is true and Q is false (P ∧ ¬Q)
2. Derive a contradiction
3. Conclude P → Q must be true

**Example: √2 is irrational**

```
Assume √2 = p/q (rational, lowest terms)
Then 2 = p²/q², so 2q² = p². Therefore p² is even, so p is even: p = 2k
Then 2q² = 4k², so q² = 2k². Therefore q is also even.
But then p/q not in lowest terms! Contradiction ✅
```

</details>

<details>
<summary><b>📋 Proof by Cases</b></summary>

**Template:**
1. Break P into cases: P ≡ P₁ ∨ P₂ ∨ ...
2. Prove P₁ → Q, P₂ → Q, ...
3. Conclude P → Q

**Example:** Prove |x · y| = |x| · |y|

| Case | Condition | Result |
|:----:|:----------|:------:|
| 1 | x ≥ 0, y ≥ 0 | ✅ Trivial |
| 2 | x < 0, y ≥ 0 | ✅ Check |
| 3 | x ≥ 0, y < 0 | ✅ Check |
| 4 | x < 0, y < 0 | ✅ Check |

All cases covered ∎

</details>

---

### 6. Formal Logic in ML Research Papers

```
📖 Reading ML Theorems
┌────────────────────────────────────────────┐
│    Theorem Statement                       │
│           │                                │
│           ▼                                │
│       Assumptions                          │
│           │                                │
│           ▼                                │
│       Conclusions                          │
└────────────────────────────────────────────┘
```

**Reading Theorems:**

```
∀ε > 0, ∃N: ∀n ≥ N, |xₙ - L| < ε
```

> **Translation:** "For any desired accuracy ε (no matter how small), there exists a step N such that after step N, the sequence is within ε of L"
>
> This is **convergence**!

---

**Assumptions and Conditions:**

| Pattern | Meaning | Danger |
|:--------|:--------|:------:|
| "Suppose f is L-Lipschitz..." | ∀ conditions stated, holds | ⚠️ |
| Missing condition | Theorem may fail! | 🚨 |

---

**Existence Statements:**

| Symbol | Meaning | What It Doesn't Say |
|:------:|:--------|:--------------------|
| ∃θ* optimal | At least one exists | ❌ Not unique |
| | | ❌ Not how to find it |

> [!TIP]
> **Constructive proof** = shows how to find it. **Non-constructive** = proves existence only.

---

### 7. Boolean Functions and Neural Networks

**Boolean Gates as Neural Networks:**

```
AND Gate (∧)           OR Gate (∨)            NOT Gate (¬)
┌─────────────────┐   ┌─────────────────┐    ┌──────────────┐
│ x₁ ─┬─▶ σ ─▶out │   │ x₁ ─┬─▶ σ ─▶out │    │ x ─▶ σ ─▶out │
│ x₂ ─┘           │   │ x₂ ─┘           │    │              │
│ w=1,1, b=-1.5   │   │ w=1,1, b=-0.5   │    │ w=-1, b=0.5  │
└─────────────────┘   └─────────────────┘    └──────────────┘
```

| Gate | Formula | (0,0) | (0,1) | (1,0) | (1,1) |
|:----:|:-------:|:-------:|:-------:|:-------:|:-------:|
| **AND** | σ(x₁ + x₂ - 1.5) | 0 | 0 | 0 | ✅ 1 |
| **OR** | σ(x₁ + x₂ - 0.5) | 0 | ✅ 1 | ✅ 1 | ✅ 1 |
| **NOT** | σ(-x + 0.5) | ✅ 1 | 0 | — | — |

**XOR Problem (Not Linearly Separable):**

```
Input Layer          Hidden Layer              Output
┌───────────┐      ┌──────────────┐      ┌──────────────────┐
│    x₁     │─────▶│   h₁ = OR    │─────▶│                  │
│           │  ┌──▶│              │      │  XOR = h₁ - h₂   │
│    x₂     │──┤   │   h₂ = AND   │─────▶│                  │
└───────────┘  └──▶└──────────────┘      └──────────────────┘
```

> [!IMPORTANT]
> **XOR requires 2 layers!** Single perceptron cannot learn XOR — this was the famous limitation discovered in 1969.

**Universal Boolean Function:**

Any boolean function f: {0,1}ⁿ → {0,1} can be represented as:

| Form | Pattern | Example |
|:----:|:--------|:--------|
| **DNF** | OR of ANDs | (x₁ ∧ x₂) ∨ (¬x₁ ∧ x₃) |
| **CNF** | AND of ORs | (x₁ ∨ x₂) ∧ (¬x₁ ∨ x₃) |

**Neural Network Implementation:**

| Layer | Neurons | Purpose |
|:-----:|:-------:|:--------|
| Input | n | Features |
| Hidden | 2ⁿ | One per input combo ⚠️ |
| Output | 1 | Result |

> [!WARNING]
> Exponentially many neurons needed! This is why we need **deep** networks.

---

### 8. Fuzzy Logic and Soft Decisions

```
🔲 Classical Logic          🌈 Fuzzy Logic
┌──────────────────┐       ┌──────────────────┐
│  P ∈ {0, 1}      │──────▶│  P ∈ [0, 1]      │
└──────────────────┘ soften└──────────────────┘
```

| Logic Type | Values | Example: "Temperature is high" |
|:----------:|:------:|:-------------------------------|
| **Classical** | {0, 1} | 45°C → TRUE |
| **Fuzzy** | [0, 1] | 30°C: 0.3, 40°C: 0.7, 50°C: 1.0 |

---

**Fuzzy Operations:**

| Operation | Formula | Example |
|:---------:|:--------|:--------|
| **AND** | μ(A ∧ B) = min(μ(A), μ(B)) | Hot(0.7) ∧ Humid(0.5) = 0.5 |
| **OR** | μ(A ∨ B) = max(μ(A), μ(B)) | Hot(0.7) ∨ Humid(0.5) = 0.7 |
| **NOT** | μ(¬A) = 1 - μ(A) | ¬Hot(0.7) = 0.3 |

**ML Connection:**

> [!TIP]
> **Neural network outputs are fuzzy truth values!**
> - p(class=cat) = 0.8 → "80% true that this is a cat"
> - **Softmax** converts logits to fuzzy values: Σᵢpᵢ = 1
> - **Attention weights** are fuzzy memberships: "How much does this word attend to that word?"

---

### 9. Modal Logic (Advanced)

**Necessity and Possibility:**

| Symbol | Name | Meaning | ML Interpretation |
|:------:|:-----|:--------|:------------------|
| □P | Necessity | "P is necessarily true" | "P holds for ALL distributions" |
| ◇P | Possibility | "P is possibly true" | "P holds for SOME distribution" |

**Relationship:** ◇P ≡ ¬□¬P ("P is possible" = "It's not necessary that P is false")

<details>
<summary><b>🤖 ML Examples</b></summary>

| Statement | Meaning |
|:----------|:--------|
| □(More capacity → Lower train loss) | Always true: capacity reduces training loss |
| ◇(Model overfits) | It's possible for the model to overfit |

</details>

---

### 10. Applications in ML Specifications

**Pre/Post Conditions:**

```
{P}  function  {Q}
```

| Term | Meaning | Example |
|:----:|:--------|:--------|
| **Precondition** P | Assumption before | x ∈ [0,1] |
| **Postcondition** Q | Guarantee after | output ∈ (0,1) |

> **Example:** `{x ∈ [0,1]}` sigmoid(x) `{output ∈ (0,1)}`

---

**Loop Invariants:**

```
Invariant I          Condition?           Body              Result
True initially  ──▶  [Yes]  ──▶  Preserves I  ──┐     I ∧ ¬condition
                       │                        │         = Result ✅
                       ◀────────────────────────┘
                     [No] ──▶ Result
```

<details>
<summary><b>🔄 GD Example</b></summary>

| Component | Value |
|:----------|:------|
| **Invariant** | θₜ is improving solution |
| **Condition** | ‖∇L‖ > ε |
| **After loop** | θₜ is good ∧ ‖∇L‖ ≤ ε ✅ |

</details>

---

**Correctness Proofs:**

| Step | Action |
|:----:|:-------|
| 1 | Define pre/postconditions |
| 2 | Prove invariants preserved |
| 3 | Prove termination |
| 4 | Conclude correctness |

> **Example:** Prove backprop computes gradients correctly → Use induction on depth ✅

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Introduction to Logic | [OpenLogic](https://openlogicproject.org/) |
| 📖 | Logic in CS | Huth & Ryan |
| 🎥 | Discrete Math Logic | [YouTube](https://www.youtube.com/watch?v=itrXYg41-V0) |
| 🇨🇳 | 数理逻辑基础 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 逻辑与证明 | [B站](https://www.bilibili.com/video/BV164411b7dx) |


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[🔢 Set Theory](../03-set-theory/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 4 of 6**<br>
**Logic**

</td>
<td align="right" width="33%">

➡️ **Next**<br>
[⏱️ Asymptotic Analysis](../05-asymptotic-analysis/README.md)

</td>
</tr>
</table>

---

<!-- Animated Footer -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FFE66D&height=80&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <a href="../README.md"><img src="https://img.shields.io/badge/📚_Part_of-ML_Researcher_Foundations-FFE66D?style=for-the-badge" alt="Series"/></a>
</p>

<p align="center">
  <sub>Made with ❤️ by <a href="https://github.com/Gaurav14cs17">Gaurav Goswami</a></sub>
</p>
