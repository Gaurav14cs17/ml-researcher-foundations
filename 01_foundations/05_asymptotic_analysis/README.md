<!-- Animated Header -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=40&pause=1000&color=00D4AA&center=true&vCenter=true&width=800&lines=⏱️+Asymptotic+Analysis;Big-O+and+Algorithm+Complexity" alt="Asymptotic Analysis" />
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=18,19,20&height=180&section=header&text=Asymptotic%20Analysis&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Big-O%20•%20Big-Ω%20•%20Big-Θ%20•%20Complexity%20Classes&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05_of_06-00D4AA?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Topics-6_Concepts-FF6B6B?style=for-the-badge&logo=buffer&logoColor=white" alt="Topics"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-6C63FF?style=for-the-badge&logo=github&logoColor=white" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-4ECDC4?style=for-the-badge&logo=calendar&logoColor=white" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

**🏠 [Home](../README.md)** · **📚 Series:** [Mathematical Thinking](../01_mathematical_thinking/README.md) → [Proof Techniques](../02_proof_techniques/README.md) → [Set Theory](../03_set_theory/README.md) → [Logic](../04_logic/README.md) → Asymptotic Analysis → [Numerical Computation](../06_numerical_computation/README.md)

---

## 📌 TL;DR

Asymptotic analysis tells you **how algorithms scale**. Essential for ML model selection:

- **Big-O (O)** — Upper bound on growth: "at most this fast"
- **Big-Omega (Ω)** — Lower bound: "at least this fast"  
- **Big-Theta (Θ)** — Tight bound: "exactly this fast"
- **little-o (o)** — Strictly slower growth

> [!IMPORTANT]
> **Transformers are O(n²) in sequence length!**
> - n=512: 262K ops | n=8192: 67M ops (256× more!)
> - This is why Flash Attention and linear attention are hot research areas.

---

## 📚 What You'll Learn

- [ ] Define and prove Big-O, Big-Ω, Big-Θ bounds formally
- [ ] Analyze time and space complexity of algorithms
- [ ] Understand ML model complexities (Transformers, CNNs, RNNs)
- [ ] Apply Master Theorem for recursive algorithms
- [ ] Compare complexity classes and their implications

---

## 📑 Table of Contents

1. [Complexity Hierarchy](#-complexity-hierarchy)
2. [Big-O Notation](#1-big-o-notation)
3. [Big-Omega and Big-Theta](#2-big-omega-and-big-theta)
4. [Little-o and Little-omega](#3-little-o-and-little-omega)
5. [ML Model Complexities](#4-ml-model-complexities)
6. [Master Theorem](#5-master-theorem)
7. [Key Formulas Summary](#-key-formulas-summary)
8. [Common Mistakes](#-common-mistakes--pitfalls)
9. [Code Implementations](#-code-implementations)
10. [Resources](#-resources)

---

## 📊 Complexity Hierarchy

```
FAST ◀--------------------------------------------------------▶ SLOW/AVOID

O(1)   O(log n)   O(n)   O(n log n)   O(n²)   O(n³)   O(2ⁿ)   O(n!)
 |         |        |         |          |        |        |       |
Hash    Binary   Array    Merge      Attention  MatMul  Brute   Perms
lookup  search   scan     sort       (naive)    (naive)  force
```

| Complexity | Name | n=1000 | Example |
|:----------:|:-----|:------:|:--------|
| O(1) | Constant | 1 | Hash lookup, array access |
| O(log n) | Logarithmic | 10 | Binary search |
| O(n) | Linear | 1,000 | Array scan, softmax |
| O(n log n) | Linearithmic | 10,000 | Merge sort, FFT |
| O(n²) | Quadratic | 1,000,000 | **Transformer attention** |
| O(n³) | Cubic | 10⁹ | Naive matrix multiply |
| O(2ⁿ) | Exponential | ∞ | Subset enumeration |
| O(n!) | Factorial | ∞ | All permutations |

---

## 1. Big-O Notation

### 📖 Definition

> **Big-O:** f(n) = O(g(n)) means there exist constants c > 0 and n₀ such that:
> $$
\forall n \geq n_0: f(n) \leq c \cdot g(n)

```math
> 
> *"f grows at most as fast as g"*

### 📐 Formal Proof Template

To prove f(n) = O(g(n)):
1. Find constants c > 0 and n₀
2. Show f(n) ≤ c·g(n) for all n ≥ n₀

### 📝 Example 1: Prove 3n² + 5n + 2 = O(n²)

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | f(n) = 3n² + 5n + 2 | Given |
| 2 | For n ≥ 1: n ≤ n² and 1 ≤ n² | Basic inequality |
| 3 | 3n² + 5n + 2 ≤ 3n² + 5n² + 2n² | Substitution |
| 4 | = 10n² | Simplify |
| 5 | Choose c = 10, n₀ = 1 | Constants |
| 6 | f(n) ≤ 10·n² for all n ≥ 1 | ∎ |

### 📝 Example 2: Prove log(n!) = O(n log n)

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | log(n!) = log(1·2·3·...·n) = Σᵢ log(i) | Definition |
| 2 | Σᵢ₌₁ⁿ log(i) ≤ Σᵢ₌₁ⁿ log(n) = n·log(n) | log(i) ≤ log(n) |
| 3 | So log(n!) ≤ n·log(n) | |
| 4 | log(n!) = O(n log n) with c=1, n₀=1 | ∎ |

### 📝 Example 3: Big-O Rules

| Rule | Statement |
|:-----|:----------|
| **Sum** | O(f) + O(g) = O(max(f,g)) |
| **Product** | O(f) · O(g) = O(f·g) |
| **Constants** | c·O(f) = O(f) |
| **Transitivity** | f = O(g), g = O(h) ⟹ f = O(h) |

---

## 2. Big-Omega and Big-Theta

### 📖 Definitions

> **Big-Omega (Ω):** f(n) = Ω(g(n)) means there exist c > 0 and n₀ such that:
> $$
\forall n \geq n_0: f(n) \geq c \cdot g(n)
```

> *"f grows at least as fast as g"*

> **Big-Theta (Θ):** f(n) = Θ(g(n)) iff f(n) = O(g(n)) AND f(n) = Ω(g(n))
> *"f grows exactly as fast as g"*

### 📐 Proof: 3n² + 5n = Θ(n²)

**Upper bound (O):** Shown in Example 1: 3n² + 5n ≤ 10n² for n ≥ 1

**Lower bound (Ω):**

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | 3n² + 5n ≥ 3n² | 5n ≥ 0 for n ≥ 0 |
| 2 | 3n² + 5n ≥ 3·n² | |
| 3 | Choose c = 3, n₀ = 0 | |
| 4 | f(n) = Ω(n²) | ∎ |

**Conclusion:** Since f(n) = O(n²) and f(n) = Ω(n²), we have f(n) = Θ(n²) ∎

### 📊 Comparison Table

| Notation | Meaning | Analogy |
|:--------:|:--------|:--------|
| f = O(g) | f ≤ c·g eventually | f ≤ g |
| f = Ω(g) | f ≥ c·g eventually | f ≥ g |
| f = Θ(g) | Both O and Ω | f = g (same order) |
| f = o(g) | f/g → 0 | f < g (strictly slower) |
| f = ω(g) | f/g → ∞ | f > g (strictly faster) |

---

## 3. Little-o and Little-omega

### 📖 Definition

> **Little-o:** f(n) = o(g(n)) means:
> $$
\lim_{n \to \infty} \frac{f(n)}{g(n)} = 0

```math
> *"f grows strictly slower than g"*

### 📝 Examples
```

n = o(n²)           # n grows slower than n²
log n = o(n)        # log grows slower than linear
n² = o(2ⁿ)          # polynomial < exponential
n^k = o(n^{k+1})    # lower power < higher power
```

### 📐 Proof: n = o(n²)

| Step | Statement | Justification |
|:----:|:----------|:--------------|
| 1 | Consider lim_{n→∞} n/n² | Definition |
| 2 | = lim_{n→∞} 1/n | Simplify |
| 3 | = 0 | Limit |
| 4 | n = o(n²) | ∎ |

### 📖 Growth Rate Hierarchy

```
1 ≺ log log n ≺ log n ≺ n^ε ≺ n ≺ n log n ≺ n² ≺ ... ≺ 2^n ≺ n!

Where f ≺ g means f = o(g) (f grows strictly slower)
```

---

## 4. ML Model Complexities

### 📊 Model Comparison

| Model | Time Complexity | Space Complexity |
|:------|:---------------:|:----------------:|
| **Transformer Attention** | O(n²d) | O(n²) or O(n) |
| **Flash Attention** | O(n²d) | O(n) 🔥 |
| **Linear Attention** | O(nd²) | O(nd) |
| **CNN (per layer)** | O(k²·c·h·w) | O(c·h·w) |
| **RNN (per step)** | O(h²) | O(h) |
| **GNN (message passing)** | O(\|E\|·d) | O(\|V\|·d) |

### 📐 Transformer Attention Derivation

```
Attention(Q, K, V) = softmax(QK^T / √d) V

Where: Q, K, V ∈ ℝ^{n×d}

Step 1: QK^T                    → O(n²d) time, O(n²) space for scores
Step 2: softmax(·/√d)           → O(n²) time
Step 3: Multiply by V           → O(n²d) time

Total: O(n²d) time, O(n²) space (for attention matrix)
```

### 📊 Scaling Impact

| Sequence Length n | O(n²) Operations | Relative |
|:-----------------:|:----------------:|:--------:|
| 512 | 262,144 | 1× |
| 2,048 | 4,194,304 | 16× |
| 8,192 | 67,108,864 | **256×** |
| 32,768 | 1,073,741,824 | **4,096×** |

> [!IMPORTANT]
> This is why GPT-3 had 2K context, and why Flash Attention, Linear Attention, and sparse attention are critical research areas!

---

## 5. Master Theorem

### 📖 Definition

For recurrences of the form: T(n) = aT(n/b) + f(n)

| Case | Condition | Result |
|:----:|:----------|:-------|
| 1 | f(n) = O(n^{log_b(a) - ε}) | T(n) = Θ(n^{log_b(a)}) |
| 2 | f(n) = Θ(n^{log_b(a)}) | T(n) = Θ(n^{log_b(a)} log n) |
| 3 | f(n) = Ω(n^{log_b(a) + ε}) | T(n) = Θ(f(n)) |

### 📝 Example: Merge Sort

```
T(n) = 2T(n/2) + O(n)

a = 2, b = 2, f(n) = O(n)
log_b(a) = log_2(2) = 1
n^{log_b(a)} = n^1 = n
f(n) = O(n) = Θ(n^1)

Case 2 applies: T(n) = Θ(n log n)
```

### 📝 Example: Binary Search

```
T(n) = T(n/2) + O(1)

a = 1, b = 2, f(n) = O(1)
log_b(a) = log_2(1) = 0
n^0 = 1
f(n) = Θ(1) = Θ(n^0)

Case 2 applies: T(n) = Θ(log n)
```

---

## 📊 Key Formulas Summary

| Notation | Definition | Intuition |
|:---------|:-----------|:----------|
| f = O(g) | ∃c,n₀: f(n) ≤ c·g(n) ∀n≥n₀ | At most g |
| f = Ω(g) | ∃c,n₀: f(n) ≥ c·g(n) ∀n≥n₀ | At least g |
| f = Θ(g) | O(g) ∧ Ω(g) | Exactly g |
| f = o(g) | lim f(n)/g(n) = 0 | Strictly < g |
| f = ω(g) | lim f(n)/g(n) = ∞ | Strictly > g |

---

## ⚠️ Common Mistakes & Pitfalls

### Mistake 1: Dropping Constants Incorrectly

```
❌ WRONG: O(2n) = O(n) means 2n ≈ n

✅ RIGHT: O(2n) = O(n) means they have SAME GROWTH RATE
          In practice, 2n is still 2× slower than n!
```

### Mistake 2: Comparing Different Variables

```
❌ WRONG: O(n²) is always worse than O(n)

✅ RIGHT: Depends on what n means!
          O(n²) in sequence length vs O(n) in dimension
          might have very different implications
```

### Mistake 3: Ignoring Space Complexity

```
❌ WRONG: Focus only on time complexity

✅ RIGHT: GPU memory is limited!
          O(n²) space for attention means 8K sequence on 40GB A100
```

---

## 💻 Code Implementations

```python
import numpy as np
import time
from typing import Callable

def measure_complexity(f: Callable, sizes: list, name: str):
    """Empirically measure algorithm complexity."""
    times = []
    for n in sizes:
        start = time.perf_counter()
        f(n)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    print(f"\n{name}:")
    print(f"{'n':>10} | {'Time (ms)':>12} | {'Ratio':>8}")
    print("-" * 35)
    for i, (n, t) in enumerate(zip(sizes, times)):
        ratio = times[i] / times[0] if i > 0 else 1.0
        print(f"{n:>10} | {t*1000:>12.3f} | {ratio:>8.2f}")

# O(n) - Linear
def linear_example(n):
    return sum(range(n))

# O(n²) - Quadratic (like attention)
def quadratic_example(n):
    total = 0
    for i in range(n):
        for j in range(n):
            total += i + j
    return total

# O(n log n) - Like merge sort
def nlogn_example(n):
    arr = np.random.rand(n)
    return np.sort(arr)

# O(log n) - Binary search
def logn_example(n):
    arr = list(range(n))
    target = n // 2
    lo, hi = 0, n - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# Measure
sizes = [100, 200, 400, 800, 1600]
measure_complexity(linear_example, sizes, "O(n) - Linear")
measure_complexity(quadratic_example, sizes, "O(n²) - Quadratic")
measure_complexity(nlogn_example, sizes, "O(n log n)")
```

---

## 📚 Resources

| Type | Title | Link |
|:-----|:------|:-----|
| 📖 Book | Introduction to Algorithms (CLRS) | Chapter 3 |
| 🎥 Video | MIT 6.006 - Algorithm Analysis | [YouTube](https://www.youtube.com/watch?v=HtSuA80QTyo) |
| 📄 Paper | Attention Is All You Need | [arXiv](https://arxiv.org/abs/1706.03762) |
| 📄 Paper | FlashAttention | [arXiv](https://arxiv.org/abs/2205.14135) |

---

## 🧭 Navigation

<table width="100%">
<tr>
<td align="left" width="33%">

⬅️ **Previous**<br>
[🔀 Logic](../04_logic/README.md)

</td>
<td align="center" width="34%">

📍 **Current: 5 of 6**<br>
**⏱️ Asymptotic Analysis**

</td>
<td align="right" width="33%">

➡️ **Next**<br>
[🔢 Numerical Computation](../06_numerical_computation/README.md)

</td>
</tr>
</table>

### Quick Links

| Direction | Destination |
|:---------:|-------------|
| 🏠 Section Home | [01: Mathematical Foundations](../README.md) |
| 📋 Full Course | [Course Home](../../README.md) |

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=18,19,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&pause=1000&color=00D4AA&center=true&vCenter=true&width=600&lines=Made+with+❤️+by+Gaurav+Goswami" alt="Footer" />
</p>

```