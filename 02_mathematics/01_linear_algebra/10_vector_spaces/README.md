<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Vector%20Spaces&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Span%20·%20Basis%20·%20Dimension%20·%20Subspaces&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.10_Vector_Spaces-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-Span_Basis_Dimension-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Vector spaces are the abstract foundation of linear algebra.** They generalize ℝⁿ to any set where you can add elements and scale them.

- 📐 **Vector Space**: Set with addition (+) and scalar multiplication (·)
- 📏 **Basis**: Minimal set of vectors that span the space
- 🔢 **Dimension**: Number of vectors in any basis
- 🎯 **Subspace**: A vector space inside another vector space

---

## 📑 Table of Contents

1. [Definition and Axioms](#1-definition-and-axioms)
2. [Subspaces](#2-subspaces)
3. [Linear Independence](#3-linear-independence)
4. [Span and Basis](#4-span-and-basis)
5. [Dimension Theorem](#5-dimension-theorem)
6. [Four Fundamental Subspaces](#6-four-fundamental-subspaces)
7. [Code Implementation](#7-code-implementation)
8. [ML Applications](#8-ml-applications)
9. [Resources](#-resources)

---

## 🎨 Visual Overview

<img src="./images/vector-space.svg" width="100%">

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        VECTOR SPACE STRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   VECTOR SPACE V (e.g., ℝ³)                                                 │
│   ─────────────────────────                                                  │
│   Contains all 3D vectors                                                    │
│                                                                              │
│         ↗                                                                    │
│        / Subspace (plane)                                                    │
│       /  ─────────────────                                                   │
│      /   All vectors in a plane through origin                              │
│     /                                                                        │
│    /      ↗                                                                  │
│   /      / Subspace (line)                                                   │
│  /      /  ───────────────                                                   │
│ /      /   All vectors on a line through origin                             │
│/______/                                                                      │
│       {0} ← Trivial subspace (just the zero vector)                         │
│                                                                              │
│   HIERARCHY:                                                                 │
│   {0} ⊂ Line ⊂ Plane ⊂ ℝ³                                                  │
│   dim=0   dim=1   dim=2   dim=3                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Definition and Axioms

### 📌 Definition

A **vector space** $V$ over a field $\mathbb{F}$ (usually $\mathbb{R}$ or $\mathbb{C}$) is a set with two operations:

- **Vector addition**: $+: V \times V \to V$
- **Scalar multiplication**: $\cdot: \mathbb{F} \times V \to V$

### 📐 Vector Space Axioms

For all $\mathbf{u}, \mathbf{v}, \mathbf{w} \in V$ and $\alpha, \beta \in \mathbb{F}$:

| # | Axiom | Formula |
|---|-------|---------|
| 1 | Commutativity | $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$ |
| 2 | Associativity (add) | $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$ |
| 3 | Zero vector | $\exists \mathbf{0}: \mathbf{v} + \mathbf{0} = \mathbf{v}$ |
| 4 | Additive inverse | $\exists (-\mathbf{v}): \mathbf{v} + (-\mathbf{v}) = \mathbf{0}$ |
| 5 | Scalar mult identity | $1 \cdot \mathbf{v} = \mathbf{v}$ |
| 6 | Associativity (scalar) | $\alpha(\beta\mathbf{v}) = (\alpha\beta)\mathbf{v}$ |
| 7 | Distributivity (vector) | $\alpha(\mathbf{u} + \mathbf{v}) = \alpha\mathbf{u} + \alpha\mathbf{v}$ |
| 8 | Distributivity (scalar) | $(\alpha + \beta)\mathbf{v} = \alpha\mathbf{v} + \beta\mathbf{v}$ |

### 💡 Examples of Vector Spaces

| Space | Elements | Addition | Scalar Mult |
|-------|----------|----------|-------------|
| $\mathbb{R}^n$ | n-tuples of reals | Component-wise | Component-wise |
| $\mathbb{R}^{m \times n}$ | m×n matrices | Matrix addition | Scalar × matrix |
| $\mathcal{P}_n$ | Polynomials degree ≤n | Add coefficients | Scale coefficients |
| $C[a,b]$ | Continuous functions | $(f+g)(x) = f(x)+g(x)$ | $(\alpha f)(x) = \alpha f(x)$ |

### 🔍 Proof: $\mathbf{0} \cdot \mathbf{v} = \mathbf{0}$ (zero scalar gives zero vector)

```
Step 1: 0·v = (0 + 0)·v          (0 = 0 + 0)

Step 2: 0·v = 0·v + 0·v          (Axiom 8: distributivity)

Step 3: 0·v + (-(0·v)) = 0·v + 0·v + (-(0·v))    (add inverse)

Step 4: 0 = 0·v + 0              (inverse cancels)

Step 5: 0 = 0·v                  (Axiom 3)  ∎
```

---

## 2. Subspaces

### 📌 Definition

A subset $W \subseteq V$ is a **subspace** if $W$ is itself a vector space under the same operations.

### 📐 Subspace Test

$W$ is a subspace iff:
1. $\mathbf{0} \in W$ (contains zero)
2. $\mathbf{u}, \mathbf{v} \in W \Rightarrow \mathbf{u} + \mathbf{v} \in W$ (closed under addition)
3. $\alpha \in \mathbb{F}, \mathbf{v} \in W \Rightarrow \alpha\mathbf{v} \in W$ (closed under scaling)

Or equivalently (single condition):
$$\alpha\mathbf{u} + \beta\mathbf{v} \in W \text{ for all } \mathbf{u}, \mathbf{v} \in W, \alpha, \beta \in \mathbb{F}$$

### 💡 Examples

**Subspace**: Lines and planes through origin in $\mathbb{R}^3$
```
W = {(x, y, z) : x + y + z = 0}  (plane through origin)

Check:
1. (0,0,0) ∈ W  ✓  (0+0+0=0)
2. (a,b,c) + (d,e,f) = (a+d, b+e, c+f)
   (a+d)+(b+e)+(c+f) = (a+b+c)+(d+e+f) = 0+0 = 0  ✓
3. α(a,b,c) = (αa,αb,αc)
   αa+αb+αc = α(a+b+c) = α·0 = 0  ✓
```

**NOT a Subspace**: Lines not through origin
```
W = {(x, y) : x + y = 1}

Check zero vector:
(0,0): 0 + 0 = 0 ≠ 1  ✗

Not a subspace! (doesn't contain origin)
```

---

## 3. Linear Independence

### 📌 Definition

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if:

$$\alpha_1\mathbf{v}_1 + \alpha_2\mathbf{v}_2 + \cdots + \alpha_k\mathbf{v}_k = \mathbf{0} \implies \alpha_1 = \alpha_2 = \cdots = \alpha_k = 0$$

Otherwise, they are **linearly dependent**.

### 🔍 Geometric Interpretation

```
Linearly Dependent:
  One vector can be written as a combination of others
  → They lie in a lower-dimensional subspace

Linearly Independent:
  No vector is a combination of the others
  → They span a full k-dimensional subspace

Examples in ℝ³:
  {e₁, e₂, e₃} independent → span ℝ³
  {e₁, e₂, e₁+e₂} dependent → e₁+e₂ is a combo of others → span is a plane
```

### 💡 Examples

**Example 1**: Checking Independence (Matrix Method)
```
Are v₁ = [1,2,3], v₂ = [4,5,6], v₃ = [7,8,9] independent?

Form matrix A = [v₁|v₂|v₃] and check if det(A) ≠ 0:

A = [1 4 7]
    [2 5 8]
    [3 6 9]

det(A) = 1(45-48) - 4(18-24) + 7(12-15)
       = 1(-3) - 4(-6) + 7(-3)
       = -3 + 24 - 21
       = 0

det = 0 → DEPENDENT

Dependency: v₃ = -v₁ + 2v₂
Check: -[1,2,3] + 2[4,5,6] = [-1+8, -2+10, -3+12] = [7,8,9] ✓
```

**Example 2**: Polynomials
```
Are 1, x, x² linearly independent in P₂?

Suppose α₁(1) + α₂(x) + α₃(x²) = 0 (the zero polynomial)

For this to equal 0 for all x:
  Coefficient of x⁰: α₁ = 0
  Coefficient of x¹: α₂ = 0
  Coefficient of x²: α₃ = 0

Only trivial solution → INDEPENDENT
```

---

## 4. Span and Basis

### 📌 Span

The **span** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is all linear combinations:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{\alpha_1\mathbf{v}_1 + \cdots + \alpha_k\mathbf{v}_k : \alpha_i \in \mathbb{F}\}$$

### 📌 Basis

A **basis** for $V$ is a set of vectors that:
1. **Spans** $V$: Every vector in $V$ can be written as their linear combination
2. Is **linearly independent**: No redundant vectors

### 🔍 Theorem: Basis Representation is Unique

```
If {v₁, ..., vₙ} is a basis for V, then every v ∈ V has a UNIQUE representation:
  v = α₁v₁ + α₂v₂ + ... + αₙvₙ

Proof:
Suppose v = α₁v₁ + ... + αₙvₙ = β₁v₁ + ... + βₙvₙ

Subtract: 0 = (α₁-β₁)v₁ + ... + (αₙ-βₙ)vₙ

Since {v₁,...,vₙ} is linearly independent:
  α₁-β₁ = ... = αₙ-βₙ = 0
  αᵢ = βᵢ for all i  ∎
```

### 💡 Examples

**Standard Basis of $\mathbb{R}^n$**:
```
e₁ = [1,0,...,0]ᵀ
e₂ = [0,1,...,0]ᵀ
...
eₙ = [0,0,...,1]ᵀ

Any v = [v₁, v₂, ..., vₙ]ᵀ = v₁e₁ + v₂e₂ + ... + vₙeₙ
```

**Non-Standard Basis**:
```
{[1,1]ᵀ, [1,-1]ᵀ} is also a basis for ℝ²

To express [3,1]ᵀ:
  [3,1]ᵀ = α[1,1]ᵀ + β[1,-1]ᵀ
  3 = α + β
  1 = α - β
  
  Solving: α = 2, β = 1
  [3,1]ᵀ = 2[1,1]ᵀ + 1[1,-1]ᵀ ✓
```

---

## 5. Dimension Theorem

### 📌 Theorem

All bases of a finite-dimensional vector space have the same number of elements. This number is the **dimension** of the space.

$$\dim(V) = |\text{any basis of } V|$$

### 🔍 Proof Sketch

```
Key Lemma (Steinitz Exchange):
  If {v₁,...,vₘ} spans V and {u₁,...,uₙ} is independent in V,
  then n ≤ m.

Proof of dimension theorem:
  Let B = {v₁,...,vₘ} and B' = {u₁,...,uₙ} be two bases.
  
  B spans V, B' is independent → n ≤ m (by lemma)
  B' spans V, B is independent → m ≤ n (by lemma)
  
  Therefore m = n.  ∎
```

### 📐 Key Results

| Property | Formula |
|----------|---------|
| Subspace dimension | $\dim(W) \leq \dim(V)$ for $W \subseteq V$ |
| Sum of subspaces | $\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$ |
| Direct sum | $\dim(U \oplus W) = \dim(U) + \dim(W)$ if $U \cap W = \{0\}$ |

---

## 6. Four Fundamental Subspaces

For matrix $A \in \mathbb{R}^{m \times n}$:

### 📐 The Four Subspaces

| Subspace | Definition | Dimension |
|----------|------------|-----------|
| **Column Space** | $\mathcal{C}(A) = \{A\mathbf{x} : \mathbf{x} \in \mathbb{R}^n\}$ | $r = \text{rank}(A)$ |
| **Row Space** | $\mathcal{C}(A^T) = \{A^T\mathbf{y} : \mathbf{y} \in \mathbb{R}^m\}$ | $r = \text{rank}(A)$ |
| **Null Space** | $\mathcal{N}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ | $n - r$ |
| **Left Null Space** | $\mathcal{N}(A^T) = \{\mathbf{y} : A^T\mathbf{y} = \mathbf{0}\}$ | $m - r$ |

### 🔍 Orthogonality Relations

```
Column Space ⊥ Left Null Space  (both in ℝᵐ)
Row Space ⊥ Null Space          (both in ℝⁿ)

     ℝⁿ                    ℝᵐ
  ┌─────────┐           ┌─────────┐
  │Row Space│    A      │Col Space│
  │  (r-dim)│  ────►    │  (r-dim)│
  │         │           │         │
  ├─────────┤           ├─────────┤
  │Null Spce│    A      │Left Null│
  │(n-r dim)│  ────►    │(m-r dim)│
  │    ↓    │   =0      │    ↓    │
  │    0    │           │    0    │
  └─────────┘           └─────────┘
```

### 💡 Example

```
A = [1  2  3]
    [4  5  6]

Rank: r = 2 (rows are independent)

Column Space: span{[1,4]ᵀ, [2,5]ᵀ} = ℝ² (dimension 2)

Row Space: span{[1,2,3], [4,5,6]} (dimension 2 in ℝ³)

Null Space: Solve Ax = 0
  [1  2  3][x]   [0]
  [4  5  6][y] = [0]
            [z]
  
  Row reduce → x = z, y = -2z
  Null space = span{[1, -2, 1]ᵀ} (dimension 1 = 3-2)

Left Null Space: dimension 0 (since m - r = 2 - 2 = 0)
```

---

## 7. Code Implementation

```python
import numpy as np

def is_linearly_independent(vectors):
    """Check if a set of vectors is linearly independent."""
    matrix = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(matrix)
    return rank == len(vectors)

def find_basis(vectors):
    """Find a basis for the span of given vectors."""
    matrix = np.column_stack(vectors)
    Q, R = np.linalg.qr(matrix)
    
    # Find non-zero diagonal elements of R
    rank = np.sum(np.abs(np.diag(R)) > 1e-10)
    
    # Corresponding columns of Q form an orthonormal basis
    return Q[:, :rank]

def compute_four_subspaces(A):
    """
    Compute bases for the four fundamental subspaces of A.
    """
    m, n = A.shape
    
    # SVD gives us everything
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)
    
    results = {
        'column_space': U[:, :rank],           # First r columns of U
        'row_space': Vt[:rank, :].T,           # First r rows of Vt (transposed)
        'null_space': Vt[rank:, :].T,          # Remaining rows of Vt (transposed)
        'left_null_space': U[:, rank:],        # Remaining columns of U
        'rank': rank,
        'nullity': n - rank,
        'left_nullity': m - rank
    }
    
    return results

def verify_orthogonality(A):
    """Verify orthogonality of fundamental subspaces."""
    subspaces = compute_four_subspaces(A)
    
    # Column space ⊥ Left null space (both in ℝᵐ)
    col = subspaces['column_space']
    left_null = subspaces['left_null_space']
    if col.size > 0 and left_null.size > 0:
        orth1 = np.allclose(col.T @ left_null, 0)
        print(f"Col ⊥ LeftNull: {orth1}")
    
    # Row space ⊥ Null space (both in ℝⁿ)
    row = subspaces['row_space']
    null = subspaces['null_space']
    if row.size > 0 and null.size > 0:
        orth2 = np.allclose(row.T @ null, 0)
        print(f"Row ⊥ Null: {orth2}")
    
    return subspaces

# Example
A = np.array([[1, 2, 3], [4, 5, 6]])
subspaces = verify_orthogonality(A)
print(f"Rank: {subspaces['rank']}")
print(f"Nullity: {subspaces['nullity']}")
```

---

## 8. ML Applications

### 🤖 Application 1: Feature Spaces

```python
# Word embeddings live in a vector space
# Similar words should be close in this space

def word_analogy(embeddings, word_a, word_b, word_c):
    """
    Solve: A is to B as C is to ?
    Answer = embedding(B) - embedding(A) + embedding(C)
    
    This works because vector arithmetic preserves relationships!
    """
    vec_a = embeddings[word_a]
    vec_b = embeddings[word_b]
    vec_c = embeddings[word_c]
    
    # Vector arithmetic in embedding space
    target = vec_b - vec_a + vec_c
    
    # Find closest word
    similarities = {word: np.dot(target, vec) / (np.linalg.norm(target) * np.linalg.norm(vec))
                   for word, vec in embeddings.items()}
    
    return max(similarities, key=similarities.get)

# "king" - "man" + "woman" ≈ "queen"
```

### 🤖 Application 2: Null Space for Solutions

```python
def find_all_solutions(A, b):
    """
    Solve Ax = b and characterize all solutions.
    
    General solution = particular solution + null space
    x = x_particular + null(A)
    """
    # Particular solution (least-norm)
    x_particular = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Null space basis
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-10)
    null_basis = Vt[rank:, :].T
    
    # All solutions: x_particular + Σ αᵢ nᵢ
    return {
        'particular': x_particular,
        'null_basis': null_basis,
        'num_free_params': null_basis.shape[1]
    }
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 📖 | [Linear Algebra Done Right](https://linear.axler.net/) | Axler (theory) |
| 🎥 | [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Visual intuition |
| 📖 | [MIT 18.06](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | Strang |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [Vectors & Matrices](../09_vectors_matrices/README.md) | [Linear Algebra](../README.md) | [Calculus](../../02_calculus/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
