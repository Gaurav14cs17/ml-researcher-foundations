<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Linear%20Transformations&fontSize=38&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Linear%20Maps%20·%20Change%20of%20Basis%20·%20Matrix%20Representations&descAlignY=52&descSize=16" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Section-01.08_Transformations-00C853?style=for-the-badge" alt="Section"/>
  <img src="https://img.shields.io/badge/📊_Topics-Linear_Maps_Basis-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/✍️_Author-Gaurav_Goswami-purple?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/📅_Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## ⚡ TL;DR

> **Every neural network layer is a linear transformation** (plus optional nonlinearity). Understanding linear maps reveals what layers actually do to your data.

- 📐 **Definition**: $T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$ (preserves linear combinations)
- 🔗 **Matrix form**: Every linear map $T: \mathbb{R}^n \to \mathbb{R}^m$ is $T(\mathbf{x}) = A\mathbf{x}$
- 🔄 **Change of basis**: Same transformation, different coordinates
- 🤖 **In ML**: Dense layers, attention projections, convolutions

---

## 📑 Table of Contents

1. [Definition and Properties](#1-definition-and-properties)
2. [Matrix Representation](#2-matrix-representation-theorem)
3. [Common Transformations](#3-common-linear-transformations)
4. [Change of Basis](#4-change-of-basis)
5. [Kernel and Image](#5-kernel-and-image)
6. [Rank-Nullity Theorem](#6-rank-nullity-theorem)
7. [Code Implementation](#7-code-implementation)
8. [ML Applications](#8-ml-applications)
9. [Resources](#-resources)

---

## 1. Definition and Properties

### 📌 Formal Definition

A function $T: V \to W$ between vector spaces is a **linear transformation** (or linear map) if:

$$T(\alpha\mathbf{u} + \beta\mathbf{v}) = \alpha T(\mathbf{u}) + \beta T(\mathbf{v})$$

for all vectors $\mathbf{u}, \mathbf{v} \in V$ and scalars $\alpha, \beta$.

### Equivalent Conditions

```
T is linear if and only if:

1. T(u + v) = T(u) + T(v)     (Additivity)
2. T(αv) = αT(v)              (Homogeneity)

Or equivalently (single condition):
3. T(αu + βv) = αT(u) + βT(v) (Linearity)
```

### 🔍 What Linearity Preserves

| Property | Preserved? | Example |
|----------|------------|---------|
| Lines through origin | ✅ Yes | Lines map to lines |
| Parallelism | ✅ Yes | Parallel lines stay parallel |
| Origin | ✅ Yes | $T(\mathbf{0}) = \mathbf{0}$ always |
| Distances | ❌ Generally no | Unless orthogonal |
| Angles | ❌ Generally no | Unless orthogonal |

### 🔍 Proof: $T(\mathbf{0}) = \mathbf{0}$

```
Step 1: Write 0 = 0·v for any vector v

Step 2: Apply linearity
        T(0) = T(0·v) = 0·T(v) = 0

Therefore T(0) = 0 for any linear map.  ∎
```

---

## 2. Matrix Representation Theorem

### 📌 Theorem

Every linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ can be uniquely represented by a matrix $A \in \mathbb{R}^{m \times n}$:

$$T(\mathbf{x}) = A\mathbf{x}$$

### 🔍 Proof (Constructive)

```
Step 1: Let {e₁, e₂, ..., eₙ} be the standard basis of ℝⁿ
        e₁ = [1,0,...,0]ᵀ, e₂ = [0,1,...,0]ᵀ, etc.

Step 2: Any vector x can be written as:
        x = x₁e₁ + x₂e₂ + ... + xₙeₙ

Step 3: Apply T (using linearity):
        T(x) = T(x₁e₁ + x₂e₂ + ... + xₙeₙ)
             = x₁T(e₁) + x₂T(e₂) + ... + xₙT(eₙ)

Step 4: Define matrix A with columns = T(eⱼ):
        A = [T(e₁) | T(e₂) | ... | T(eₙ)]

Step 5: Verify T(x) = Ax:
        Ax = [T(e₁)|...|T(eₙ)] [x₁]   = x₁T(e₁) + ... + xₙT(eₙ) = T(x)  ✓
                                [⋮ ]
                                [xₙ]

Step 6: Uniqueness follows because A is determined by T(e₁),...,T(eₙ).  ∎
```

### Key Insight

> **Column j of matrix A = where basis vector eⱼ gets mapped**

```
A = [T(e₁) | T(e₂) | ... | T(eₙ)]
     ↑        ↑            ↑
   where    where        where
   e₁ goes  e₂ goes      eₙ goes
```

---

## 3. Common Linear Transformations

### 📊 2D Transformations

| Transformation | Matrix | Effect |
|----------------|--------|--------|
| **Identity** | $\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$ | No change |
| **Scaling** | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$ | Scale by $(s_x, s_y)$ |
| **Rotation** | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Rotate by $\theta$ |
| **Reflection (x-axis)** | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | Flip vertically |
| **Reflection (y-axis)** | $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ | Flip horizontally |
| **Shear (horizontal)** | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$ | Slant by factor $k$ |
| **Projection (onto x-axis)** | $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$ | Collapse to x-axis |

### 🔍 Derivation: Rotation Matrix

```
Goal: Find matrix for counterclockwise rotation by angle θ

Step 1: Where does e₁ = [1, 0]ᵀ go?
        Rotating [1, 0]ᵀ by θ gives [cos θ, sin θ]ᵀ
        This is column 1 of R(θ)

Step 2: Where does e₂ = [0, 1]ᵀ go?
        Rotating [0, 1]ᵀ by θ gives [-sin θ, cos θ]ᵀ
        (90° ahead of e₁, so add 90° to the angle)
        This is column 2 of R(θ)

Step 3: Assemble matrix:
        R(θ) = [cos θ  -sin θ]
               [sin θ   cos θ]

Verification: R(θ) is orthogonal
  R(θ)ᵀR(θ) = [cos θ   sin θ][cos θ  -sin θ]
              [-sin θ  cos θ][sin θ   cos θ]
            = [cos²θ + sin²θ           0        ]
              [      0         cos²θ + sin²θ    ]
            = I  ✓
```

### 🔍 Proof: Projection is Linear

```
Projection onto unit vector u:
  P_u(x) = (uᵀx)u = u(uᵀx) = (uuᵀ)x

Matrix form: P_u = uuᵀ (outer product)

Verify linearity:
  P_u(αx + βy) = (uuᵀ)(αx + βy)
               = α(uuᵀ)x + β(uuᵀ)y
               = αP_u(x) + βP_u(y)  ✓
```

---

## 4. Change of Basis

### 📌 Problem

The same linear transformation has different matrix representations in different bases. How do they relate?

### 📐 Theorem: Similarity Transformation

If $A$ represents $T$ in the standard basis and $B$ represents $T$ in basis $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$, then:

$$B = P^{-1}AP$$

where $P = [\mathbf{v}_1 | \cdots | \mathbf{v}_n]$ (change of basis matrix).

### 🔍 Proof

```
Step 1: P converts coordinates from new basis to standard basis
        If x has coordinates c in the new basis, then x = Pc

Step 2: Consider T applied in two ways:

        In standard basis:
        T(x) = Ax = A(Pc) = (AP)c

        In new basis:
        [T(x)]_new = B[x]_new = Bc

Step 3: Convert result back to standard:
        T(x) = P·[T(x)]_new = P(Bc) = (PB)c

Step 4: Equate:
        (AP)c = (PB)c for all c
        AP = PB
        B = P⁻¹AP  ∎
```

### 💡 Example: Diagonalization

```
For diagonalizable A:
  A = PΛP⁻¹

Here:
  - P = matrix of eigenvectors
  - Λ = diagonal matrix of eigenvalues

Interpretation:
  - In the eigenbasis, T is just scaling along each axis!
  - P⁻¹ converts to eigenbasis
  - Λ scales each coordinate
  - P converts back to standard basis
```

---

## 5. Kernel and Image

### 📌 Definitions

For linear transformation $T: V \to W$:

**Kernel (Null Space)**:
$$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$$

**Image (Range)**:
$$\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$$

### 🔍 Proof: Kernel is a Subspace

```
To show ker(T) is a subspace, verify three properties:

1. Contains zero vector:
   T(0) = 0 (proved earlier), so 0 ∈ ker(T) ✓

2. Closed under addition:
   If u, v ∈ ker(T), then T(u) = T(v) = 0
   T(u + v) = T(u) + T(v) = 0 + 0 = 0
   So u + v ∈ ker(T) ✓

3. Closed under scalar multiplication:
   If v ∈ ker(T) and α is a scalar:
   T(αv) = αT(v) = α·0 = 0
   So αv ∈ ker(T) ✓

Therefore ker(T) is a subspace.  ∎
```

### 🔍 Proof: Image is a Subspace

```
1. Contains zero: T(0) = 0, so 0 ∈ Im(T) ✓

2. Closed under addition:
   If y₁, y₂ ∈ Im(T), then y₁ = T(x₁), y₂ = T(x₂) for some x₁, x₂
   y₁ + y₂ = T(x₁) + T(x₂) = T(x₁ + x₂) ∈ Im(T) ✓

3. Closed under scaling:
   If y ∈ Im(T), then y = T(x) for some x
   αy = αT(x) = T(αx) ∈ Im(T) ✓

Therefore Im(T) is a subspace.  ∎
```

---

## 6. Rank-Nullity Theorem

### 📌 Theorem

For linear transformation $T: V \to W$ where $V$ is finite-dimensional:

$$\dim(\ker(T)) + \dim(\text{Im}(T)) = \dim(V)$$

Or equivalently, for matrix $A \in \mathbb{R}^{m \times n}$:

$$\text{nullity}(A) + \text{rank}(A) = n$$

### 🔍 Proof

```
Step 1: Let {u₁, ..., uₖ} be a basis for ker(T)
        So dim(ker(T)) = k

Step 2: Extend to a basis of V: {u₁, ..., uₖ, v₁, ..., vᵣ}
        So n = dim(V) = k + r

Step 3: Claim: {T(v₁), ..., T(vᵣ)} is a basis for Im(T)

Step 3a: Spanning:
        For any y ∈ Im(T), write y = T(x) for some x ∈ V
        x = α₁u₁ + ... + αₖuₖ + β₁v₁ + ... + βᵣvᵣ
        T(x) = α₁T(u₁) + ... + αₖT(uₖ) + β₁T(v₁) + ... + βᵣT(vᵣ)
             = 0 + ... + 0 + β₁T(v₁) + ... + βᵣT(vᵣ)  (since uᵢ ∈ ker(T))
        So y ∈ span{T(v₁), ..., T(vᵣ)} ✓

Step 3b: Linear independence:
        Suppose c₁T(v₁) + ... + cᵣT(vᵣ) = 0
        T(c₁v₁ + ... + cᵣvᵣ) = 0
        So c₁v₁ + ... + cᵣvᵣ ∈ ker(T)
        c₁v₁ + ... + cᵣvᵣ = d₁u₁ + ... + dₖuₖ for some dⱼ
        c₁v₁ + ... + cᵣvᵣ - d₁u₁ - ... - dₖuₖ = 0
        Since {u₁,...,uₖ,v₁,...,vᵣ} is a basis: all coefficients = 0
        In particular: c₁ = ... = cᵣ = 0 ✓

Step 4: Therefore dim(Im(T)) = r

Step 5: Conclude:
        dim(ker(T)) + dim(Im(T)) = k + r = n = dim(V)  ∎
```

### 💡 Applications

| Condition | Implication |
|-----------|-------------|
| $\text{rank}(A) = n$ | $A$ has trivial kernel, injective |
| $\text{rank}(A) = m$ | $A$ is surjective (onto) |
| $\text{rank}(A) = \min(m,n)$ | Full rank |

---

## 7. Code Implementation

```python
import numpy as np
import torch

def create_rotation_matrix(theta):
    """Create 2D rotation matrix"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])

def create_scaling_matrix(sx, sy):
    """Create 2D scaling matrix"""
    return np.array([[sx, 0], [0, sy]])

def create_projection_matrix(u):
    """Create projection matrix onto unit vector u"""
    u = u / np.linalg.norm(u)
    return np.outer(u, u)

def create_reflection_matrix(u):
    """Create reflection matrix across hyperplane with normal u"""
    u = u / np.linalg.norm(u)
    return np.eye(len(u)) - 2 * np.outer(u, u)

def compute_kernel(A, tol=1e-10):
    """Compute basis for kernel (null space) of A"""
    U, S, Vh = np.linalg.svd(A)
    null_mask = S < tol
    kernel = Vh[null_mask].T
    return kernel

def compute_image(A, tol=1e-10):
    """Compute basis for image (column space) of A"""
    U, S, Vh = np.linalg.svd(A)
    rank = np.sum(S > tol)
    return U[:, :rank]

def verify_linearity(T, x, y, alpha=2.0, beta=3.0, tol=1e-10):
    """Verify T is linear: T(αx + βy) = αT(x) + βT(y)"""
    lhs = T(alpha * x + beta * y)
    rhs = alpha * T(x) + beta * T(y)
    is_linear = np.linalg.norm(lhs - rhs) < tol
    return is_linear

def change_of_basis(A, P):
    """Compute matrix representation in new basis defined by P"""
    # B = P⁻¹ A P
    return np.linalg.inv(P) @ A @ P

# Example: Verify rotation is orthogonal
theta = np.pi / 4
R = create_rotation_matrix(theta)
print(f"R @ R.T = I? {np.allclose(R @ R.T, np.eye(2))}")
print(f"det(R) = {np.linalg.det(R):.4f}")  # Should be 1

# Example: Rank-Nullity
A = np.array([[1, 2, 3], [4, 5, 6]])  # 2×3 matrix
rank = np.linalg.matrix_rank(A)
nullity = A.shape[1] - rank  # n - rank
print(f"rank(A) = {rank}, nullity(A) = {nullity}")
print(f"rank + nullity = {rank + nullity} = n = {A.shape[1]}")

# Compute kernel and image bases
kernel = compute_kernel(A)
image = compute_image(A)
print(f"Kernel dimension: {kernel.shape[1] if kernel.size > 0 else 0}")
print(f"Image dimension: {image.shape[1]}")
```

### PyTorch: Neural Network as Linear Transformation

```python
import torch
import torch.nn as nn

class LinearLayerAnalysis:
    """Analyze linear layer as a linear transformation"""
    
    def __init__(self, layer: nn.Linear):
        self.W = layer.weight.detach().numpy()  # Shape: (out, in)
        self.b = layer.bias.detach().numpy() if layer.bias is not None else None
    
    def rank(self):
        return np.linalg.matrix_rank(self.W)
    
    def nullity(self):
        return self.W.shape[1] - self.rank()
    
    def singular_values(self):
        return np.linalg.svd(self.W, compute_uv=False)
    
    def condition_number(self):
        s = self.singular_values()
        return s[0] / s[-1] if s[-1] > 1e-10 else np.inf
    
    def is_information_bottleneck(self):
        """Check if layer reduces dimensionality"""
        return self.rank() < min(self.W.shape)

# Example
layer = nn.Linear(512, 256)
analysis = LinearLayerAnalysis(layer)
print(f"Rank: {analysis.rank()}")
print(f"Condition number: {analysis.condition_number():.2f}")
print(f"Top singular values: {analysis.singular_values()[:5]}")
```

---

## 8. ML Applications

### 🤖 Application 1: Dense/Linear Layers

```python
# Dense layer: y = Wx + b
# 
# The W matrix IS a linear transformation from ℝⁿ → ℝᵐ
# Bias b makes it AFFINE (not strictly linear)

class DenseLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        self.b = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # Linear transformation + translation
        return x @ self.W.T + self.b
```

### 🤖 Application 2: Attention Projections

```python
class AttentionProjections(nn.Module):
    """
    Attention uses three linear transformations:
    Q = XW_Q (query projection)
    K = XW_K (key projection)  
    V = XW_V (value projection)
    """
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_k, bias=False)
        self.W_K = nn.Linear(d_model, d_k, bias=False)
        self.W_V = nn.Linear(d_model, d_v, bias=False)
    
    def forward(self, X):
        Q = self.W_Q(X)  # Project to query space
        K = self.W_K(X)  # Project to key space
        V = self.W_V(X)  # Project to value space
        return Q, K, V
```

### 🤖 Application 3: Embedding Lookup as Linear Transformation

```python
# Embedding lookup is a linear transformation!
# One-hot × Embedding matrix = Embedding vector

class EmbeddingAsLinear(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # Embedding matrix: vocab_size × embed_dim
        self.E = nn.Parameter(torch.randn(vocab_size, embed_dim))
    
    def forward(self, indices):
        # One-hot encoding
        one_hot = torch.zeros(len(indices), self.E.shape[0])
        one_hot.scatter_(1, indices.unsqueeze(1), 1)
        
        # Linear transformation: one_hot @ E
        return one_hot @ self.E
        
        # Note: nn.Embedding does this efficiently without one-hot

# Verify: nn.Embedding is mathematically identical
embed = nn.Embedding(1000, 256)
indices = torch.tensor([5, 10, 15])

# Method 1: Direct lookup
out1 = embed(indices)

# Method 2: One-hot @ weight
one_hot = torch.zeros(3, 1000)
one_hot.scatter_(1, indices.unsqueeze(1), 1)
out2 = one_hot @ embed.weight

print(f"Same result: {torch.allclose(out1, out2)}")  # True
```

---

## 📚 Resources

| Type | Resource | Description |
|------|----------|-------------|
| 🎥 | [3Blue1Brown: Linear Transformations](https://www.youtube.com/watch?v=kYB8IZa5AuE) | Visual |
| 📖 | Linear Algebra Done Right (Axler) | Theory |
| 📖 | MIT 18.06 Lecture Notes | Strang |

---

## 🗺️ Navigation

| ⬅️ Previous | 🏠 Home | ➡️ Next |
|:-----------:|:-------:|:-------:|
| [SVD](../07_svd/README.md) | [Linear Algebra](../README.md) | [Vectors & Matrices](../09_vectors_matrices/README.md) |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer&animation=twinkling" width="100%"/>
</p>
