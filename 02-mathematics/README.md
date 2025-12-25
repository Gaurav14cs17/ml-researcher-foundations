<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=120&section=header&text=Topic&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-02-00C853?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📊 Learning Path

``` mermaid
graph LR
    A[🚀 Start] --> B[📐 Linear Alg]
    B --> C[Vectors]
    C --> D[Matrices]
    D --> E[Eigen/SVD]
    E --> F[📈 Calculus]
    F --> G[Gradients]
    G --> H[🎯 Optimization]
    H --> I[✅ Ready]

```

## 🎯 What You'll Learn

> 💡 Every ML algorithm can be understood through **linear algebra, calculus, and optimization**.

<table>
<tr>
<td align="center">

### 📐 Linear Algebra
<img src="https://img.shields.io/badge/12_hours-blue?style=flat-square"/>

Vectors, Matrices, SVD, PCA

</td>
<td align="center">

### 📈 Calculus
<img src="https://img.shields.io/badge/10_hours-green?style=flat-square"/>

Gradients, Chain Rule, Hessian

</td>
<td align="center">

### 🎯 Optimization
<img src="https://img.shields.io/badge/10_hours-orange?style=flat-square"/>

GD, SGD, Adam, Convergence

</td>
</tr>
</table>

---

## 📚 Main Topics

### 1️⃣ Linear Algebra

<img src="https://img.shields.io/badge/Time-12_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/Priority-⭐⭐⭐-gold?style=flat-square"/>

```mermaid
graph LR
    A[Vectors] --> B[Matrices]
    B --> C[Mult]
    C --> D[Eigen]
    D --> E[SVD]
    E --> F[PCA]
```

<details>
<summary><b>🔍 Core Concepts</b></summary>

- Vectors & Vector Spaces
- Matrix Operations & Properties
- Linear Transformations
- Eigenvalues & Eigenvectors
- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)

</details>

<details>
<summary><b>🎯 Why It Matters</b></summary>

- Neural networks are matrix multiplications
- PCA for dimensionality reduction
- SVD for recommender systems
- Eigenvalues for stability analysis

</details>

<a href="./01-linear-algebra/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-4285F4?style=for-the-badge" alt="Learn"/></a>

---

### 2️⃣ Calculus

<img src="https://img.shields.io/badge/Time-10_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/Priority-⭐⭐⭐-gold?style=flat-square"/>

```mermaid
graph LR
    A[Limits] --> B[Derivatives]
    B --> C[Partial]
    C --> D[Gradient]
    D --> E[Chain Rule]
    E --> F[Jacobian]
    F --> G[Hessian]
```

<details>
<summary><b>🔍 Core Concepts</b></summary>

- Derivatives & Partial Derivatives
- Gradient Vectors
- Chain Rule (backbone of backpropagation)
- Jacobian Matrices
- Hessian Matrices
- Taylor Series Approximation

</details>

<details>
<summary><b>🎯 Why It Matters</b></summary>

- Gradients are how neural networks learn
- Chain rule enables backpropagation
- Hessian for second-order optimization
- Taylor series for function approximation

</details>

<a href="./02-calculus/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-4285F4?style=for-the-badge" alt="Learn"/></a>

---

### 3️⃣ Optimization Theory

<img src="https://img.shields.io/badge/Time-10_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/Priority-⭐⭐⭐-gold?style=flat-square"/>

```mermaid
graph LR
    A[Convex] --> B[GD]
    B --> C[SGD]
    C --> D[Momentum]
    D --> E[Adam]
    E --> F[Converge]
```

<details>
<summary><b>🔍 Core Concepts</b></summary>

- Convex vs Non-Convex Functions
- Gradient Descent Variants
- First-Order Methods
- Second-Order Methods (Newton)
- Convergence Guarantees

</details>

<details>
<summary><b>🎯 Why It Matters</b></summary>

- Training = Optimization
- Understand SGD, Adam, AdamW
- Know when optimization will work
- Debug training issues

</details>

<a href="./03-optimization/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-4285F4?style=for-the-badge" alt="Learn"/></a>

---

## 🔄 How These Connect

```mermaid
graph TD
    A[📐 Linear Algebra] --> D[🧠 Neural Nets]
    B[📈 Calculus] --> D
    C[🎯 Optimization] --> D
    A --> E[Dim Reduction]
    B --> F[Backprop]
    C --> G[Training]
    D --> H[🚀 Deep Learning]
    E --> H
    F --> H
    G --> H
```

---

## 💡 Key Formulas

<table>
<tr>
<td>

### 📐 Linear Algebra
```
Matrix:    (AB)ᵀ = BᵀAᵀ
Eigen:     A = QΛQᵀ
SVD:       A = UΣVᵀ
```

</td>
<td>

### 📈 Calculus
```
Gradient:  ∇f = [∂f/∂xᵢ]ᵀ
Chain:     ∂z/∂x = ∂z/∂y · ∂y/∂x
Jacobian:  J = [∂fᵢ/∂xⱼ]
```

</td>
<td>

### 🎯 Optimization
```
GD:   θ ← θ - η∇L(θ)
Adam: θ ← θ - η·m̂/(√v̂+ε)
```

</td>
</tr>
</table>

---

## 🔗 Prerequisites & Next Steps

```mermaid
graph LR
    A[✅ Foundations] --> B[📊 Mathematics]
    B --> C[📈 Prob/Stats]
    C --> D[🎯 Optimization]
    D --> E[🧬 ML Theory]
    E --> F[🚀 Deep Learning]
```

<p align="center">
  <a href="../01-foundations/README.md"><img src="https://img.shields.io/badge/←_Prerequisites:_Foundations-gray?style=for-the-badge" alt="Prev"/></a>
  <a href="../03-probability-statistics/README.md"><img src="https://img.shields.io/badge/Next:_Probability_→-00C853?style=for-the-badge" alt="Next"/></a>
</p>

---

## 📚 Recommended Resources

| Type | Resource | Focus |
|:----:|----------|-------|
| 📘 | [Mathematics for ML](https://mml-book.github.io/) | Complete reference |
| 🎬 | [3Blue1Brown - Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) | Visual intuition |
| 🎬 | [3Blue1Brown - Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) | Fundamentals |
| 🎓 | MIT 18.06 | Linear Algebra |

---

## 🗺️ Quick Navigation

| Previous | Current | Next |
|:--------:|:-------:|:----:|
| [🔢 Foundations](../01-foundations/README.md) | **📊 Mathematics** | [📈 Probability →](../03-probability-statistics/README.md) |

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=00C853&height=80&section=footer" width="100%"/>
</p>
