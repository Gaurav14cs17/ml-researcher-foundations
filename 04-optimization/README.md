<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=120&section=header&text=Topic&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-04-FF6B6B?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📊 Learning Path

```mermaid
graph LR
    A[🚀 Start] --> B[📐 Foundations]
    B --> C[📉 GD]
    C --> D[⚡ SGD]
    D --> E[🚀 Adam]
    E --> F[📊 Convex]
    F --> G[🔒 Constrained]
    G --> H[🏆 Master]
```

## 🎯 What You'll Learn

> 💡 **Training = Optimization.** Every neural network learns by minimizing a loss function.

<table>
<tr>
<td align="center">

### 📉 Gradient Descent
Foundation of learning

</td>
<td align="center">

### 🚀 Adam
Default optimizer (90%)

</td>
<td align="center">

### 🔒 Constrained
KKT, SVM derivation

</td>
</tr>
</table>

---

## 📚 Main Topics

### 1️⃣ Foundations

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Gradients] --> B[Hessian]
    B --> C[Taylor]
    C --> D[Convexity]
    D --> E[Lipschitz]
```

**Core:** Gradients, Hessian, Taylor Series, Convexity

<a href="./01-foundations/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-E91E63?style=for-the-badge" alt="Learn"/></a>

---

### 2️⃣ Basic Methods

<img src="https://img.shields.io/badge/Time-8_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[GD] --> B[Line Search]
    B --> C[Newton]
    C --> D[Quasi-Newton]
    D --> E[CG]
```

**Core:** GD: θ ← θ - η∇L(θ), Newton's Method, BFGS

<a href="./02-basic-methods/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-E91E63?style=for-the-badge" alt="Learn"/></a>

---

### 3️⃣ Convex Optimization

<img src="https://img.shields.io/badge/Time-8_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Convex Sets] --> B[Convex Funcs]
    B --> C[Optimality]
    C --> D[Duality]
    D --> E[ML Apps]
```

**Core:** Convex Functions, First-Order Optimality, Duality

<a href="./04-convex-optimization/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-E91E63?style=for-the-badge" alt="Learn"/></a>

---

### 4️⃣ Constrained Optimization

<img src="https://img.shields.io/badge/Time-8_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Lagrange] --> B[KKT]
    B --> C[Inequality]
    C --> D[SVM]
    D --> E[Dual]
```

**Core:** Lagrange Multipliers, KKT Conditions, SVM Derivation

<a href="./05-constrained-optimization/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-E91E63?style=for-the-badge" alt="Learn"/></a>

---

### 5️⃣ ML Optimizers ⭐⭐⭐

<img src="https://img.shields.io/badge/Time-10_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_MOST_IMPORTANT-critical?style=flat-square"/>

```mermaid
graph LR
    A[SGD] --> B[Momentum]
    B --> C[RMSprop]
    C --> D[Adam]
    D --> E[AdamW]
    E --> F[LR Schedule]
```

> ⭐ **Adam is the default optimizer for 90% of models**

| Optimizer | Speed | Best For |
|:---------:|:-----:|----------|
| SGD | Slow | Simple, regularization |
| Momentum | Medium | Convex problems |
| **Adam** | **Fast** | **Default choice** ⭐ |
| AdamW | Fast | Transformers, LLMs |

<a href="./08-machine-learning/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-E91E63?style=for-the-badge" alt="Learn"/></a>

---

### 6️⃣ Linear & Integer Programming

<img src="https://img.shields.io/badge/Time-14_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[LP] --> B[Simplex]
    B --> C[Interior Pt]
    C --> D[IP]
    D --> E[Branch Bound]
```

<a href="./06-linear-programming/README.md"><img src="https://img.shields.io/badge/📖_Linear_Programming-E91E63?style=for-the-badge" alt="LP"/></a>
<a href="./07-integer-programming/README.md"><img src="https://img.shields.io/badge/📖_Integer_Programming-E91E63?style=for-the-badge" alt="IP"/></a>

---

### 7️⃣ Metaheuristics

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Genetic] --> B[Annealing]
    B --> C[Swarm]
    C --> D[Evolution]
```

<a href="./09-metaheuristics/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-E91E63?style=for-the-badge" alt="Learn"/></a>

---

## 💡 Key Algorithms

<table>
<tr>
<td>

### 📉 Gradient Descent
```python
θ ← θ - η∇L(θ)
```

</td>
<td>

### 🏃 Momentum
```python
v ← βv + ∇L(θ)
θ ← θ - ηv
```

</td>
<td>

### 🚀 Adam
```python
m ← β₁m + (1-β₁)∇L
v ← β₂v + (1-β₂)(∇L)²
θ ← θ - η·m̂/(√v̂+ε)
```

</td>
</tr>
</table>

---

## 🔗 Prerequisites & Next Steps

```mermaid
graph LR
    A[📊 Math] --> B[🎯 Optimization]
    B --> C[🧬 ML Theory]
    C --> D[🚀 Deep Learning]
    D --> E[⚡ Training]
```

<p align="center">
  <a href="../02-mathematics/README.md"><img src="https://img.shields.io/badge/←_Prerequisites:_Mathematics-gray?style=for-the-badge" alt="Prev"/></a>
  <a href="../05-ml-theory/README.md"><img src="https://img.shields.io/badge/Next:_ML_Theory_→-00C853?style=for-the-badge" alt="Next"/></a>
</p>

---

## 📚 Recommended Resources

| Type | Resource | Focus |
|:----:|----------|-------|
| 📘 | [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) | Boyd & Vandenberghe |
| 📄 | [Adam Paper](https://arxiv.org/abs/1412.6980) | Original Adam |
| 📄 | [AdamW Paper](https://arxiv.org/abs/1711.05101) | Weight decay fix |

---

## 🗺️ Quick Navigation

| Previous | Current | Next |
|:--------:|:-------:|:----:|
| [📈 Probability](../03-probability-statistics/README.md) | **🎯 Optimization** | [🧬 ML Theory →](../05-ml-theory/README.md) |

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=FF6B6B&height=80&section=footer" width="100%"/>
</p>
