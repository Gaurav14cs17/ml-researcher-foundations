# 📐 Foundations

> **The mathematical building blocks for all optimization**

## 🎯 Visual Overview

<img src="./images/foundations.svg" width="100%">

*Caption: Mathematical foundations for optimization include calculus (derivatives, gradients, Hessian), linear algebra (matrices, eigenvalues, SVD), and analysis (continuity, convexity, convergence).*

---

## 📂 Subtopics

| Folder | Topic | Why Important |
|--------|-------|---------------|
| [calculus/](./02-calculus/) | Derivatives, Gradients, Hessian | Direction of steepest descent |
| [linear-algebra/](./01-linear-algebra/) | Matrices, Eigenvalues, SVD | Convergence rates, conditioning |

---

## 🎯 What You'll Learn

```
+---------------------------------------------------------+
|                    FOUNDATIONS                          |
+---------------------------------------------------------+
|                                                         |
|   Calculus                    Linear Algebra            |
|   ---------                   --------------            |
|   • Partial derivatives       • Matrix operations       |
|   • Gradient ∇f               • Eigenvalues λ          |
|   • Hessian H                 • Positive definite      |
|   • Chain rule                • Condition number       |
|   • Taylor expansion          • Decompositions         |
|                                                         |
|   "Which way to go?"          "How fast to get there?" |
|                                                         |
+---------------------------------------------------------+
```

---

## 🌍 Why This Matters

| Concept | Used In | Example |
|---------|---------|---------|
| **Gradient** | All neural networks | Backpropagation in GPT |
| **Hessian** | Newton's method | Fast optimization |
| **Eigenvalues** | PCA, Spectral methods | Dimensionality reduction |
| **Condition number** | Numerical stability | Why training diverges |

---

## 🔗 Dependencies

```
foundations/
    |
    +-- calculus/ --------------+
    |   +-- gradients.md        |
    |   +-- hessian.md          +--> basic-methods/
    |   +-- chain-rule.md       |
    |                           |
    +-- linear-algebra/ --------+
        +-- eigenvalues.md
        +-- positive-definite.md
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd: Convex Optimization | [Book](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal: Numerical Optimization | [Book](https://www.springer.com/gp/book/9780387303031) |
| 🎥 | 3Blue1Brown: Calculus | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) |
| 🇨🇳 | 优化基础数学 | [知乎](https://zhuanlan.zhihu.com/p/25383715) |
| 🇨🇳 | 凸优化基础 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 微积分与线性代数 | [B站](https://www.bilibili.com/video/BV1ys411472E) |

---

⬅️ [Back: Optimization](../) | ➡️ [Next: Basic Methods](../02-basic-methods/)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---
