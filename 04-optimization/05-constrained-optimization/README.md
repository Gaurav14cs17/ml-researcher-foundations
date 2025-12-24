# 🔒 Constrained Optimization

> **Optimizing with limits and boundaries**

## 🎯 Visual Overview

<img src="./images/constrained-opt.svg" width="100%">

*Caption: Constrained optimization minimizes f(x) subject to g(x)≤0, h(x)=0. Lagrangian L = f + λᵀg + νᵀh. KKT conditions characterize optimality for convex problems.*

---

## 📂 Topics in This Folder

| Folder | Topic | Key Concept |
|--------|-------|-------------|
| [lagrange/](./lagrange/) | Lagrange Multipliers | Equality constraints |
| [kkt/](./kkt/) | KKT Conditions | Inequality constraints |

---

## 🎯 The Problem

```
Unconstrained:              Constrained:
                            
   Find x that               Find x that
   minimizes f(x)            minimizes f(x)
                             SUBJECT TO:
   • Go anywhere             • g(x) ≤ 0 (inequality)
                             • h(x) = 0 (equality)
                             
   ↓                         ↓
   
   Just set ∇f = 0           Need Lagrange/KKT!
```

---

## 📐 The Two Main Tools

```
+---------------------------------------------------------+
|                                                         |
|   LAGRANGE MULTIPLIERS              KKT CONDITIONS      |
|   --------------------              --------------      |
|                                                         |
|   For: h(x) = 0 only                For: g(x) ≤ 0 AND  |
|   (equality)                         h(x) = 0           |
|                                                         |
|   L = f(x) - λh(x)                  + Complementarity:  |
|                                     μᵢgᵢ(x) = 0         |
|   Solve: ∇L = 0                     μᵢ ≥ 0              |
|                                                         |
+---------------------------------------------------------+
```

---

## 🌍 Real-World Applications

| Application | Constraint Type | Example |
|-------------|-----------------|---------|
| **SVM** | Inequality | Margin ≥ 1 for all points |
| **Portfolio** | Equality + Inequality | Weights sum to 1, non-negative |
| **Physics** | Equality | Conserve energy/momentum |
| **RL (TRPO, PPO)** | KL constraint | Trust region |
| **Optimal Control** | Dynamics as equality | Trajectory optimization |

---

## 📊 Visual Comparison

```
Unconstrained:            Constrained:

    ∇f = 0                  ∇f = λ∇g
       |                        |
       ↓                        ↓
       •                        • ← on boundary!
      ╱ ╲                      -+-
     ╱   ╲                   ╱  |  ╲  feasible
    ╱     ╲                 ╱   |   ╲ region
   
   Interior optimum        Boundary optimum
```

---

## 💡 Key Insight: Shadow Prices

```
The Lagrange multiplier λ has economic meaning:

λ* = ∂f*/∂b

"How much would the optimal value improve
 if we relaxed the constraint by 1 unit?"

Example:
• Constraint: budget ≤ $100
• λ* = 5
• Meaning: $1 more budget → $5 more profit
• This is the "shadow price" of money!
```

---

## 🔗 Dependencies

```
foundations/calculus
         |
         ↓
basic-methods/gradient-descent
         |
         ↓
+--------+--------+
| CONSTRAINED OPT |
+-----------------+
| • lagrange/     |--> Used in SVM!
| • kkt/          |--> Used in RLHF!
+--------+--------+
         |
         ↓
    Interior Point Methods
    (linear-programming/)
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📖 | Boyd CVX Ch.5 | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 | Nocedal Ch.12 | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 | KKT Conditions | [YouTube](https://www.youtube.com/watch?v=uh1Dk68cfWs) |
| 🇨🇳 | 知乎 KKT条件 | [知乎](https://zhuanlan.zhihu.com/p/38163970) |
| 🇨🇳 | B站 拉格朗日 | [B站](https://www.bilibili.com/video/BV1HP4y1Y79e) |

---

⬅️ [Back: Convex Optimization](../04-convex-optimization/) | ➡️ [Next: Linear Programming](../06-linear-programming/)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---
