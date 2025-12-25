<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=KKT%20Conditions&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# KKT Conditions

> **K**arush-**K**uhn-**T**ucker — The cornerstone of constrained optimization

## 📊 Visual Intuition

```
Unconstrained:           Constrained:
                         
    minimum at           constraint
    gradient = 0         boundary
        |                    |
        v                    v
    +-------+           +-------+
    |   •   |           | ------|--  g(x)=0
    |  ╱ ╲  |           |   •←--|    optimal HERE
    | ╱   ╲ |           |  ╱ ╲  |    (not at ∇f=0)
    |╱     ╲|           | ╱   ╲ |
    +-------+           +-------+
```

## 📐 The Problem

```
Minimize    f(x)
Subject to  gᵢ(x) ≤ 0    (inequality constraints)
            hⱼ(x) = 0    (equality constraints)
```

## 📜 KKT Conditions

```
+---------------------------------------------------------+
| 1. STATIONARITY                                         |
|    ∇f(x*) = Σᵢ μᵢ ∇gᵢ(x*) + Σⱼ λⱼ ∇hⱼ(x*)              |
|                                                         |
| 2. PRIMAL FEASIBILITY                                   |
|    gᵢ(x*) ≤ 0    for all i                              |
|    hⱼ(x*) = 0    for all j                              |
|                                                         |
| 3. DUAL FEASIBILITY                                     |
|    μᵢ ≥ 0        for all i                              |
|                                                         |
| 4. COMPLEMENTARY SLACKNESS                              |
|    μᵢ · gᵢ(x*) = 0    for all i                         |
+---------------------------------------------------------+
```

## 🎯 Complementary Slackness Explained

```
For each inequality constraint gᵢ(x) ≤ 0:

Either:  μᵢ = 0           (constraint doesn't matter)
Or:      gᵢ(x*) = 0       (constraint is active/tight)
Never:   μᵢ > 0 AND gᵢ < 0 (impossible!)

Visual:
   |
   | constraint      μᵢ > 0, gᵢ = 0
   | active ------------•-----
   |                    |
   |                    | optimal
   | constraint         | on boundary
   | inactive           |
   | (μᵢ = 0)          |
```

## 📋 Step-by-Step Process

```
1. Identify active constraints (guess which gᵢ = 0)
2. Write stationarity: ∇f = Σμᵢ∇gᵢ + Σλⱼ∇hⱼ
3. Solve system with active constraints
4. Check: μᵢ ≥ 0 for active constraints
5. Check: gᵢ(x*) ≤ 0 for inactive constraints
6. If any check fails, try different active set
```

## ⚙️ Constraint Qualification

```
KKT conditions are NECESSARY only when:

LICQ (Linear Independence CQ):
  Active constraint gradients are linearly independent
  
Slater's Condition (for convex):
  ∃ x such that gᵢ(x) < 0 for all i (strictly feasible)
```

## 💡 Sufficiency

```
+---------------------------------------------+
| KKT is SUFFICIENT when:                     |
|                                             |
|   • f(x) is convex                          |
|   • gᵢ(x) are convex                        |
|   • hⱼ(x) are affine (linear)              |
|                                             |
| Then: KKT point = Global optimum            |
+---------------------------------------------+
```

## 🔬 Example: SVM

```
SVM optimization uses KKT conditions!

min  ½||w||²
s.t. yᵢ(wᵀxᵢ + b) ≥ 1

KKT gives:
• Support vectors: points where μᵢ > 0
• These are exactly points on the margin
• Complementarity: μᵢ[yᵢ(wᵀxᵢ + b) - 1] = 0
```

## 📚 Resources

| Type | Resource | Link |
|------|----------|------|
| 📄 Paper | Original Kuhn-Tucker | Classic (1951) |
| 📖 Book | Boyd CVX Ch.5 | [Free PDF](https://web.stanford.edu/~boyd/cvxbook/) |
| 📖 Book | Nocedal Ch.12 | [Springer](https://link.springer.com/book/10.1007/978-0-387-40065-5) |
| 🎥 Video | KKT Explained | [YouTube](https://www.youtube.com/watch?v=uh1Dk68cfWs) |
| 🇨🇳 知乎 | KKT条件详解 | [知乎](https://zhuanlan.zhihu.com/p/38163970) |
| 🇨🇳 CSDN | KKT条件推导 | [CSDN](https://blog.csdn.net/johnnyconstantine/article/details/46335763) |
| 🇨🇳 B站 | 拉格朗日与KKT | [B站](https://www.bilibili.com/video/BV1HP4y1Y79e) |

---

---

➡️ [Next: Lagrange](./lagrange.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
