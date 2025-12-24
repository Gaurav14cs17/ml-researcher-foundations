<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=01 Probability&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🎲 Probability Theory

> **The foundation of machine learning**

---

## 🎯 Visual Overview

<img src="./images/probability-ml-view.svg" width="100%">

*Caption: Probability provides the mathematical framework for ML. P(Y|X,θ) defines predictions, P(D|θ) is the likelihood we maximize in training (MLE), and P(θ|D) is the posterior in Bayesian learning. Bayes' theorem connects them all.*

---

## 📂 Topics in This Folder

| Folder | Topics | ML Application |
|--------|--------|----------------|
| [spaces/](./spaces/) | Sample space, σ-algebra, measure | Rigorous foundations |
| [random-variables/](./random-variables/) | Discrete, continuous, mixed | Data modeling |
| [distributions/](./distributions/) | PMF, PDF, CDF, common distributions | Likelihood |
| [expectation/](./expectation/) | Mean, variance, covariance | Loss functions |
| [conditional/](./conditional/) | Bayes theorem, independence | 🔥 Bayesian ML |
| [limit-theorems/](./limit-theorems/) | LLN, CLT, concentration | Generalization |

---

## 📐 Mathematical Foundations

### Probability Axioms (Kolmogorov)
```
1. P(A) ≥ 0 for all events A
2. P(Ω) = 1 (certainty)
3. P(⋃ᵢAᵢ) = Σᵢ P(Aᵢ) for disjoint Aᵢ
```

### Bayes' Theorem
```
P(θ|D) = P(D|θ)P(θ) / P(D)

Posterior = (Likelihood × Prior) / Evidence
```

### Expectation and Variance
```
Discrete:   E[X] = Σₓ x P(X=x)
Continuous: E[X] = ∫ x f(x) dx

Var(X) = E[(X - E[X])²] = E[X²] - E[X]²

Properties:
E[aX + b] = aE[X] + b
Var(aX + b) = a²Var(X)
```

### Law of Total Probability
```
P(A) = Σᵢ P(A|Bᵢ)P(Bᵢ)  (partition)

Continuous:
P(Y) = ∫ P(Y|X=x)p(x) dx
```

---

## 🎯 The Probabilistic View of ML

```
Machine Learning as Probability:

Data:    D = {(x₁,y₁), ..., (xₙ,yₙ)}
Model:   p(y|x,θ) - probabilistic prediction
Goal:    Find θ that maximizes p(D|θ)  ← MLE!

This perspective gives us:
• Loss functions (negative log-likelihood)
• Regularization (priors)
• Uncertainty quantification
• Generalization theory
```

---

## 🌍 Why Probability for ML?

| Concept | ML Translation |
|---------|----------------|
| P(Y\|X) | Prediction |
| P(D\|θ) | Likelihood (training objective) |
| P(θ\|D) | Posterior (Bayesian learning) |
| E[L] | Expected loss (what we minimize) |
| Var(∇L) | Gradient variance (SGD noise) |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Probability Theory (Jaynes) | [Book](https://bayes.wustl.edu/etj/prob/book.pdf) |
| 📖 | Pattern Recognition (Bishop) | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | Probability Primer | [YouTube](https://www.youtube.com/playlist?list=PLC58778F28211FA19) |
| 🇨🇳 | 概率论基础 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 机器学习概率视角 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 概率论讲解 | [B站](https://www.bilibili.com/video/BV164411b7dx) |


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Probability & Statistics](../) | ➡️ [Next: 02-Multivariate](../02-multivariate/)



---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
