<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=04 Estimation&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 📊 Estimation

> **Learning parameters from data**

---

## 🎯 Visual Overview

<img src="./images/estimation.svg" width="100%">

*Caption: Statistical estimation infers unknown parameters θ from data. MLE maximizes likelihood, MAP adds prior (= regularization), Bayesian computes full posterior distribution for uncertainty quantification.*

---

## 📐 Mathematical Foundations

### Maximum Likelihood Estimation
```
θ_MLE = argmax_θ L(θ) = argmax_θ Πᵢ p(xᵢ|θ)

Log-likelihood (more stable):
θ_MLE = argmax_θ ℓ(θ) = argmax_θ Σᵢ log p(xᵢ|θ)
```

### Examples
```
Gaussian (known σ):
p(x|μ) = (2πσ²)^(-1/2) exp(-(x-μ)²/(2σ²))
μ_MLE = (1/n) Σᵢ xᵢ  (sample mean)

Bernoulli:
p(x|θ) = θˣ(1-θ)^(1-x)
θ_MLE = (1/n) Σᵢ xᵢ  (sample proportion)
```

### MAP Estimation
```
θ_MAP = argmax_θ p(θ|D) = argmax_θ p(D|θ)p(θ)

Log form:
θ_MAP = argmax_θ [Σᵢ log p(xᵢ|θ) + log p(θ)]
                  -----------------   ---------
                  likelihood          prior (regularization!)

Gaussian prior → L2 regularization
Laplace prior → L1 regularization
```

### Bayesian Posterior
```
p(θ|D) = p(D|θ)p(θ) / ∫ p(D|θ')p(θ') dθ'

Predictive distribution:
p(x_new|D) = ∫ p(x_new|θ)p(θ|D) dθ
```

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [mle/](./mle/) | Maximum Likelihood | argmax p(data\|θ) |
| [map/](./map/) | Maximum A Posteriori | argmax p(θ\|data) |
| [bayesian/](./bayesian/) | Full Bayesian | Full p(θ\|data) |

---

## 📊 Comparison

```
MLE:  θ* = argmax_θ p(data|θ)
MAP:  θ* = argmax_θ p(data|θ)p(θ)
Bayesian: p(θ|data) = p(data|θ)p(θ)/p(data)

MLE     → Point estimate, no regularization
MAP     → Point estimate with regularization
Bayesian → Full uncertainty quantification
```

---

## 🔥 MLE = Training!

```
Minimizing cross-entropy loss
= Maximizing log-likelihood
= MLE!

L = -Σᵢ log p(yᵢ|xᵢ; θ)
θ* = argmin_θ L = argmax_θ Σᵢ log p(yᵢ|xᵢ; θ)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Statistical Inference | Casella & Berger |
| 📖 | Pattern Recognition (Bishop) | [PRML](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 🎥 | MLE/MAP Explained | [YouTube](https://www.youtube.com/watch?v=XepXtl9YKwc) |
| 🇨🇳 | 极大似然估计 | [知乎](https://zhuanlan.zhihu.com/p/26614750) |
| 🇨🇳 | MLE与MAP对比 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: 03-Information Theory](../03-information-theory/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
