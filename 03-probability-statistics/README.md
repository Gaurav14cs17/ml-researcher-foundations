<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=120&section=header&text=Topic&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-03-9C27B0?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📊 Learning Path

```mermaid
graph LR
    A[🚀 Start] --> B[🎲 Probability]
    B --> C[📊 Distributions]
    C --> D[🔮 Bayes]
    D --> E[📈 Multivariate]
    E --> F[📡 Info Theory]
    F --> G[✅ Ready]
```

## 🎯 What You'll Learn

> 💡 Machine learning is fundamentally about **learning from uncertain data**.

<table>
<tr>
<td align="center">

### 🎲 Probability
Bayes, Distributions

</td>
<td align="center">

### 📡 Information Theory
Entropy, KL Divergence

</td>
<td align="center">

### 📊 Estimation
MLE, MAP, Bayesian

</td>
</tr>
</table>

---

## 📚 Main Topics

### 1️⃣ Probability Theory

<img src="https://img.shields.io/badge/Time-8_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/Priority-⭐⭐⭐-gold?style=flat-square"/>

```mermaid
graph LR
    A[Sample Space] --> B[Events]
    B --> C[Conditional]
    C --> D[Bayes]
    D --> E[Random Vars]
    E --> F[Distributions]
```

**Core:** Bayes' Theorem, Gaussian, Bernoulli, Expectation, Variance

<a href="./01-probability/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-9C27B0?style=for-the-badge" alt="Learn"/></a>

---

### 2️⃣ Multivariate Statistics

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Multi Vars] --> B[Joint Dist]
    B --> C[Covariance]
    C --> D[MVN]
    D --> E[Correlation]
    E --> F[PCA Link]
```

**Core:** Joint Distributions, Covariance Matrix, Multivariate Gaussian

<a href="./02-multivariate/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-9C27B0?style=for-the-badge" alt="Learn"/></a>

---

### 3️⃣ Information Theory

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_Essential-critical?style=flat-square"/>

```mermaid
graph LR
    A[Information] --> B[Entropy]
    B --> C[Cross-Entropy]
    C --> D[KL Divergence]
    D --> E[Mutual Info]
    E --> F[Loss Funcs]
```

> 🔥 **Cross-entropy is THE loss function** for classification

**Core:** Entropy H(X), Cross-Entropy, KL Divergence (VAE, RLHF)

<a href="./03-information-theory/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-9C27B0?style=for-the-badge" alt="Learn"/></a>

---

### 4️⃣ Statistical Estimation

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Estimation] --> B[MLE]
    B --> C[MAP]
    C --> D[Bayesian]
    D --> E[Confidence]
```

**Core:** MLE (Training), MAP (Regularization), Bayesian Inference

<a href="./04-estimation/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-9C27B0?style=for-the-badge" alt="Learn"/></a>

---

## 💡 Key Formulas

<table>
<tr>
<td>

### 🎲 Probability
```
Bayes: P(A|B) = P(B|A)P(A)/P(B)
E[X] = Σ x·p(x)
Var(X) = E[(X - E[X])²]
```

</td>
<td>

### 📡 Information Theory
```
H(X) = -Σ p(x)log p(x)
H(p,q) = -Σ p(x)log q(x)
KL(p||q) = Σ p(x)log[p(x)/q(x)]
```

</td>
</tr>
</table>

---

## 🔗 ML Applications

| Concept | Application | Used In |
|:-------:|-------------|---------|
| 🔮 **Bayes** | Posterior inference | Bayesian NN |
| 📊 **Entropy** | Decision trees | ID3, C4.5 |
| 📈 **Cross-Entropy** | Classification | All classifiers |
| 🔀 **KL Divergence** | Variational inference | VAE, RLHF |

---

## 🔗 Prerequisites & Next Steps

```mermaid
graph LR
    A[📊 Math] --> B[📈 Prob/Stats]
    B --> C[🎯 Optimization]
    C --> D[🧬 ML Theory]
    D --> E[🚀 Deep Learning]
```

<p align="center">
  <a href="../02-mathematics/README.md"><img src="https://img.shields.io/badge/←_Prerequisites:_Mathematics-gray?style=for-the-badge" alt="Prev"/></a>
  <a href="../04-optimization/README.md"><img src="https://img.shields.io/badge/Next:_Optimization_→-00C853?style=for-the-badge" alt="Next"/></a>
</p>

---

## 📚 Recommended Resources

| Type | Resource | Focus |
|:----:|----------|-------|
| 📘 | [Pattern Recognition & ML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) | Bishop's classic |
| 📘 | [Information Theory](http://www.inference.org.uk/itprnn/book.pdf) | David MacKay |
| 🎓 | [MIT 6.041](https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/) | Probability |

---

## 🗺️ Quick Navigation

| Previous | Current | Next |
|:--------:|:-------:|:----:|
| [📊 Mathematics](../02-mathematics/README.md) | **📈 Probability** | [🎯 Optimization →](../04-optimization/README.md) |

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=9C27B0&height=80&section=footer" width="100%"/>
</p>
