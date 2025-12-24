<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=18,19,20&height=180&section=header&text=🎯%20ML%20Theory&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=Understand%20Why%20Things%20Work&descAlignY=52&descSize=18" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/📚_Topics-4_Modules-blue?style=for-the-badge" alt="Topics"/>
  <img src="https://img.shields.io/badge/⏱️_Time-2_Weeks-green?style=for-the-badge" alt="Time"/>
  <img src="https://img.shields.io/badge/📊_Level-Advanced-red?style=for-the-badge" alt="Level"/>
</p>

<p align="center">
  <a href="#-main-topics"><img src="https://img.shields.io/badge/Start_Learning-00BCD4?style=for-the-badge&logo=rocket&logoColor=white" alt="Start"/></a>
  <a href="../06-deep-learning/README.md"><img src="https://img.shields.io/badge/Next:_Deep_Learning-00C853?style=for-the-badge&logo=arrow-right&logoColor=white" alt="Next"/></a>
</p>

---

**✍️ Author:** [Gaurav Goswami](https://github.com/Gaurav14cs17) • **📅 Updated:** December 2024

---

## 📊 Learning Path

```mermaid
graph LR
    A[🚀 Start] --> B[📚 Frameworks]
    B --> C[ERM/PAC]
    C --> D[📈 Generalize]
    D --> E[Bias-Var]
    E --> F[🎯 Kernels]
    F --> G[🏆 Master]
```

## 🎯 What You'll Learn

> 💡 ML Theory explains **why algorithms work** and when they fail.

<table>
<tr>
<td align="center">

### 📊 Generalization
⭐ **THE GOAL**

</td>
<td align="center">

### ⚖️ Bias-Variance
Model selection

</td>
<td align="center">

### 🎯 Kernels
SVM, Feature spaces

</td>
</tr>
</table>

---

## 📚 Main Topics

### 1️⃣ Learning Frameworks

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Supervised] --> B[ERM]
    B --> C[PAC]
    C --> D[Sample Size]
    D --> E[Bounds]
```

**Core:** ERM, PAC Learning, Sample Complexity, No Free Lunch

<a href="./01-learning-frameworks/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-00BCD4?style=for-the-badge" alt="Learn"/></a>

---

### 2️⃣ Generalization ⭐⭐⭐

<img src="https://img.shields.io/badge/Time-8_hours-blue?style=flat-square"/> <img src="https://img.shields.io/badge/🔥_MOST_IMPORTANT-critical?style=flat-square"/>

```mermaid
graph LR
    A[Train Err] --> B[Test Err]
    B --> C[Gap]
    C --> D[Bias-Var]
    D --> E[Overfit]
    E --> F[Regularize]
```

> ⭐ **GENERALIZATION IS THE GOAL** - This is why ML works

| Concept | Impact |
|---------|--------|
| Bias-Variance | Model complexity choice |
| VC Dimension | Capacity measure |
| Regularization | L1, L2, Dropout |

<a href="./02-generalization/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-00BCD4?style=for-the-badge" alt="Learn"/></a>

---

### 3️⃣ Kernel Methods

<img src="https://img.shields.io/badge/Time-6_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[Features] --> B[Kernel Trick]
    B --> C[Kernels]
    C --> D[SVM]
    D --> E[RKHS]
```

**Core:** Kernel Trick, RBF/Polynomial Kernels, SVM, Gaussian Processes

<a href="./03-kernel-methods/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-00BCD4?style=for-the-badge" alt="Learn"/></a>

---

### 4️⃣ Risk Minimization

<img src="https://img.shields.io/badge/Time-4_hours-blue?style=flat-square"/>

```mermaid
graph LR
    A[True Risk] --> B[Empirical]
    B --> C[Structural]
    C --> D[Selection]
    D --> E[CV]
```

**Core:** True vs Empirical Risk, Cross-Validation, Model Selection

<a href="./05-risk-minimization/README.md"><img src="https://img.shields.io/badge/📖_Dive_In-00BCD4?style=for-the-badge" alt="Learn"/></a>

---

## 🔄 Bias-Variance Tradeoff

```mermaid
graph TD
    A[Simple Model] --> B[High Bias]
    B --> C[❌ Underfit]
    D[Complex Model] --> E[High Variance]
    E --> F[❌ Overfit]
    G[🎯 Sweet Spot] --> H[Balanced]
    H --> I[✅ Generalize]
```

### Key Equation
```
Expected Test Error = Bias² + Variance + Irreducible Error
```

---

## 💡 Key Concepts

<table>
<tr>
<td>

### 📊 Generalization Bound
```
P(|R_true - R_emp| > ε) ≤ δ
n ≥ O((VC_dim + log(1/δ))/ε²)
```

</td>
<td>

### ⚖️ Bias-Variance
```
E[(y - ŷ)²] = Bias² + Var + σ²
```

</td>
<td>

### 🎯 SVM
```
min  ½||w||²
s.t. yᵢ(w·xᵢ + b) ≥ 1
```

</td>
</tr>
</table>

---

## 🔗 Prerequisites & Next Steps

```mermaid
graph LR
    A[🎯 Optimization] --> B[🧬 ML Theory]
    B --> C[🚀 Deep Learning]
    C --> D[Neural Nets]
    D --> E[Training]
```

<p align="center">
  <a href="../04-optimization/README.md"><img src="https://img.shields.io/badge/←_Prerequisites:_Optimization-gray?style=for-the-badge" alt="Prev"/></a>
  <a href="../06-deep-learning/README.md"><img src="https://img.shields.io/badge/Next:_Deep_Learning_→-00C853?style=for-the-badge" alt="Next"/></a>
</p>

---

## 📚 Recommended Resources

| Type | Resource | Focus |
|:----:|----------|-------|
| 📘 | [Understanding ML](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) | Theory foundations |
| 🎓 | [Caltech CS156](https://work.caltech.edu/telecourse.html) | ML Theory |
| 📄 | [SVM Tutorial](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-04.pdf) | Burges (1998) |

---

## 🗺️ Quick Navigation

| Previous | Current | Next |
|:--------:|:-------:|:----:|
| [🎯 Optimization](../04-optimization/README.md) | **🧬 ML Theory** | [🚀 Deep Learning →](../06-deep-learning/README.md) |

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=18,19,20&height=100&section=footer" width="100%"/>
</p>
