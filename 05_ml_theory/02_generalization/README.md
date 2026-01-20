<!-- Navigation -->
<p align="center">
  <a href="../01_learning_theory/">⬅️ Prev: Learning Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_kernel_methods/">Next: Kernel Methods ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Generalization%20Theory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Bias-Variance Decomposition

```
For squared error loss:
E[(y - f̂(x))²] = Bias²(f̂) + Var(f̂) + σ²

Where:
• Bias(f̂) = E[f̂(x)] - f(x)  (systematic error)
• Var(f̂) = E[(f̂(x) - E[f̂(x)])²]  (variance across training sets)
• σ² = irreducible noise

```

### Generalization Bound (PAC)

```
With probability ≥ 1-δ:
R(h) ≤ R̂(h) + √(d log(2n/d) + log(1/δ)) / n

Where:
• R(h) = true risk (expected loss)
• R̂(h) = empirical risk (training loss)
• d = VC dimension
• n = sample size

```

### Rademacher Complexity

```
R_n(H) = E_σ[sup_{h∈H} (1/n) Σᵢ σᵢ h(xᵢ)]

Generalization bound:
R(h) ≤ R̂(h) + 2R_n(H) + √(log(1/δ)/2n)

```

---

## 📂 Topics in This Folder

| Folder | Topics | Application |
|--------|--------|-------------|
| [bias-variance/](./bias-variance/) | Error decomposition, tradeoff | Model selection |
| [overfitting/](./overfitting/) | Detection, causes, double descent | Training strategy |
| [complexity/](./complexity/) | VC dimension, Rademacher | Theoretical bounds |
| [regularization/](./regularization/) | L1/L2, dropout, early stopping | Practical techniques |

---

## 🎯 The Central Question of ML

```
We observe: Training loss decreases
We want:    Test loss to decrease too

The gap between them is the GENERALIZATION GAP

+------------------------------------------------------------+

|                                                            |
|  Test Error                                                |
|     |                                                      |
|     |   ▓▓▓▓▓▓▓▓                                          |
|     |   ▓▓▓▓▓▓▓▓▓▓▓▓                                      |
|     |   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  Generalization                  |
|     |   +---------------+     Gap                          |
|     |   |               |                                  |
|     |   |   Train Error |                                  |
|     |   |       ▓▓▓▓▓▓▓▓|▓▓▓▓▓▓▓▓                         |
|     |   |       ▓▓▓▓▓▓▓▓|▓▓▓▓▓▓▓▓▓▓                       |
|     +---+---------------+--------------> Model Complexity  |
|                 ↑                                          |
|            Optimal (min test error)                        |
|                                                            |
+------------------------------------------------------------+

```

---

## 🔥 Double Descent: Modern Phenomenon

```
Classical view: More parameters = more overfitting

Modern reality: After interpolation threshold, test error DECREASES again!

Test
Error  |
       |   ▓
       |  ▓ ▓              Classical
       | ▓   ▓             regime
       |▓     ▓▓▓
       |        ▓▓▓▓
       |            ▓▓▓▓▓▓▓▓▓▓▓    Modern
       |                      ▓▓▓▓▓ regime
       |                           ▓▓▓
       +----------------------------------> Parameters
                  ↑
           Interpolation
           threshold (train error = 0)

Why? Implicit regularization, over-parameterization benefits

```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | Reconciling Modern ML | [arXiv](https://arxiv.org/abs/1812.11118) |
| 📄 | Deep Double Descent | [arXiv](https://arxiv.org/abs/1912.02292) |

---

## 🔗 Where This Topic Is Used

| Topic | How Generalization Theory Is Used |
|-------|-----------------------------------|
| **Model Selection** | Bias-variance tradeoff guides complexity |
| **Regularization** | L1/L2 derived from generalization bounds |
| **Dropout** | Prevents overfitting in NNs |
| **Early Stopping** | Stop when generalization gap grows |
| **Data Augmentation** | Improve generalization with more "data" |
| **Scaling Laws** | Double descent explains LLM behavior |
| **PAC Learning** | Formal generalization guarantees |
| **VC Dimension** | Capacity control |
| **Transfer Learning** | Pretrained models generalize better |
| **Few-shot Learning** | Generalization from limited examples |

### Concepts That Use Generalization

| Concept | Connection |
|---------|------------|
| **Weight Decay** | Regularization → better generalization |
| **Batch Norm** | Implicit regularization effect |
| **Dropout** | Ensemble → better generalization |
| **Data Augmentation** | More diverse training → generalize |
| **Pretraining** | Learn general features first |

### Used To Understand

| Phenomenon | Explained By |
|------------|--------------|
| Why LLMs work | Double descent, overparameterization |
| When to stop training | Bias-variance tradeoff |
| How much data needed | Sample complexity bounds |
| Model capacity | VC dimension, Rademacher complexity |

---

⬅️ [Back: 01-Learning Frameworks](../01_learning_frameworks/) | ➡️ [Next: 03-Kernel Methods](../03_kernel_methods/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../01_learning_theory/">⬅️ Prev: Learning Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../03_kernel_methods/">Next: Kernel Methods ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
