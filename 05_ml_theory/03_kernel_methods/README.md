<!-- Navigation -->
<p align="center">
  <a href="../02_generalization/">⬅️ Prev: Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_representation/">Next: Representation ➡️</a>
</p>

---

<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=120&section=header&text=Kernel%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-05-4ECDC4?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Topics

| Folder | Topic | Key Idea |
|--------|-------|----------|
| [kernels/](./kernels/) | Kernel functions | K(x, y) |
| [rkhs/](./rkhs/) | Hilbert spaces | Function spaces |
| [svm/](./svm/) | Support Vector Machines | Max margin |
| [gaussian-processes/](./gaussian-processes/) | Bayesian | Distribution over functions |

---

## 📐 The Kernel Trick

```
Instead of: φ(x)ᵀφ(y) (compute in high-dim)
Compute:    K(x, y)    (direct computation!)

Example - RBF Kernel:
K(x, y) = exp(-||x-y||²/(2σ²))

This corresponds to infinite-dimensional φ!
```

---

## 📊 Common Kernels

| Kernel | Formula | Properties |
|--------|---------|------------|
| Linear | xᵀy | Basic |
| Polynomial | (xᵀy + c)^d | Non-linear |
| RBF/Gaussian | exp(-\|\|x-y\|\|²/2σ²) | Universal |
| Laplacian | exp(-\|\|x-y\|\|/σ) | Sharp |

---

## 🔗 Where This Topic Is Used

| Topic | How Kernel Methods Are Used |
|-------|----------------------------|
| **Attention Mechanism** | QKᵀ is a linear kernel! |
| **Neural Tangent Kernel** | Explains infinite-width NNs |
| **Gaussian Processes** | Bayesian optimization, uncertainty |
| **SVM** | Classification with margin |
| **Kernel PCA** | Non-linear dimensionality reduction |
| **MMD (Max Mean Discrepancy)** | Distribution comparison in GANs |
| **RKHS Regularization** | Theoretical analysis of NNs |
| **Nyström Approximation** | Efficient kernel computation |

### Kernel Ideas in Modern ML

| Modern Concept | Kernel Connection |
|---------------|-------------------|
| **Self-Attention** | Softmax(QKᵀ/√d) ≈ kernel |
| **Linear Attention** | Replace softmax with kernel |
| **Performer** | FAVOR+ kernel approximation |
| **Neural Tangent Kernel** | Infinite-width NN = kernel |
| **Bayesian NN** | Connection to GP |

### Prerequisite For

```
Kernel Methods --> Understanding attention as kernel
              --> Gaussian Processes
              --> Bayesian Optimization
              --> SVM and max-margin theory
              --> RKHS theory in ML papers
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Learning with Kernels | [Springer](https://mitpress.mit.edu/9780262536578/) |
| 📄 | SVM Tutorial | [Paper](https://www.cs.cmu.edu/~./awm/tutorials/svm15.pdf) |
| 🎥 | Kernel Methods Explained | [YouTube](https://www.youtube.com/watch?v=Qc5IyLW_hns) |
| 🇨🇳 | 核方法详解 | [知乎](https://zhuanlan.zhihu.com/p/24291579) |
| 🇨🇳 | SVM与核技巧 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88776412) |
| 🇨🇳 | 核方法讲解 | [B站](https://www.bilibili.com/video/BV1Hs411w7ci) |

---

⬅️ [Back: 02-Generalization](../02_generalization/) | ➡️ [Next: 04-Representation](../04_representation/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<!-- Navigation -->
<p align="center">
  <a href="../02_generalization/">⬅️ Prev: Generalization</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../">📚 ML Theory</a> &nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="../04_representation/">Next: Representation ➡️</a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=4ECDC4&height=80&section=footer" width="100%"/>
</p>
