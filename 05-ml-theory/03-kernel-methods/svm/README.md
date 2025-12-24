<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Svm&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Support Vector Machines (SVM)

> **Maximum margin classification**

---

## 🎯 Visual Overview

<img src="./images/svm-maximum-margin.svg" width="100%">

*Caption: SVM finds the hyperplane that maximizes the margin between classes. Support vectors (circled points) are the data points closest to the decision boundary. The kernel trick enables non-linear decision boundaries by mapping data to higher dimensions.*

---

## 📂 Topics in This Folder

| File | Topic | Application |
|------|-------|-------------|

---

## 🎯 The Key Insight

```
Find the hyperplane that MAXIMIZES the margin to nearest points

        ○ ○                    ○ ○
       ○ ○ ○                  ○ ○ ○
      ○ ○ ○ ○      -->       ○ ○ ○ ○
         |                      | ← Maximum margin!
     ● ● |                   ● ●|
    ● ● ●| ●                ● ● |● ●
   ● ● ● |                 ● ● ●|
        Any                  Optimal
     hyperplane             hyperplane
```

---

## 📐 Formulation

### Primal (Hard Margin)

```
minimize    ½||w||²
subject to  yᵢ(wᵀxᵢ + b) ≥ 1   for all i

Margin = 2/||w||, so minimizing ||w|| maximizes margin
```

### Dual (Enables Kernel Trick!)

```
maximize    Σᵢαᵢ - ½ΣᵢΣⱼαᵢαⱼyᵢyⱼxᵢᵀxⱼ
subject to  αᵢ ≥ 0,  Σᵢαᵢyᵢ = 0

Key: Only depends on inner products xᵢᵀxⱼ!
Replace with K(xᵢ, xⱼ) for non-linear SVM
```

---

## 💻 Code Example

```python
from sklearn.svm import SVC
import numpy as np

# Data
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int) * 2 - 1

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X, y)

# RBF kernel SVM (non-linear)
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X, y)

# Support vectors are the points near the margin
print(f"Support vectors: {len(svm_rbf.support_vectors_)}")
```

---

## 📚 Resources

| Type | Title | Link |
|------|-------|------|
| 📄 | Original SVM Paper | Cortes & Vapnik 1995 |
| 🎥 | SVM Explained | MIT OpenCourseWare |


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Kernel Methods](../)

---

⬅️ [Back: Rkhs](../rkhs/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
