<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Overfitting&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Overfitting

> **When your model memorizes instead of learning**

---

## 🎯 Visual Overview

<img src="./images/overfitting-visual.svg" width="100%">

*Caption: Underfitting (high bias) fails to capture patterns; overfitting (high variance) memorizes noise. The goal is finding the sweet spot where the model generalizes well to unseen data.*

---

## 📂 Overview

Overfitting occurs when a model learns the training data too well, including its noise and outliers, leading to poor generalization. This is one of the most critical problems in machine learning.

---

## 📐 Mathematical Framework

### Generalization Error Decomposition

```
E[(f(x) - y)²] = Bias² + Variance + Irreducible Noise

Bias² = E[f̂(x) - f(x)]²     ← Underfitting
Variance = E[(f̂(x) - E[f̂(x)])²]   ← Overfitting
Noise = σ²                    ← Cannot reduce
```

### Training vs Test Error

```
Overfitting Indicator:

    Train Error << Test Error
    
    +---------------------------------+
    |  Error                          |
    |   ^                             |
    |   |    ╲                        |
    |   |     ╲___Test Error___/      |
    |   |      ╲              /       |
    |   |       ╲___________╱         |
    |   |        Training Error       |
    |   |                             |
    |   +-------------------->        |
    |              Model Complexity   |
    +---------------------------------+
```

### Model Complexity & VC Dimension

```
Generalization Bound (VC Theory):

    Test Error ≤ Train Error + O(√(d/n))
    
    where:
    d = VC dimension (model complexity)
    n = number of training samples
    
High d/n ratio → High overfitting risk
```

---

## 🔑 Signs of Overfitting

| Sign | Training | Validation | Diagnosis |
|------|----------|------------|-----------|
| **Perfect fit** | ~0% error | High error | Severe overfitting |
| **Increasing val loss** | Decreasing | Increasing | Start early stopping |
| **Large weights** | Good fit | Poor fit | Need regularization |
| **Low training loss** | Very low | Much higher | Model too complex |

---

## 🛡️ Solutions

### 1. Regularization

```python
# L2 Regularization (Weight Decay)
loss = cross_entropy(y, ŷ) + λ * ||w||²

# L1 Regularization (Sparsity)
loss = cross_entropy(y, ŷ) + λ * ||w||₁

# Dropout
h = mask * h / (1 - p)  # During training
```

### 2. Early Stopping

```python
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    train_loss = train(model)
    val_loss = validate(model)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # Early stop!
```

### 3. Data Augmentation

```python
# More data reduces overfitting
transforms = [
    RandomCrop(224),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.2),
    Normalize(mean, std)
]
```

### 4. Cross-Validation

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True)
for train_idx, val_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[val_idx], y[val_idx])
```

---

## 💻 Code: Detecting Overfitting

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(train_losses, val_losses):
    """Visualize overfitting via learning curves"""
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    
    # Find overfitting point
    gap = np.array(val_losses) - np.array(train_losses)
    overfit_start = np.argmax(gap > 0.1)  # Threshold
    
    if overfit_start > 0:
        plt.axvline(x=overfit_start, color='g', linestyle='--', 
                    label=f'Overfitting starts at epoch {overfit_start}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Learning Curves: Detecting Overfitting')
    plt.show()
```

---

## 🔗 Connection to Other Topics

```
Overfitting
    +-- Bias-Variance Tradeoff (theory)
    +-- Regularization (solution)
    |   +-- L1/L2 penalties
    |   +-- Dropout
    +-- Cross-Validation (detection)
    +-- VC Dimension (bounds)
    +-- Early Stopping (practice)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bias-Variance Tradeoff | [../bias-variance/](../bias-variance/) |
| 📖 | Regularization Methods | [../regularization/](../regularization/) |
| 📖 | VC Dimension Theory | [../vc-dimension/](../vc-dimension/) |
| 📄 | ESL Ch. 7 | [Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) |
| 🇨🇳 | 过拟合与欠拟合详解 | [知乎](https://zhuanlan.zhihu.com/p/72038532) |
| 🇨🇳 | 正则化防止过拟合 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/87719528) |
| 🇨🇳 | 吴恩达机器学习 | [B站](https://www.bilibili.com/video/BV164411b7dx)


## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

⬅️ [Back: Generalization](../)

---

⬅️ [Back: Complexity](../complexity/) | ➡️ [Next: Regularization](../regularization/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
