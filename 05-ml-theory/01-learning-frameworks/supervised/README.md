# Supervised Learning

> **Learning from labeled data**

---

## 🎯 Visual Overview

<img src="./images/supervised.svg" width="100%">

*Caption: Supervised learning uses labeled training data (x,y) pairs to learn a function f(x)→y. Tasks include classification (discrete y) and regression (continuous y). The goal is to generalize to unseen data.*

---

## 📂 Overview

Supervised learning is the most common ML paradigm, where models learn from labeled examples. Given input-output pairs, the model learns to predict outputs for new inputs.

---

## 📐 Mathematical Foundations

### Supervised Learning Problem
```
Given: Training set D = {(x₁,y₁), ..., (xₙ,yₙ)}
Find: f: X → Y minimizing expected loss

f* = argmin_f E_{(x,y)~P} [ℓ(f(x), y)]
```

### Classification Loss
```
Cross-entropy:
L = -Σᵢ yᵢ log(p̂ᵢ)  (multi-class)
L = -[y log(p̂) + (1-y) log(1-p̂)]  (binary)

Where p̂ = softmax(f(x)) or σ(f(x))
```

### Regression Loss
```
MSE: L = (1/n) Σᵢ (yᵢ - f(xᵢ))²
MAE: L = (1/n) Σᵢ |yᵢ - f(xᵢ)|
Huber: L = { ½(y-f)² if |y-f| ≤ δ
           { δ|y-f| - ½δ² otherwise
```

---

## 🔑 Key Topics

| Topic | Description |
|-------|-------------|
| Fundamentals | Core concepts |
| Applications | ML use cases |
| Implementation | Code examples |

---

## 💻 Example

```python
import torch.nn as nn
import torch.optim as optim

# Classification
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Regression  
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
criterion = nn.MSELoss()
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Bishop PRML | [Book](https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/) |
| 📖 | ESL | [Book](https://hastie.su.domains/ElemStatLearn/) |
| 🎓 | Stanford CS229 | [Course](http://cs229.stanford.edu/) |
| 🇨🇳 | 监督学习详解 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 分类与回归 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | 机器学习入门 | [B站](https://www.bilibili.com/video/BV164411b7dx) |

---

<- [Back](../)

---

## 🔗 Where This Topic Is Used

| Application | Usage |
|-------------|-------|
| **Machine Learning** | Core concept for ML systems |
| **Deep Learning** | Foundation for neural networks |
| **Research** | Important for understanding papers |

---

---

⬅️ [Back: supervised](../)

---

⬅️ [Back: Self Supervised](../self-supervised/) | ➡️ [Next: Unsupervised](../unsupervised/)
