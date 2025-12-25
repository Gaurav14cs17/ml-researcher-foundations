<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=14 Continual Learning&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🔄 Continual Learning

> **Learning new tasks without forgetting old ones**

---

## 🎯 Visual Overview

<img src="./images/continual-learning-complete.svg" width="100%">

*Caption: Continual learning addresses catastrophic forgetting. Methods include regularization, replay, and architectural approaches.*

---

## 📐 Key Approaches

```
Catastrophic Forgetting:
Training on new task degrades performance on old tasks

Solutions:

1. Regularization (EWC):
   L = L_new + λ Σᵢ F_i (θᵢ - θ*ᵢ)²
   Penalize changes to important parameters

2. Replay:
   Store subset of old data
   Train on mix of old and new

3. Architecture:
   Add new modules for new tasks
   Freeze old modules
```

---

## 💻 Code Example

```python
class EWC:
    def __init__(self, model, dataset, importance=1000):
        self.importance = importance
        self.saved_params = {n: p.clone() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(model, dataset)
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.saved_params[n])**2).sum()
        return self.importance * loss
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | EWC | [arXiv](https://arxiv.org/abs/1612.00796) |
| 🇨🇳 | 持续学习详解 | [知乎](https://zhuanlan.zhihu.com/p/363144973) |

---

⬅️ [Back: 13-Federated](../13-federated-learning/) | ➡️ [Next: 15-RAG](../15-retrieval-augmented/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

