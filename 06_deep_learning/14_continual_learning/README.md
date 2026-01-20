<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Continual%20Learning&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

⬅️ [Back: Federated Learning](../13_federated_learning/README.md) | ➡️ [Next: RAG](../15_retrieval_augmented/README.md)

---

⬅️ [Back: Main](../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
