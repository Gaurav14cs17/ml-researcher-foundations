<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=12 Meta-Learning&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🧠 Meta-Learning

> **Learning to learn - adapting quickly to new tasks**

---

## 🎯 Visual Overview

<img src="./images/meta-learning-complete.svg" width="100%">

*Caption: Meta-learning trains on many tasks to learn good initialization or learning strategy. Few-shot learning is a key application.*

---

## 📐 Key Approaches

```
MAML (Model-Agnostic Meta-Learning):
θ* = θ - α∇_θ Σᵢ Lᵢ(θ - β∇_θLᵢ(θ))

Learn initialization that adapts quickly with few gradient steps

Prototypical Networks:
Compute class prototypes from support set
Classify query by distance to prototypes
```

---

## 💻 Code Example

```python
def maml_inner_loop(model, support_x, support_y, inner_lr=0.01):
    """One MAML inner loop step"""
    # Clone model parameters
    fast_weights = [p.clone() for p in model.parameters()]
    
    # Inner loop gradient step
    loss = compute_loss(model, support_x, support_y)
    grads = torch.autograd.grad(loss, model.parameters())
    
    fast_weights = [p - inner_lr * g for p, g in zip(fast_weights, grads)]
    
    return fast_weights
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | MAML | [arXiv](https://arxiv.org/abs/1703.03400) |
| 📄 | Prototypical Networks | [arXiv](https://arxiv.org/abs/1703.05175) |
| 🇨🇳 | 元学习详解 | [知乎](https://zhuanlan.zhihu.com/p/28639662) |

---

⬅️ [Back: 11-Multi-Task](../11-multi-task-learning/) | ➡️ [Next: 13-Federated](../13-federated-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

