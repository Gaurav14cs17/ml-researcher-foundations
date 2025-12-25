<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=13 Federated Learning&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# 🌐 Federated Learning

> **Training models across decentralized data**

---

## 🎯 Visual Overview

<img src="./images/federated-learning-complete.svg" width="100%">

*Caption: Federated learning trains on distributed data without centralizing it. Clients train locally, server aggregates model updates.*

---

## 📐 FedAvg Algorithm

```
For each round t:
1. Server sends global model w_t to selected clients
2. Each client k:
   • Trains locally: w_k ← w_t - η∇L_k(w_t)
   • Sends update Δw_k = w_k - w_t to server
3. Server aggregates:
   w_{t+1} = w_t + Σ_k (n_k/n) Δw_k

Challenges:
• Non-IID data across clients
• Communication efficiency
• Privacy preservation
```

---

## 💻 Code Example

```python
def fed_avg(client_models, client_sizes):
    """Federated averaging"""
    total = sum(client_sizes)
    
    # Weighted average of parameters
    avg_params = {}
    for key in client_models[0].state_dict():
        avg_params[key] = sum(
            model.state_dict()[key] * (size/total)
            for model, size in zip(client_models, client_sizes)
        )
    
    return avg_params
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | FedAvg | [arXiv](https://arxiv.org/abs/1602.05629) |
| 🇨🇳 | 联邦学习详解 | [知乎](https://zhuanlan.zhihu.com/p/100688371) |

---

⬅️ [Back: 12-Meta-Learning](../12-meta-learning/) | ➡️ [Next: 14-Continual](../14-continual-learning/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

