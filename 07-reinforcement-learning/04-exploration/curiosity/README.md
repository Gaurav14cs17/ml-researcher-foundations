<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Curiosity&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Curiosity-Driven Exploration

> **Learning by being surprised**

---

## 🎯 Visual Overview

<img src="./images/curiosity-driven.svg" width="100%">

*Caption: ICM (Intrinsic Curiosity Module) generates curiosity rewards from prediction error. The forward model predicts next state features; high prediction error means surprise, which becomes intrinsic reward.*

---

## 📂 Overview

Curiosity-driven exploration rewards the agent for encountering surprising outcomes that it cannot predict. This solves hard exploration problems without any external reward.

---

## 📐 ICM Architecture

```
Intrinsic Curiosity Module:

1. Feature Encoder φ: s → feature space
2. Forward Model: predicts φ(s_{t+1}) from φ(s_t) and a_t
3. Inverse Model: predicts a_t from φ(s_t) and φ(s_{t+1})

Curiosity Reward:
r_i = ||φ̂(s_{t+1}) - φ(s_{t+1})||²

High prediction error = Surprising = High reward
```

---

## 🔑 Key Features

| Feature | Description |
|---------|-------------|
| **Feature Space** | Ignores noise (TV static problem) |
| **Self-supervised** | No labels needed |
| **Scalable** | Works with high-dim states |
| **Sparse Reward** | Solves Montezuma's Revenge |

---

## 🌍 Results

| Environment | Without Curiosity | With Curiosity |
|-------------|-------------------|----------------|
| Montezuma's Revenge | 0 | 11,500 |
| VizDoom | Random | Explores map |
| Mario | Stuck at start | Completes levels |

---

## 💻 Code

```python
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Forward model: predict next features
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def curiosity_reward(self, state, action, next_state):
        phi_s = self.encoder(state)
        phi_s_next = self.encoder(next_state)
        
        # Predict next state features
        action_onehot = F.one_hot(action, num_classes=self.action_dim)
        phi_s_next_pred = self.forward_model(torch.cat([phi_s, action_onehot], dim=-1))
        
        # Curiosity = prediction error
        return ((phi_s_next_pred - phi_s_next.detach()) ** 2).mean(dim=-1)
```


## 🔗 Where This Topic Is Used

| Application | Curiosity |
|-------------|----------|
| **ICM** | Prediction error as reward |
| **RND** | Random network distillation |
| **Hard Games** | Sparse reward navigation |
| **Lifelong Learning** | Continuous exploration |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: Exploration](../)

---

➡️ [Next: Epsilon Greedy](../epsilon-greedy/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
