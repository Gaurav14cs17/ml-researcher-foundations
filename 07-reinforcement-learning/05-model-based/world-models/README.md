<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=World%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="../images/world-models.svg" width="100%">

*Caption: World models learn to predict future states and rewards given actions. The agent can then "dream" - simulate trajectories in imagination to plan without real environment interaction.*

---

## 📂 Overview

World models are learned representations of environment dynamics. They enable sample-efficient RL by allowing agents to learn from imagined experience.

---

## 🔑 Key Components

| Component | Function |
|-----------|----------|
| **Encoder** | State → latent representation z |
| **Dynamics Model** | Predict next latent: z' = f(z, a) |
| **Reward Model** | Predict reward: r̂ = g(z, a) |
| **Decoder** | Latent → reconstructed state (optional) |

---

## 📐 Learning

```
World Model Loss:
L = reconstruction + KL + reward_prediction

1. Collect real experience (s, a, r, s')
2. Train world model on real data
3. Imagine trajectories using world model
4. Train policy on imagined data
```

---

## 🌍 Applications

| Model | Key Innovation |
|-------|----------------|
| **World Models (Ha 2018)** | VAE + RNN for racing |
| **SimPLe** | Model-based Atari |
| **Dreamer** | RSSM, backprop through model |
| **MuZero** | Value prediction, no reconstruction |

---

## 💻 Code

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim * 2)  # mean + logvar
        )
        self.dynamics = nn.GRU(latent_dim + action_dim, 256)
        self.reward_head = nn.Linear(256, 1)
        
    def imagine(self, z, actions, horizon):
        """Rollout in imagination"""
        imagined = []
        h = self.dynamics.init_hidden(z.size(0))
        for t in range(horizon):
            z_a = torch.cat([z, actions[:, t]], dim=-1)
            h = self.dynamics(z_a, h)
            z = self.transition(h)
            r = self.reward_head(h)
            imagined.append((z, r))
        return imagined
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | World Models Paper | [arXiv](https://arxiv.org/abs/1803.10122) |
| 📄 | Dreamer Paper | [arXiv](https://arxiv.org/abs/1912.01603) |
| 📄 | MuZero Paper | [Nature](https://www.nature.com/articles/s41586-020-03051-4) |
| 🇨🇳 | 世界模型详解 | [知乎](https://zhuanlan.zhihu.com/p/563656219) |
| 🇨🇳 | Dreamer系列 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/123629543) |
| 🇨🇳 | 模型基RL | [B站](https://www.bilibili.com/video/BV1C34y1H7Eq) |


## 🔗 Where This Topic Is Used

| Application | World Models |
|-------------|-------------|
| **Dreamer** | Learning in imagination |
| **MuZero** | Learned dynamics model |
| **Planning** | Model predictive control |
| **Sim-to-Real** | Domain randomization |

---

⬅️ [Back: Model-Based](../)

---

⬅️ [Back: Planning](../planning/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
