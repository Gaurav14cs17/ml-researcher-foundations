<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=RL%20Applications&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Model-Based](../05_model_based/) | ➡️ [Start: Games](./01_games/)

---

## 🎯 Visual Overview

<img src="./images/rl-applications.svg" width="100%">

*Caption: RL powers games (AlphaGo), robotics, LLM alignment (RLHF), autonomous vehicles, recommendation systems, and scientific discovery.*

---

## 📐 Mathematical Foundations

### RLHF Objective

```
J(π) = E[r_human(x, y)] - β KL(π || π_ref)

Where:
• r_human = learned reward model
• π = current policy (LLM)
• π_ref = reference policy (pre-RLHF)
• β = KL penalty coefficient

```

### Bradley-Terry Reward Model

```
P(y₁ > y₂ | x) = σ(r(x, y₁) - r(x, y₂))

Train reward model on human preference pairs

```

### DPO (Direct Preference Optimization)

```
L_DPO = -log σ(β log(π/π_ref)(y_w) - β log(π/π_ref)(y_l))

Directly optimizes policy from preferences
No separate reward model needed!

```

---

## 📂 Topics

| Folder | Domain | Examples |
|--------|--------|----------|
| [01_games/](./01_games/) | Game playing | AlphaGo, Atari |
| [02_rlhf/](./02_rlhf/) | 🔥 LLM alignment | ChatGPT |
| [03_robotics/](./03_robotics/) | Robot control | Manipulation |

---

## 🏆 Notable Achievements

| System | Year | Achievement |
|--------|------|-------------|
| **DQN** | 2013 | Human-level Atari |
| **AlphaGo** | 2016 | Beat world Go champion |
| **OpenAI Five** | 2019 | Beat Dota 2 pros |
| **AlphaStar** | 2019 | Grandmaster StarCraft |
| **ChatGPT** | 2022 | RLHF for chat |

---

## 🔥 RLHF: The Modern Application

```
Pre-training (self-supervised)
         |
         v
Supervised Fine-tuning (demonstrations)
         |
         v
RLHF (human preferences)
         |
         v
Aligned, helpful AI assistant

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | AlphaGo Paper | [Nature](https://www.nature.com/articles/nature16961) |
| 📄 | InstructGPT Paper | [arXiv](https://arxiv.org/abs/2203.02155) |
| 📄 | OpenAI Five | [Blog](https://openai.com/five/) |
| 🇨🇳 | AlphaGo详解 | [知乎](https://zhuanlan.zhihu.com/p/25345778) |
| 🇨🇳 | RLHF原理与实践 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/129405866) |
| 🇨🇳 | 强化学习应用 | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |

## 🔗 Where This Topic Is Used

| Domain | RL Applications |
|--------|----------------|
| **LLMs** | RLHF for alignment |
| **Games** | Superhuman play |
| **Robotics** | Control policies |
| **Recommendation** | Sequential decisions |

---

⬅️ [Back: Model-Based](../05_model_based/) | ➡️ [Start: Games](./01_games/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
