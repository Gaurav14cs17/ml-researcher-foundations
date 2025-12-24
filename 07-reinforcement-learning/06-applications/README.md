# 🎯 RL Applications

> **Real-world uses of reinforcement learning**

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
| [games/](./games/) | Game playing | AlphaGo, Atari |
| [robotics/](./robotics/) | Robot control | Manipulation |
| [rlhf/](./rlhf/) | 🔥 LLM alignment | ChatGPT |

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

⬅️ [Back: 05-Model-Based](../05-model-based/)


