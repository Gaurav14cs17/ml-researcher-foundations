<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Policy-Based%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Value Methods](../02_value_methods/) | ➡️ [Next: Exploration](../04_exploration/)

---

## 🎯 Visual Overview

<img src="./images/ppo-clipping.svg" width="100%">

*Caption: PPO's clipped objective prevents destructive policy updates. For good actions (A>0), it limits how much the policy can increase; for bad actions (A<0), it limits how much it can decrease. This stabilization is why PPO is the default choice for RLHF in ChatGPT and Claude.*

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [01_actor_critic/](./01_actor_critic/) | A2C, A3C | Value + Policy |
| [02_policy_gradient/](./02_policy_gradient/) | REINFORCE | ∇J(θ), baseline |
| [03_ppo/](./03_ppo/) | Proximal Policy Optimization | 🔥 Modern standard |
| [04_trpo/](./04_trpo/) | Trust Region | Natural gradient |

---

## 📐 Mathematical Foundations

### Policy Parameterization

```
Discrete: π_θ(a|s) = softmax(fθ(s))_a
Continuous: π_θ(a|s) = N(μθ(s), σθ(s)²)

```

### Policy Gradient Derivation

```
J(θ) = E_π [Σₜ γᵗ rₜ]

∇J(θ) = E_π [Σₜ ∇log π_θ(aₜ|sₜ) G_t]

Where G_t = Σ_{k=t}^T γ^{k-t} r_k (return from t)

```

### Advantage Estimation (GAE)

```
Aₜ^GAE = Σ_{l=0}^∞ (γλ)^l δₜ₊ₗ

Where δₜ = rₜ + γV(sₜ₊₁) - V(sₜ)

λ = 0: A = δ (high bias, low variance)
λ = 1: A = G_t - V(s) (low bias, high variance)

```

### PPO Clipped Objective

```
L^CLIP(θ) = E [min(rₜ(θ)Aₜ, clip(rₜ(θ), 1-ε, 1+ε)Aₜ)]

Where rₜ(θ) = π_θ(aₜ|sₜ) / π_{θ_old}(aₜ|sₜ)

```

---

## 🔑 Policy Gradient Theorem

```
∇J(θ) = E_π[∇log πθ(a|s) · Qπ(s,a)]
      = E_π[∇log πθ(a|s) · Aπ(s,a)]  # With advantage

Where:
• J(θ) = Expected return
• π_θ(a|s) = Policy parameterized by θ
• A(s,a) = Q(s,a) - V(s) = Advantage

```

---

## 📊 Comparison

| Method | Variance | Bias | Stability |
|--------|----------|------|-----------|
| REINFORCE | High | None | Low |
| Actor-Critic | Medium | Some | Medium |
| PPO | Low | Some | 🔥 High |

---

## 🔗 Where This Topic Is Used

| Topic | How Policy Methods Are Used |
|-------|----------------------------|
| **RLHF** | PPO optimizes LLM policy |
| **InstructGPT** | PPO + KL penalty |
| **ChatGPT** | PPO for alignment |
| **Claude** | Constitutional AI (policy-based) |
| **LLaMA-2 Chat** | PPO for instruction following |
| **Robotics** | PPO/TRPO for continuous control |
| **Game AI** | A3C for Atari, PPO for games |
| **AlphaGo** | Policy network + MCTS |
| **Autonomous Driving** | Policy gradient for decisions |

### Prerequisite For

```
Policy Methods --> RLHF (PPO is standard)
              --> DPO (derived from policy gradient)
              --> Actor-Critic variants
              --> Multi-agent RL

```

### Methods Used In

| Method | Used By |
|--------|---------|
| **PPO** | ChatGPT, Claude, LLaMA-2, OpenAI Five |
| **TRPO** | Safety-critical robotics |
| **A3C/A2C** | DeepMind Impala, distributed RL |
| **SAC** | Robotics (continuous control) |

### PPO's Role in LLM Training

```
Pretrain (unsupervised) → SFT (supervised) → RLHF (PPO!)
                                               ↑
                                        Policy gradient
                                        optimizes for
                                        human preference

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | PPO Paper | [arXiv](https://arxiv.org/abs/1707.06347) |
| 📄 | TRPO Paper | [arXiv](https://arxiv.org/abs/1502.05477) |
| 📖 | OpenAI Spinning Up | [Docs](https://spinningup.openai.com/) |
| 🇨🇳 | 策略梯度方法详解 | [知乎](https://zhuanlan.zhihu.com/p/26174099) |
| 🇨🇳 | PPO算法详解 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/81275638) |
| 🇨🇳 | 强化学习策略方法 | [B站](https://www.bilibili.com/video/BV1sd4y167NS) |

---

⬅️ [Back: Value Methods](../02_value_methods/) | ➡️ [Next: Exploration](../04_exploration/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
