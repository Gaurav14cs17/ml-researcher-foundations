<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Robotics&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: RLHF](../02_rlhf/) | ➡️ [Back: Applications](../)

---

## 🎯 Visual Overview

<img src="./images/robotics.svg" width="100%">

*Caption: RL for robotics faces unique challenges: sample efficiency (real robots are slow), safety, and sim-to-real gap. Solutions include simulation training, domain randomization, and model-based RL.*

---

## 📂 Overview

Robotics is a compelling application of RL where agents must learn to control physical systems in the real world. The challenges of sample efficiency and safety have driven innovations in sim-to-real transfer and model-based methods.

---

## 🤖 Robotics RL Challenges

| Challenge | Problem | Solution |
|-----------|---------|----------|
| **Sample Efficiency** | Real robots are slow | Sim-to-real, model-based |
| **Safety** | Damage from exploration | Constrained RL, safe policies |
| **Sim-to-Real Gap** | Sim ≠ real physics | Domain randomization |
| **Partial Observability** | Noisy, limited sensors | Recurrent policies, state estimation |
| **Continuous Actions** | Joint angles, forces | PPO, SAC, DDPG |

---

## 📐 Mathematical Foundations

### Continuous Control Formulation

```
Robot state: s ∈ ℝⁿ (joint angles, velocities, etc.)
Action: a ∈ ℝᵐ (torques, target positions, etc.)
Dynamics: s_{t+1} = f(s_t, a_t) + ε  (deterministic + noise)

Objective: max_π E[Σ_t γ^t R(s_t, a_t)]

Common rewards:
  R_task = -||s - s_goal||²     (reach target)
  R_energy = -||a||²            (minimize effort)
  R_safety = -1{s ∈ S_unsafe}   (safety constraint)

```

### Sim-to-Real Transfer

```
Domain Randomization formulation:

Source domain (sim): p_sim(s'|s,a,ξ) where ξ ~ P(ξ)
  ξ = randomization parameters (mass, friction, etc.)

Target domain (real): p_real(s'|s,a)

Goal: Find π* such that:
  π* = argmax_π E_{ξ~P(ξ)} E_τ~π,p_sim(·|ξ) [R(τ)]
  
If P(ξ) is broad enough, π* generalizes to real:
  E_τ~π*,p_real [R(τ)] ≈ E_τ~π*,p_sim [R(τ)]

```

### System Identification vs Domain Randomization

```
System Identification:
  1. Collect real data: D_real = {(s, a, s')}
  2. Fit sim params: ξ* = argmin_ξ ||f_sim(s,a;ξ) - s'||
  3. Train in calibrated sim: π* from p_sim(·|ξ*)
  
  Pro: Accurate sim
  Con: Requires real data, may overfit

Domain Randomization:
  1. Define distribution P(ξ) over params
  2. Train on diverse sims: π* from E_ξ[p_sim(·|ξ)]
  
  Pro: No real data needed
  Con: May be overly conservative

```

### Safe Reinforcement Learning

```
Constrained MDP formulation:

max_π E[Σ_t γ^t R(s_t, a_t)]
s.t. E[Σ_t γ^t C(s_t, a_t)] ≤ d

Where:
  C(s,a) = cost function (e.g., collision indicator)
  d = maximum allowed cumulative cost

Solution approaches:
  1. Lagrangian relaxation: L = R - λ(C - d)
  2. Constrained Policy Optimization (CPO)
  3. Safety layers: project actions to safe set

```

### Model-Based RL for Sample Efficiency

```
MBPO (Model-Based Policy Optimization):

1. Collect real data: D_real ← {(s,a,r,s')}
2. Train dynamics ensemble: {f_θ₁, ..., f_θ_K}
3. Generate synthetic data:
   For k = 1 to K_rollouts:
     Sample model f ~ {f_θ₁, ..., f_θ_K}
     Rollout H steps: D_model ← trajectory
4. Train policy on D_real ∪ D_model using SAC

Rollout horizon H matters:
  Small H: Less model error, more real data needed
  Large H: More synthetic data, but errors compound
  
  Optimal H ≈ log(1/ε) / log(1/γ) where ε = model error

```

---

## 💻 Code Examples

```python
import gymnasium as gym
import torch

# MuJoCo continuous control
env = gym.make('Ant-v4')
obs, info = env.reset()

# SAC agent for continuous actions
class SACAgent:
    def __init__(self, obs_dim, action_dim):
        self.actor = GaussianPolicy(obs_dim, action_dim)
        self.critic = QNetwork(obs_dim, action_dim)
    
    def act(self, obs, deterministic=False):
        with torch.no_grad():
            action, _ = self.actor.sample(obs, deterministic)
        return action.cpu().numpy()

# Domain randomization
class RandomizedEnv(gym.Wrapper):
    def reset(self):
        # Randomize physics
        self.env.model.body_mass *= np.random.uniform(0.8, 1.2)
        self.env.model.dof_damping *= np.random.uniform(0.5, 2.0)
        return self.env.reset()

# Sim-to-real training
def train_sim2real(agent, sim_env, real_env):
    # Train in randomized sim
    for _ in range(1000000):
        sim_step(agent, RandomizedEnv(sim_env))
    
    # Fine-tune on real (few samples)
    for _ in range(100):
        real_step(agent, real_env)

```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Sim-to-Real Survey | [arXiv](https://arxiv.org/abs/2009.13303) |
| 📄 | OpenAI Rubik's Cube | [arXiv](https://arxiv.org/abs/1910.07113) |
| 📄 | SAC Paper | [arXiv](https://arxiv.org/abs/1801.01290) |
| 🇨🇳 | 机器人强化学习 | [知乎](https://zhuanlan.zhihu.com/p/563656219) |
| 🇨🇳 | Sim-to-Real详解 | [CSDN](https://blog.csdn.net/qq_37006625/article/details/123629543) |
| 🇨🇳 | 机器人控制 | [B站](https://www.bilibili.com/video/BV1yp4y1s7Qw) |

---

## 🔗 Where This Topic Is Used

| Application | RL in Robotics |
|-------------|---------------|
| **Manipulation** | Dexterous hands |
| **Locomotion** | Quadruped walking |
| **Navigation** | Autonomous driving |
| **Sim-to-Real** | Domain adaptation |

---

⬅️ [Back: RLHF](../02_rlhf/) | ➡️ [Back: Applications](../)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
