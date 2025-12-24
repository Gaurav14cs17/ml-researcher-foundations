<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=States Actions&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# States and Actions

> **The building blocks of any MDP**

---

## 🎯 Visual Overview

<img src="./images/states-actions.svg" width="100%">

*Caption: State space S contains all possible situations the agent can observe. Action space A contains all possible decisions the agent can make. These can be discrete (finite set) or continuous (real-valued).*

---

## 📂 Overview

States represent what the agent observes about the environment. Actions represent what the agent can do to change the environment.

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **State s** | Complete description of environment at time t |
| **Action a** | Decision/control taken by agent |
| **State Space S** | Set of all possible states |
| **Action Space A** | Set of all possible actions |

---

## 📐 Types of Spaces

| Type | State Examples | Action Examples |
|------|----------------|-----------------|
| **Discrete** | Grid positions, game boards | Left/Right/Up/Down |
| **Continuous** | Robot joint angles, velocity | Force, torque values |
| **High-Dim** | Images (84×84×4) | Multi-joint control |
| **Hybrid** | Mixed discrete + continuous | Discrete choice + continuous param |

---

## 💻 Code

```python
import gymnasium as gym

# Discrete: CartPole
env = gym.make("CartPole-v1")
print(f"State: {env.observation_space}")  # Box(4,) - continuous
print(f"Actions: {env.action_space}")     # Discrete(2) - left/right

# Continuous: MuJoCo
env = gym.make("HalfCheetah-v4")
print(f"State: {env.observation_space}")  # Box(17,) - joint positions/velocities
print(f"Actions: {env.action_space}")     # Box(6,) - continuous torques
```


## 🔗 Where This Topic Is Used

| Application | How States/Actions Are Used |
|-------------|---------------------------|
| **Game Playing** | Board state → legal moves |
| **Robotics** | Joint positions → torques |
| **Trading** | Market state → buy/sell |
| **Dialogue** | Conversation history → responses |


## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📖 | Textbook | See parent folder |
| 🎥 | Video Lectures | YouTube/Coursera |
| 🇨🇳 | 中文资源 | 知乎/B站 |

---

⬅️ [Back: MDP](../)

---

⬅️ [Back: Rewards](../rewards/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
