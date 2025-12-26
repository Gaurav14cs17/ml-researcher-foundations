<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Games&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Applications](../) | ➡️ [Next: RLHF](../02_rlhf/)

---

## 🎯 Visual Overview

<img src="./images/game-playing.svg" width="100%">

*Caption: RL has achieved superhuman performance in games: DQN on Atari (2013), AlphaGo on Go (2016), AlphaStar on StarCraft II (2019), and beyond. Key techniques include self-play, MCTS, and population-based training.*

---

## 📂 Overview

Games are the classic testbed for RL algorithms. The combination of clear rewards, structured environments, and ability to simulate billions of games enables development of superhuman agents.

---

## 🏆 Landmark Achievements

| System | Year | Achievement | Method |
|--------|------|-------------|--------|
| **DQN** | 2013 | Human-level Atari | Deep Q-learning |
| **AlphaGo** | 2016 | Beat Go world champion | MCTS + Policy/Value nets |
| **AlphaZero** | 2017 | Master Chess/Shogi/Go | Self-play only |
| **OpenAI Five** | 2019 | Beat Dota 2 pros | PPO + self-play |
| **AlphaStar** | 2019 | Grandmaster StarCraft | Population-based + imitation |
| **MuZero** | 2020 | Master without rules | Learned world model |

---

## 📐 Mathematical Foundations

### Self-Play Training

```
Self-play creates a curriculum of increasingly strong opponents:

At iteration i:
  1. Generate games: π_i plays against π_i (or π_{i-k})
  2. Collect trajectories: τ = {(s_t, π_i(s_t), r_t)}
  3. Update policy: π_{i+1} = improve(π_i, τ)

Theoretical guarantee (fictitious play):
  In two-player zero-sum games, self-play converges to Nash equilibrium
  as iterations → ∞ (under certain conditions).
```

### MCTS with Neural Network Guidance (AlphaGo/AlphaZero)

```
PUCT (Polynomial Upper Confidence Trees) formula:

a = argmax_a [Q(s,a) + c · P(a|s) · √(N(s)) / (1 + N(s,a))]
              --------   ------------------------------------
              exploit          explore (prior-guided)

Where:
  Q(s,a) = average value of action a from simulations
  P(a|s) = policy network prior probability  
  N(s) = visit count of state s
  N(s,a) = visit count of state-action pair
  c = exploration constant (typically 1.0-2.0)

Training targets:
  Policy: π_θ(a|s) → MCTS policy π_MCTS(a|s)
  Value: v_θ(s) → game outcome z ∈ {-1, +1}

Loss: L = (z - v_θ(s))² - π_MCTS^T log π_θ + c||θ||²
```

### AlphaZero Training Pipeline

```
1. Self-Play Data Generation:
   For each game:
     - Start from initial position s_0
     - At each turn, run MCTS (800 simulations)
     - Sample action: a_t ~ π_MCTS(·|s_t)^{1/τ}
     - Record (s_t, π_MCTS, z) where z = game outcome

2. Neural Network Training:
   Sample batch from replay buffer
   Minimize: L = MSE(v_θ, z) + CrossEntropy(π_θ, π_MCTS)

3. Evaluation:
   New network plays against current best
   Replace if win rate > 55%
```

### Deep Q-Network (DQN) for Atari

```
DQN Loss:
  L(θ) = E[(r + γ max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ))²]

Key innovations:
  1. Experience replay: Break correlation in sequential data
  2. Target network θ⁻: Stabilize training (update every C steps)
  3. Frame stacking: 4 frames as state for temporal info
  4. Reward clipping: All rewards ∈ {-1, 0, +1}

Rainbow DQN combines 6 improvements:
  Double DQN, Prioritized Replay, Dueling, Multi-step,
  Distributional, Noisy Networks
```

---

## 💻 Code Examples

```python
import gymnasium as gym
import torch

# Atari environment
env = gym.make('ALE/Breakout-v5')
obs, info = env.reset()

# DQN-style agent
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.q_net = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def act(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            return self.q_net(state).argmax().item()

# Self-play training loop
def self_play_episode(agent):
    env = GameEnv()
    states, actions, rewards = [], [], []
    while not env.done:
        action = agent.act(env.state)
        next_state, reward = env.step(action)
        states.append(env.state)
        actions.append(action)
        rewards.append(reward)
    return states, actions, rewards
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | DQN Paper | [Nature](https://www.nature.com/articles/nature14236) |
| 📄 | AlphaGo Paper | [Nature](https://www.nature.com/articles/nature16961) |
| 📄 | AlphaZero Paper | [Science](https://www.science.org/doi/10.1126/science.aar6404) |
| 🇨🇳 | AlphaGo详解 | [知乎](https://zhuanlan.zhihu.com/p/25345778) |
| 🇨🇳 | DQN原理 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80739243) |
| 🇨🇳 | 游戏AI | [B站](https://www.bilibili.com/video/BV1C34y1H7Eq) |

---

## 🔗 Where This Topic Is Used

| Game | RL Method |
|------|----------|
| **Go** | AlphaGo, MCTS + NN |
| **Chess** | AlphaZero |
| **Atari** | DQN, Rainbow |
| **Dota 2** | OpenAI Five (PPO) |
| **StarCraft** | AlphaStar |

---

⬅️ [Back: Applications](../) | ➡️ [Next: RLHF](../02_rlhf/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
