# Games

> **RL for Game Playing: From Atari to AlphaGo**

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

## 📐 Key Techniques

### Self-Play
```
Agent plays against itself/past versions
Curriculum: opponents grow with agent
No human data needed (AlphaZero)
```

### MCTS + Neural Networks
```
AlphaGo architecture:
1. Policy network p(a|s): action probabilities
2. Value network v(s): win probability
3. MCTS guided by networks

Selection: UCB with policy prior
a = argmax[Q(s,a) + c·p(a|s)·√N(s)/(1+N(s,a))]
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

<- [Back](../)

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

---

⬅️ [Back: games](../)

---

➡️ [Next: Rlhf](../rlhf/)
