<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Deep%20Q-Networks%20DQN&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## 🔗 Navigation

⬅️ [Back: Bellman](../01_bellman/) | ➡️ [Next: Dynamic Programming](../03_dynamic_programming/)

---

## 🎯 Visual Overview

<img src="./images/dqn-architecture.svg" width="100%">

*Caption: DQN approximates Q-values with a neural network. Two key innovations enable stable training: Experience Replay (breaks correlation) and Target Network (stable bootstrap targets). This enabled RL to master Atari games from pixels.*

---

## 📂 Overview

DQN extends tabular Q-learning to continuous state spaces using neural networks. Two key innovations solved the instability problem: experience replay and target networks.

---

## 📐 Mathematical Foundation

### Q-Learning with Function Approximation

```
Tabular Q-Learning:
    Q(s, a) ← Q(s, a) + α [r + γ max_a' Q(s', a') - Q(s, a)]

DQN (neural network):
    Q(s, a; θ) ≈ Q*(s, a)
    
    Loss = E[(r + γ max_a' Q(s', a'; θ⁻) - Q(s, a; θ))²]
                                    +- Target network

```

### Why Instability?

```
Problems with naive Q-learning + neural nets:

1. Correlated samples:
   Sequential (s, a, r, s') are highly correlated
   → Gradient updates biased
   
2. Non-stationary targets:
   Target y = r + γ max Q(s'; θ) changes as θ updates
   → "Chasing a moving target"
   
Solutions:
1. Experience Replay → Break correlation
2. Target Network → Stabilize targets

```

---

## 🎯 Key Innovations

| Innovation | Purpose | How It Works |
|------------|---------|--------------|
| **Experience Replay** | Break correlation, reuse data | Store (s,a,r,s') in buffer, sample randomly |
| **Target Network** | Stabilize training | Separate θ⁻ for target, update periodically |
| **Huber Loss** | Robust to outliers | Quadratic for small errors, linear for large |

### Experience Replay

```
Replay Buffer D of capacity N:

During experience:
    Store transition (s, a, r, s', done) in D
    If |D| > N: Remove oldest

During training:
    Sample mini-batch uniformly from D
    Update Q on mini-batch
    
Benefits:

- Breaks temporal correlation

- Each experience used many times (data efficiency)

- More stable gradients

```

### Target Network

```
Two networks:
    Q(s, a; θ)    - Online network (updated every step)
    Q(s, a; θ⁻)   - Target network (updated every C steps)

Target: y = r + γ max_a' Q(s', a'; θ⁻)

Why stable:
    θ⁻ doesn't change during mini-batch
    Targets are stable for C steps
    No "chasing moving target"

```

---

## 📐 Algorithm

```
Initialize online network Q(θ), target network Q(θ⁻ = θ)
Initialize replay buffer D of capacity N

For each episode:
    s ← initial state
    For each step:
        a ← ε-greedy(Q(s; θ))
        Execute a, observe r, s'
        Store (s, a, r, s', done) in D
        s ← s'
        
        Sample mini-batch {(sⱼ, aⱼ, rⱼ, s'ⱼ, dⱼ)} from D
        
        For each sample j:
            yⱼ = rⱼ + γ max_a' Q(s'ⱼ, a'; θ⁻) × (1 - dⱼ)
        
        Update θ by minimizing:
            L = (1/B) Σⱼ (yⱼ - Q(sⱼ, aⱼ; θ))²
        
        Every C steps: θ⁻ ← θ  (or soft update)

```

---

## 💻 Code Implementation

### Complete DQN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class DQNetwork(nn.Module):
    """Q-Network for Atari (image input)"""
    
    def __init__(self, action_dim):
        super().__init__()
        # Convolutional layers (for image input)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Fully connected
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, action_dim)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Complete DQN Agent"""
    
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=32, target_update=1000):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.q_network = DQNetwork(action_dim)
        self.target_network = DQNetwork(action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0
    
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax(1).item()
    
    def update(self):
        """One gradient step"""
        if len(self.buffer) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # Current Q values
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values (use target network)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Huber loss (more robust than MSE)
        loss = F.smooth_l1_loss(q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

```

### Double DQN

```python
def double_dqn_update(q_network, target_network, batch, gamma):
    """
    Double DQN: Use online network to SELECT action,
                target network to EVALUATE value
    
    Reduces overestimation bias
    """
    states, actions, rewards, next_states, dones = batch
    
    # Current Q values
    q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    
    with torch.no_grad():
        # Online network selects best action
        next_actions = q_network(next_states).argmax(1, keepdim=True)
        
        # Target network evaluates the action
        next_q = target_network(next_states).gather(1, next_actions).squeeze(1)
        
        target_q = rewards + gamma * next_q * (1 - dones)
    
    return F.smooth_l1_loss(q_values, target_q)

```

### Dueling DQN

```python
class DuelingDQN(nn.Module):
    """
    Dueling DQN: Separate value and advantage streams
    
    Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Shared layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q

```

---

## 📊 DQN Variants

| Variant | Key Improvement | Problem Solved |
|---------|-----------------|----------------|
| **Double DQN** | Separate action selection/evaluation | Overestimation bias |
| **Dueling DQN** | V(s) + A(s,a) decomposition | Better value estimation |
| **Prioritized ER** | Sample important transitions more | Sample efficiency |
| **Noisy Nets** | Learned exploration | Fixed ε-greedy limitation |
| **Distributional** | Model return distribution | Risk awareness |
| **Rainbow** | All of the above | Best performance |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | DQN (Original) | [arXiv](https://arxiv.org/abs/1312.5602) |
| 📄 | Double DQN | [arXiv](https://arxiv.org/abs/1509.06461) |
| 📄 | Dueling DQN | [arXiv](https://arxiv.org/abs/1511.06581) |
| 📄 | Prioritized ER | [arXiv](https://arxiv.org/abs/1511.05952) |
| 📄 | Rainbow | [arXiv](https://arxiv.org/abs/1710.02298) |
| 📖 | Sutton & Barto Ch. 9 | [RL Book](http://incompleteideas.net/book/) |
| 🇨🇳 | DQN系列算法详解 | [知乎](https://zhuanlan.zhihu.com/p/26052182) |
| 🇨🇳 | DQN完整实现 | [CSDN](https://blog.csdn.net/qq_30615903/article/details/80952771) |
| 🇨🇳 | 深度强化学习-DQN | [B站](https://www.bilibili.com/video/BV1yp4y1s7Qw) |
| 🇨🇳 | DQN论文解读 | [机器之心](https://www.jiqizhixin.com/articles/2018-04-17-3) |
| 🇨🇳 | Rainbow DQN | [PaperWeekly](https://www.paperweekly.site/papers/notes/1196)

## 🔗 Where This Topic Is Used

| Application | DQN Variant |
|-------------|------------|
| **Atari Games** | Original DQN paper |
| **Robotics** | Continuous control adaptation |
| **Recommendation** | Deep Q-network for ranking |
| **Game AI** | Many improvements (Rainbow) |

---

⬅️ [Back: Bellman](../01_bellman/) | ➡️ [Next: Dynamic Programming](../03_dynamic_programming/)

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
