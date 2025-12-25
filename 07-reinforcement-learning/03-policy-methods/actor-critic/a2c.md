<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Actor-Critic%20Methods&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Architecture

```
Actor: π_θ(a|s) - policy network
Critic: V_ω(s) - value network

Actor loss: -E[log π_θ(a|s) · A(s,a)]
Critic loss: E[(V_ω(s) - G)²]

Where A(s,a) = r + γV(s') - V(s) (advantage)
```

---

## 🔑 Why Two Networks?

```
REINFORCE: High variance (uses full return)
Actor-Critic: Lower variance (uses value baseline)

A(s,a) = Q(s,a) - V(s)
       ≈ r + γV(s') - V(s)  (TD error)
```

---

## 📊 A2C vs A3C

| Method | Difference |
|--------|------------|
| A2C | Synchronous updates |
| A3C | Asynchronous parallel actors |

---

## 💻 Code

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Linear(state_dim, hidden)
        self.actor = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        h = F.relu(self.shared(x))
        logits = self.actor(h)
        value = self.critic(h)
        return logits, value

def a2c_loss(states, actions, rewards, next_states, dones, model, gamma=0.99):
    logits, values = model(states)
    _, next_values = model(next_states)
    
    # TD target and advantage
    td_targets = rewards + gamma * next_values * (1 - dones)
    advantages = td_targets.detach() - values
    
    # Actor loss (policy gradient with advantage)
    log_probs = F.log_softmax(logits, dim=-1)
    action_log_probs = log_probs.gather(1, actions.unsqueeze(1))
    actor_loss = -(action_log_probs * advantages.detach()).mean()
    
    # Critic loss (value function)
    critic_loss = F.mse_loss(values, td_targets.detach())
    
    return actor_loss + 0.5 * critic_loss
```

---

## 🌍 Extensions

| Method | Innovation |
|--------|------------|
| PPO | Clipped objective |
| TRPO | Trust region constraint |
| SAC | Entropy regularization |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
