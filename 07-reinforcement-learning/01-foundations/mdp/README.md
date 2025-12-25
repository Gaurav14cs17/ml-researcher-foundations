<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=120&section=header&text=Markov%20Decision%20Process&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-07-F39C12?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 MDP Definition

```
MDP = (S, A, P, R, γ)

S: State space
A: Action space
P: P(s'|s,a) - Transition probability
R: r(s,a,s') - Reward function
γ: Discount factor ∈ [0,1]

Markov Property:
P(sₜ₊₁|sₜ,aₜ,...,s₀,a₀) = P(sₜ₊₁|sₜ,aₜ)
```

---

## 📐 Value Functions

```
State Value:
V^π(s) = E[Σₜ γᵗrₜ | s₀=s, π]

Action Value:
Q^π(s,a) = E[Σₜ γᵗrₜ | s₀=s, a₀=a, π]

Bellman Equation:
V^π(s) = Σₐ π(a|s) Σₛ' P(s'|s,a)[R + γV^π(s')]
```

---

## 💻 Code Example

```python
# Simple MDP
class MDP:
    def __init__(self, n_states, n_actions):
        self.P = np.random.rand(n_states, n_actions, n_states)
        self.P /= self.P.sum(axis=2, keepdims=True)
        self.R = np.random.randn(n_states, n_actions, n_states)
        self.gamma = 0.99
    
    def step(self, state, action):
        next_state = np.random.choice(len(self.P), p=self.P[state, action])
        reward = self.R[state, action, next_state]
        return next_state, reward
```

---

➡️ See [01-MDP](../../01-mdp/) for complete theory

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=F39C12&height=80&section=footer" width="100%"/>
</p>
