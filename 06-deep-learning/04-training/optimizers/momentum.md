<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Momentum&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Algorithm

```
Classical Momentum:
v_{t+1} = βv_t + ∇f(θ_t)
θ_{t+1} = θ_t - αv_{t+1}

Nesterov Momentum:
v_{t+1} = βv_t + ∇f(θ_t - αβv_t)  # Look ahead
θ_{t+1} = θ_t - αv_{t+1}
```

---

## 🔑 Intuition

```
Ball rolling down hill:
• Accumulates velocity in consistent directions
• Dampens oscillations in inconsistent directions
• Escapes shallow local minima

β = 0.9 typical (heavy ball)
```

---

## 📊 Why It Helps

| Problem | How Momentum Helps |
|---------|-------------------|
| Ravines | Builds speed along valley |
| Oscillation | Cancels perpendicular components |
| Saddle points | Carries through flat regions |

---

## 💻 Code

```python
import torch

# SGD with momentum
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Nesterov variant
)

# Manual implementation
class MomentumOptimizer:
    def __init__(self, params, lr, momentum):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.v = [torch.zeros_like(p) for p in self.params]
    
    def step(self):
        for i, p in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] + p.grad
            p.data -= self.lr * self.v[i]
```

---

## 📊 Convergence

| Method | Convex Rate | Strongly Convex |
|--------|-------------|-----------------|
| GD | O(1/k) | O((1-μ/L)^k) |
| Momentum | O(1/k²) | O((1-√(μ/L))^k) |

Momentum achieves optimal rate!

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
