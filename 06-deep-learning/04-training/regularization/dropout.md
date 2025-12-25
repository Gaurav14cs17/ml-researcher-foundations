<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Dropout&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
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
Training:
For each forward pass:
    mask ~ Bernoulli(1 - p)  # Keep with prob 1-p
    h = mask * h / (1 - p)    # Scale to maintain expectation

Inference:
    h = h  # No dropout, no scaling needed (inverted dropout)
```

---

## 🔑 Why It Works

```
1. Prevents co-adaptation of neurons
2. Ensemble of sub-networks (2^n models!)
3. Adds noise → regularization
4. Approximately Bayesian (Monte Carlo Dropout)
```

---

## 📊 Where to Apply

| Position | Common |
|----------|--------|
| After activation | ✓ Yes |
| Before attention | ✓ Yes |
| Embedding | ✓ Yes |
| Last layer | Sometimes |

---

## 💻 Code

```python
import torch.nn as nn
import torch.nn.functional as F

# Module
dropout = nn.Dropout(p=0.1)

# In model
class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout_p=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # After activation
        x = self.fc2(x)
        return x

# IMPORTANT: model.train() enables dropout
# model.eval() disables dropout
model.train()  # Dropout ON
model.eval()   # Dropout OFF
```

---

## 🌍 Variants

| Variant | Change | Use |
|---------|--------|-----|
| Standard | Drop neurons | Dense layers |
| Spatial | Drop channels | CNNs |
| DropConnect | Drop weights | Alternative |
| DropPath | Drop layers | ResNets |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
