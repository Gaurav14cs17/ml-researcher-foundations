<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=RNN&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🔄 Recurrent Neural Networks

> **Sequential data processing with memory**

---

## 📐 RNN Equations

```
Vanilla RNN:
  hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)
  yₜ = Wₕᵧhₜ

LSTM:
  fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)     (forget gate)
  iₜ = σ(Wi[hₜ₋₁, xₜ] + bi)     (input gate)
  c̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc)  (candidate)
  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ      (cell state)
  oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)     (output gate)
  hₜ = oₜ ⊙ tanh(cₜ)            (hidden state)
```

---

## 💻 Code Example

```python
import torch.nn as nn

# LSTM
lstm = nn.LSTM(input_size=128, hidden_size=256, 
               num_layers=2, batch_first=True)

# GRU (simpler alternative)
gru = nn.GRU(input_size=128, hidden_size=256,
             num_layers=2, batch_first=True)

# Bidirectional
bilstm = nn.LSTM(128, 256, bidirectional=True)
```

---

## 🔗 Comparison

| Model | Gates | Parameters | Use |
|-------|-------|------------|-----|
| **RNN** | 0 | Fewest | Simple sequences |
| **LSTM** | 3 | Most | Long sequences |
| **GRU** | 2 | Medium | Efficient LSTM |

---

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

