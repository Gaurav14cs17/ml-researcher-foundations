<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Recurrent%20Neural%20Networks&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

⬅️ [Back: ResNet](../08_resnet/README.md) | ➡️ [Next: Seq2Seq](../10_seq2seq/README.md)

---

⬅️ [Back: Architectures](../../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
