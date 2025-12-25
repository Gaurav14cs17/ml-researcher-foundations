<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=LSTM%20Long%20Short-Term%20Memory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Architecture

```
Forget gate:  f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:   i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:    C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
Cell state:   C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
Output gate:  o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden:       h_t = o_t ⊙ tanh(C_t)
```

---

## 🔑 Key Components

| Gate | Function |
|------|----------|
| Forget (f) | What to discard from memory |
| Input (i) | What to add to memory |
| Output (o) | What to output |
| Cell (C) | Long-term memory |
| Hidden (h) | Short-term output |

---

## 🎯 Why It Works

```
Cell state C provides "highway" for gradients:

∂C_t/∂C_{t-1} = f_t  (can be ≈1!)

Compare to vanilla RNN:
∂h_t/∂h_{t-1} = W·diag(tanh') → vanishes or explodes
```

---

## 💻 Code

```python
import torch
import torch.nn as nn

# LSTM layer
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    dropout=0.1
)

# Input: (batch, seq_len, input_size)
x = torch.randn(32, 100, 128)
output, (h_n, c_n) = lstm(x)

print(output.shape)  # (32, 100, 256) - all hidden states
print(h_n.shape)     # (2, 32, 256) - final hidden per layer
print(c_n.shape)     # (2, 32, 256) - final cell per layer
```

---

## 📊 Variants

| Variant | Change |
|---------|--------|
| GRU | 2 gates (simpler) |
| Peephole | Gates see cell state |
| Bidirectional | Process both directions |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
