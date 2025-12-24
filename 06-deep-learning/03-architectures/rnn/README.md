# Recurrent Neural Networks (RNN)

> **Neural networks with memory for sequential data**

---

## 🎯 Visual Overview

<img src="./images/rnn-architecture.svg" width="100%">

*Caption: RNNs process sequences by maintaining a hidden state hₜ that is updated at each time step. The same weights are shared across all time steps. LSTM and GRU add gating mechanisms to handle long-range dependencies.*

---

## 📂 Overview

RNNs are designed for sequential data (text, time series, audio) where the order matters. They maintain a hidden state that acts as "memory" of previous inputs.

---

## 🔑 Key Variants

| Variant | Gates | Use Case |
|---------|-------|----------|
| **Vanilla RNN** | None | Simple sequences |
| **LSTM** | 3 (forget, input, output) | Long sequences, NLP |
| **GRU** | 2 (reset, update) | Faster training, similar performance |

---

## 📐 RNN Equations

```
# Vanilla RNN
hₜ = tanh(Wₓₕxₜ + Wₕₕhₜ₋₁ + b)
yₜ = Wₕᵧhₜ

# The Problem: Vanishing/Exploding gradients
∂L/∂h₀ = ∏ₜ Wₕₕᵀ · ... → 0 or ∞
```

---

## 💻 Code

```python
import torch.nn as nn

# LSTM for sequence modeling
lstm = nn.LSTM(
    input_size=128,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True
)

# Forward pass
output, (h_n, c_n) = lstm(x)  # x: (batch, seq_len, input_size)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | LSTM Paper | [Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) |
| 📄 | GRU Paper | [arXiv](https://arxiv.org/abs/1406.1078) |
| 🎥 | Colah's Blog: LSTM | [Blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) |
| 🇨🇳 | LSTM详解 | [知乎](https://zhuanlan.zhihu.com/p/32085405) |
| 🇨🇳 | RNN与LSTM原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88785898) |
| 🇨🇳 | 循环神经网络 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |


## 🔗 Where This Topic Is Used

| Application | RNN Type |
|-------------|---------|
| **Language Modeling** | LSTM (before Transformers) |
| **Speech Recognition** | CTC with LSTM |
| **Time Series** | Sequence prediction |
| **Machine Translation** | Seq2Seq (before Attention) |

---

⬅️ [Back: Architectures](../)

---

⬅️ [Back: Moe](../moe/) | ➡️ [Next: Transformer](../transformer/)
