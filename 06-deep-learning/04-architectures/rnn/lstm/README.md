<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Long%20Short-Term%20Memory&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 LSTM Equations

```
Forget Gate: fₜ = σ(Wf[hₜ₋₁, xₜ] + bf)
Input Gate:  iₜ = σ(Wi[hₜ₋₁, xₜ] + bi)
Candidate:   c̃ₜ = tanh(Wc[hₜ₋₁, xₜ] + bc)
Cell State:  cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ
Output Gate: oₜ = σ(Wo[hₜ₋₁, xₜ] + bo)
Hidden:      hₜ = oₜ ⊙ tanh(cₜ)

Key insight: Cell state cₜ flows with minimal transformation
→ Gradient can flow through long sequences
```

---

## 💻 Code Example

```python
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        out, (h, c) = self.lstm(x)
        return self.fc(out)
```

---

⬅️ [Back: RNN](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
