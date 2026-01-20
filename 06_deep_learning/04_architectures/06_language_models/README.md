<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Language%20Models&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Language Model Types

```
Autoregressive (GPT):
  P(x₁,...,xₙ) = ∏ P(xᵢ|x₁,...,xᵢ₋₁)

Masked (BERT):
  P(x_mask|x_context)

Seq2Seq (T5):
  P(output|input)

```

---

## 💻 GPT-style Generation

```python
def generate(model, prompt, max_len=100):
    tokens = tokenize(prompt)
    for _ in range(max_len):
        logits = model(tokens)
        next_token = sample(logits[-1])
        tokens.append(next_token)
        if next_token == EOS:
            break
    return tokens

```

---

## 🔗 Key Models

| Model | Type | Parameters |
|-------|------|------------|
| **GPT-4** | Decoder | ~1.8T |
| **BERT** | Encoder | 340M |
| **T5** | Enc-Dec | 11B |
| **LLaMA** | Decoder | 7B-70B |

---

⬅️ [Back: GNN](../05_gnn/README.md) | ➡️ [Next: MoE](../07_mixture_of_experts/README.md)

---

⬅️ [Back: Architectures](../../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
