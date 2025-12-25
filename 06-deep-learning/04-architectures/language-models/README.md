<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Language Models&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 📝 Language Models

> **Modeling probability of text sequences**

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

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

