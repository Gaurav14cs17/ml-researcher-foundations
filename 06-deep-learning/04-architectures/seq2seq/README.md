<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Seq2Seq&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🔄 Sequence-to-Sequence

> **Mapping sequences to sequences**

---

## 📐 Architecture

```
Encoder-Decoder:
  Encoder: Input sequence → Context vector
  Decoder: Context vector → Output sequence

With Attention:
  Each decoder step attends to all encoder outputs
  context_t = Σ α_t,s × h_s
```

---

## 💻 Code Example

```python
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, tgt):
        encoder_out = self.encoder(src)
        output = self.decoder(tgt, encoder_out)
        return output
```

---

## 🔗 Applications

| Application | Example |
|-------------|---------|
| **Translation** | English → French |
| **Summarization** | Document → Summary |
| **Q&A** | Question → Answer |

---

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

