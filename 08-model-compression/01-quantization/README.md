<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Quantization&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Types

```
Post-Training Quantization (PTQ):
  Quantize pre-trained model

Quantization-Aware Training (QAT):
  Train with simulated quantization

Uniform Quantization:
  q = round(x / scale) + zero_point
  x ≈ scale × (q - zero_point)
```

---

## 💻 Code

```python
import torch.quantization as quant

# Post-training quantization
model.eval()
model_quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
```

---

## 🔗 Precision Levels

| Bits | Memory | Speed | Accuracy |
|------|--------|-------|----------|
| **FP32** | 1× | 1× | Baseline |
| **FP16** | 0.5× | 2× | ~Same |
| **INT8** | 0.25× | 4× | Slight drop |
| **INT4** | 0.125× | 8× | Noticeable |

---

⬅️ [Back: Model Compression](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
