<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=01 Quantization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🔢 Quantization

> **Reducing precision for efficient inference**

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

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

