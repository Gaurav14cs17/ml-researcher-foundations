<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=INT8 Quantization&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🔢 INT8 Quantization

> **8-bit integer inference**

---

## 📐 Formula

```
Symmetric:
  q = round(x / scale)
  scale = max(|x|) / 127

Asymmetric:
  q = round(x / scale) + zero_point
  scale = (max(x) - min(x)) / 255
```

---

## 💻 Code

```python
# PyTorch quantization
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)

# Calibrate with representative data
for batch in calibration_data:
    model(batch)

torch.quantization.convert(model, inplace=True)
```

---

## 🔗 Benefits

- 4× memory reduction
- 2-4× speedup on CPU
- Minimal accuracy loss (~1%)

---

⬅️ [Back: Quantization](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

