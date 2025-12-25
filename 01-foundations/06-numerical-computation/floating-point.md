<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=120&section=header&text=Floating%20Point%20Representation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-01-6C63FF?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 IEEE 754 Format

```
(-1)^s × 1.mantissa × 2^(exponent - bias)

Float32 (32 bits):
+---+----------+-----------------------+
| s | exponent |       mantissa        |
| 1 |    8     |          23           |
+---+----------+-----------------------+
bias = 127

Float16 (16 bits):
+---+-----+------------+
| s | exp |  mantissa  |
| 1 |  5  |     10     |
+---+-----+------------+
bias = 15
```

---

## 📊 Comparison

| Format | Bits | Range | Precision | ML Use |
|--------|------|-------|-----------|--------|
| FP32 | 32 | ±3.4×10³⁸ | ~7 digits | Default |
| FP16 | 16 | ±65,504 | ~3 digits | Mixed precision |
| BF16 | 16 | ±3.4×10³⁸ | ~2 digits | TPU, A100 |

---

## 🔑 Key Numbers

```python
import numpy as np

# Machine epsilon (smallest x where 1+x ≠ 1)
np.finfo(np.float32).eps  # 1.19e-07
np.finfo(np.float16).eps  # 9.77e-04

# Max values
np.finfo(np.float32).max  # 3.40e+38
np.finfo(np.float16).max  # 65504.0
```

---

## ⚠️ Gotchas

```python
# Precision loss
0.1 + 0.2 == 0.3  # False!
0.1 + 0.2  # 0.30000000000000004

# Comparison with tolerance
np.isclose(0.1 + 0.2, 0.3)  # True
```

---

---

➡️ [Next: Stability](./stability.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=80&section=footer" width="100%"/>
</p>
