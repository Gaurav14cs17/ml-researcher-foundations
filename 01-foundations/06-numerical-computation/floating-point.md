# Floating Point Representation

> **How computers represent real numbers**

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
