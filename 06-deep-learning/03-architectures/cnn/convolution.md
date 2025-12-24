# Convolution Operation

> **The fundamental operation of CNNs**

---

## 📐 Definition

```
(f * g)(i, j) = ΣΣ f(m, n) · g(i-m, j-n)

For images:
output[i,j] = Σ_m Σ_n input[i+m, j+n] · kernel[m, n]
```

---

## 🔑 Key Parameters

| Parameter | Effect |
|-----------|--------|
| Kernel size | Receptive field (3×3, 5×5, ...) |
| Stride | Step size (1, 2, ...) |
| Padding | Border handling (same, valid) |
| Dilation | Gaps in kernel |

---

## 📊 Output Size

```
H_out = floor((H_in + 2*padding - dilation*(kernel-1) - 1) / stride + 1)

Common cases:
• Same padding, stride=1: H_out = H_in
• No padding, stride=2: H_out ≈ H_in / 2
```

---

## 💻 Code

```python
import torch
import torch.nn as nn

# 2D convolution layer
conv = nn.Conv2d(
    in_channels=3,      # e.g., RGB
    out_channels=64,    # number of filters
    kernel_size=3,      # 3×3 kernel
    stride=1,
    padding=1           # "same" padding
)

x = torch.randn(1, 3, 224, 224)  # (batch, channels, H, W)
y = conv(x)
print(y.shape)  # (1, 64, 224, 224)

# Parameters
print(conv.weight.shape)  # (64, 3, 3, 3)
print(conv.bias.shape)    # (64,)
```

---

## 🌍 Properties

| Property | Benefit |
|----------|---------|
| Translation equivariance | Detect patterns anywhere |
| Parameter sharing | Efficient (vs fully connected) |
| Local connectivity | Capture local patterns |

---

<- [Back](./README.md)


