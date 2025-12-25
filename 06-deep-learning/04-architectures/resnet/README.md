<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=ResNet&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# 🏗️ Residual Networks

> **Skip connections for very deep networks**

---

## 📐 Residual Block

```
Standard: y = F(x)
Residual: y = F(x) + x  (skip connection)

Why it works:
• Easy to learn identity: F(x) = 0
• Gradient flows directly through skip
• Enables 100+ layer networks
```

---

## 💻 Code Example

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return nn.ReLU()(out)
```

---

## 🔗 Variants

| Model | Depth | Parameters |
|-------|-------|------------|
| **ResNet-18** | 18 | 11M |
| **ResNet-50** | 50 | 25M |
| **ResNet-152** | 152 | 60M |

---

⬅️ [Back: Architectures](../)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>

