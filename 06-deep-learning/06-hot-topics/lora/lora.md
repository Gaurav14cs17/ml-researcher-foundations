<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=LoRA%20Low-Rank%20Adaptation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Key Insight

```
Full fine-tuning: W → W + ΔW

Observation: ΔW has low intrinsic rank!

LoRA: W + ΔW = W + BA
Where: B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵏ, r << min(d,k)
```

---

## 📊 Parameter Comparison

```
Original weight W: d × k parameters
Full fine-tuning: d × k additional parameters

LoRA: r(d + k) parameters
Example: d=4096, k=4096, r=8
• Full: 16M params
• LoRA: 65K params (250x reduction!)
```

---

## 🔑 Connection to SVD

```
Any matrix: W = UΣVᵀ

Low-rank approximation: W ≈ U_r Σ_r V_r^T
                       = (U_r Σ_r) V_r^T
                       = B · A

LoRA learns the "important directions" for adaptation
```

---

## 💻 Code

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        d, k = original.out_features, original.in_features
        
        self.original = original
        self.original.weight.requires_grad = False  # Freeze
        
        self.lora_A = nn.Linear(k, rank, bias=False)
        self.lora_B = nn.Linear(rank, d, bias=False)
        self.scale = alpha / rank
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scale
        return original_out + lora_out

# Usage with PEFT library
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(base_model, config)
```

---

## 🌍 Variants

| Method | Change |
|--------|--------|
| QLoRA | 4-bit quantized base + LoRA |
| AdaLoRA | Adaptive rank |
| DoRA | Decomposed magnitude + direction |

---

<- [Back](./README.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
