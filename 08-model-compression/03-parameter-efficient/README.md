<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Parameter-Efficient%20Fine-Tunin&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📂 Methods

| Method | Trainable Params | Approach |
|--------|------------------|----------|
| **LoRA** | ~0.1% | Low-rank adapters |
| **Prefix Tuning** | ~0.1% | Learnable prefixes |
| **Adapters** | ~2% | Small modules |
| **QLoRA** | ~0.1% | Quantized + LoRA |

---

## 💻 LoRA Example

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,           # Rank
    lora_alpha=32,  # Scaling
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(base_model, config)
```

---

⬅️ [Back: Model Compression](../) | ➡️ [LoRA Details](./lora/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
