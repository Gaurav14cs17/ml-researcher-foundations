<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=120&section=header&text=Low-Rank%20Adaptation%20LoRA&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-08-E74C3C?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 How It Works

```
Original: W (frozen)
LoRA: W + ΔW = W + BA

Where:
- W: d × k (original weight, frozen)
- B: d × r (trainable, r << d)
- A: r × k (trainable)

Parameters: r(d+k) instead of d×k
```

---

## 💻 Code

```python
class LoRALayer(nn.Module):
    def __init__(self, original, r=16, alpha=32):
        super().__init__()
        self.original = original
        self.original.weight.requires_grad = False
        
        d, k = original.weight.shape
        self.A = nn.Parameter(torch.randn(r, k) / r)
        self.B = nn.Parameter(torch.zeros(d, r))
        self.scale = alpha / r
    
    def forward(self, x):
        return self.original(x) + (x @ self.A.T @ self.B.T) * self.scale
```

---

## 🔗 Benefits

- 10-100× fewer trainable params
- No inference latency (merge weights)
- Switch adapters easily

---

⬅️ [Back: Parameter Efficient](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=E74C3C&height=80&section=footer" width="100%"/>
</p>
