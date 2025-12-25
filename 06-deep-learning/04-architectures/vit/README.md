<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Vision%20Transformer%20ViT&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Architecture

```
1. Split image into patches (16×16)
2. Flatten and project patches
3. Add positional embeddings
4. Prepend [CLS] token
5. Apply transformer encoder
6. Use [CLS] for classification

Patch embedding:
  z_0 = [x_cls; x_1E; x_2E; ...] + E_pos
```

---

## 💻 Code Example

```python
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, dim, depth, heads, num_classes):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.transformer = nn.TransformerEncoder(...)
        self.head = nn.Linear(dim, num_classes)
```

---

## 🔗 Variants

| Model | Params | Use |
|-------|--------|-----|
| **ViT-B** | 86M | Base model |
| **ViT-L** | 307M | Large |
| **DeiT** | - | Data-efficient |
| **Swin** | - | Hierarchical |

---

⬅️ [Back: Architectures](../)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
