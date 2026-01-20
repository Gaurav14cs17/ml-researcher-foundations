<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Vision%20Transformer%20(ViT)&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2025-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 📐 Mathematical Foundations

### Core Idea

Treat image as sequence of patches, apply standard Transformer.

**Key Equation:**
```math
\text{Image} \in \mathbb{R}^{H \times W \times C} \rightarrow \text{Patches} \in \mathbb{R}^{N \times (P^2 \cdot C)}
```

Where $N = \frac{H \cdot W}{P^2}$ is the number of patches.

---

## 📐 Architecture

### 1. Patch Embedding

Split image into $P \times P$ patches:

```math
x_p^{(i)} = \text{Flatten}(\text{Patch}_i) \in \mathbb{R}^{P^2 \cdot C}
```

Project to $D$ dimensions:

```math
z_0^{(i)} = x_p^{(i)} E + e_{pos}^{(i)}, \quad E \in \mathbb{R}^{(P^2 \cdot C) \times D}
```

### 2. Prepend [CLS] Token

```math
z_0 = [x_{class}; z_0^{(1)}; z_0^{(2)}; ...; z_0^{(N)}] + E_{pos}
```

Where:
- $x\_{class} \in \mathbb{R}^D$ is learnable
- $E\_{pos} \in \mathbb{R}^{(N+1) \times D}$ is positional embedding

### 3. Transformer Encoder

```math
z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}
z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l
```

**Multi-Head Self-Attention (MSA):**
```math
\text{MSA}(z) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
```

### 4. Classification Head

Use [CLS] token for classification:

```math
y = \text{MLP}(z_L^{(0)})
```

---

## 📐 Positional Embeddings

### Learnable 1D Positional Embeddings

```math
E_{pos} \in \mathbb{R}^{(N+1) \times D}
```

Simply added to patch embeddings.

### 2D Positional Embeddings (Variant)

Separate embeddings for row and column positions:

```math
e_{pos}(i, j) = e_{row}(i) + e_{col}(j)
```

### Relative Position (Swin)

Position relative to other tokens in attention.

---

## 📐 Complexity Analysis

### Self-Attention Complexity

```math
O(N^2 \cdot D) = O\left(\left(\frac{HW}{P^2}\right)^2 \cdot D\right)
```

**Comparison:**
- CNN: $O(H \cdot W \cdot k^2 \cdot C^2)$
- ViT: $O\left(\frac{H^2 W^2}{P^4} \cdot D\right)$

Smaller patch size → more patches → higher cost.

### Memory

Attention matrix: $O(N^2) = O\left(\frac{H^2 W^2}{P^4}\right)$

For $224 \times 224$ image with $16 \times 16$ patches:
- $N = 196$ patches
- Attention matrix: $196 \times 196 \approx 38K$ elements

---

## 📊 Model Variants

| Model | Layers | Hidden | Heads | Params | Patch |
|-------|--------|--------|-------|--------|-------|
| **ViT-B/16** | 12 | 768 | 12 | 86M | 16×16 |
| **ViT-B/32** | 12 | 768 | 12 | 86M | 32×32 |
| **ViT-L/16** | 24 | 1024 | 16 | 307M | 16×16 |
| **ViT-H/14** | 32 | 1280 | 16 | 632M | 14×14 |

Notation: ViT-{Size}/{Patch_Size}

---

## 💻 Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use Conv2d for efficient patching + embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # (B, C, H, W) -> (B, embed_dim, n_patches_h, n_patches_w)
        x = self.proj(x)
        # (B, embed_dim, n_h, n_w) -> (B, n_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout)
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_channels=3,
        num_classes=1000, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification
        x = self.norm(x)
        cls_output = x[:, 0]  # Use [CLS] token
        return self.head(cls_output)

# Model configurations
def vit_base_patch16_224(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12
    )

def vit_large_patch16_224(num_classes=1000):
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=1024, depth=24, num_heads=16
    )

# Using timm library
import timm

model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=100)
```

---

## 📐 Key Insights

### Inductive Bias Comparison

| Model | Inductive Bias |
|-------|----------------|
| **CNN** | Local connectivity, translation equivariance |
| **ViT** | Minimal (positional embeddings only) |

ViT requires more data to learn spatial structure!

### Scaling Properties

- ViT benefits from larger datasets more than CNNs
- At small scale: CNN > ViT
- At large scale: ViT ≥ CNN

### Pretrain + Fine-tune

Pretrain on JFT-300M or ImageNet-21k, fine-tune on target.

---

## 📊 Variants and Extensions

| Model | Key Innovation |
|-------|----------------|
| **DeiT** | Knowledge distillation, data-efficient training |
| **Swin Transformer** | Hierarchical + shifted windows |
| **BEiT** | BERT-style pretraining (masked patches) |
| **MAE** | Masked autoencoder (75% masking) |
| **DINO** | Self-supervised with self-distillation |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | ViT Paper | [arXiv](https://arxiv.org/abs/2010.11929) |
| 📄 | DeiT Paper | [arXiv](https://arxiv.org/abs/2012.12877) |
| 📄 | Swin Transformer | [arXiv](https://arxiv.org/abs/2103.14030) |
| 📄 | MAE Paper | [arXiv](https://arxiv.org/abs/2111.06377) |
| 💻 | timm library | [GitHub](https://github.com/huggingface/pytorch-image-models) |
| 🇨🇳 | ViT详解 | [知乎](https://zhuanlan.zhihu.com/p/340149804) |

---

⬅️ [Back: Transformer](../11_transformer/README.md)

---

⬅️ [Back: Architectures](../../README.md)

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
