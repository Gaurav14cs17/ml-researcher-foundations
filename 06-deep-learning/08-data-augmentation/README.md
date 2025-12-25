<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=Data%20Augmentation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/data-augmentation-complete.svg" width="100%">

*Caption: Data augmentation creates new training samples through transformations. Image augmentations include flips, rotations, crops. Mixup and CutMix blend multiple images.*

---

## 📐 Key Techniques

### Image Augmentation
```
Geometric:
• Random crop, flip, rotation
• Scale, shear, perspective

Color:
• Brightness, contrast, saturation
• Color jitter, grayscale

Advanced:
• Mixup: x̃ = λx₁ + (1-λ)x₂
• CutMix: Paste patch from one image to another
• AutoAugment: Learned augmentation policies
```

### Text Augmentation
```
• Synonym replacement
• Random insertion/deletion
• Back-translation
• EDA (Easy Data Augmentation)
```

---

## 💻 Code Examples

```python
import torch
from torchvision import transforms

# Standard augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

# Mixup
def mixup(x, y, alpha=0.2):
    lam = torch.distributions.Beta(alpha, alpha).sample()
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

# RandAugment
from torchvision.transforms import RandAugment
transform = transforms.Compose([
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor()
])
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | Mixup | [arXiv](https://arxiv.org/abs/1710.09412) |
| 📄 | AutoAugment | [arXiv](https://arxiv.org/abs/1805.09501) |
| 🇨🇳 | 数据增强详解 | [知乎](https://zhuanlan.zhihu.com/p/41679153) |

---

⬅️ [Back: 07-Transfer Learning](../07-transfer-learning/) | ➡️ [Next: 09-Self-Supervised](../09-self-supervised/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
