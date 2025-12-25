<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=120&section=header&text=CNN%20Convolutional%20Neural%20Netwo&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-06-45B7D1?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

---

## 🎯 Visual Overview

<img src="./images/cnn-architecture.svg" width="100%">

*Caption: Complete CNN architecture showing input image flowing through convolutional layers (feature extraction), pooling layers (downsampling), and fully connected layers (classification). The diagram illustrates how spatial features are progressively extracted and transformed into class predictions. This architecture is the foundation for computer vision tasks.*

---

## 📐 Key Operations

| Operation | Purpose |
|-----------|---------|
| **Convolution** | Local feature extraction |
| **Pooling** | Downsampling |
| **Stride** | Skip positions |
| **Padding** | Preserve spatial size |

---

## 🔑 Convolution

```
Output[i,j] = Σₘ Σₙ Input[i+m, j+n] × Kernel[m, n]

Properties:
• Translation equivariance
• Parameter sharing
• Local connectivity
```

---

## 🏗️ Famous Architectures

| Architecture | Year | Innovation |
|--------------|------|------------|
| **LeNet** | 1998 | First CNN |
| **AlexNet** | 2012 | Deep + GPU |
| **VGG** | 2014 | Very deep |
| **ResNet** | 2015 | Skip connections |
| **EfficientNet** | 2019 | Scaling |
| **ViT** | 2020 | Pure attention |

---

## 💻 Code

```python
import torch.nn as nn

cnn = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 10)
)
```

---

## 🔗 Where This Topic Is Used

| Topic | How CNN Is Used |
|-------|-----------------|
| **ResNet** | Deep CNN with skip connections |
| **U-Net** | CNN encoder-decoder (diffusion, segmentation) |
| **YOLO** | CNN for object detection |
| **EfficientNet** | Scaled CNN architecture |
| **CLIP (image encoder)** | CNN or ViT for visual features |
| **Stable Diffusion** | U-Net backbone (CNN + attention) |
| **Face Recognition** | CNN for face embeddings |
| **Medical Imaging** | CNN for diagnosis |
| **Autonomous Driving** | CNN for perception |

### CNN Components Used In

| Component | Used By |
|-----------|---------|
| **Convolution** | All CNNs, U-Net in diffusion |
| **Pooling** | Downsampling in vision models |
| **ResNet blocks** | U-Net, backbone networks |
| **Depthwise Conv** | MobileNet, efficient models |

### Prerequisite For

```
CNN --> Object detection (YOLO, Faster R-CNN)
   --> Semantic segmentation (U-Net)
   --> Diffusion models (U-Net backbone)
   --> Vision encoders (CLIP)
   --> Medical image analysis
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📄 | AlexNet Paper | [NeurIPS 2012](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) |
| 📄 | ResNet Paper | [arXiv](https://arxiv.org/abs/1512.03385) |
| 🎓 | Stanford CS231n | [Course](http://cs231n.stanford.edu/) |
| 🇨🇳 | CNN详解 | [知乎](https://zhuanlan.zhihu.com/p/25249694) |
| 🇨🇳 | 卷积神经网络原理 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/88619088) |
| 🇨🇳 | CS231n中文 | [B站](https://www.bilibili.com/video/BV1nJ411z7fe) |

---

⬅️ [Back: Architectures](../)

---

➡️ [Next: Diffusion](../diffusion/)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=45B7D1&height=80&section=footer" width="100%"/>
</p>
