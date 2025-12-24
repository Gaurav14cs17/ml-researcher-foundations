# 🧠 Neural Network Basics

> **The building blocks of deep learning**

## 🎯 Visual Overview

<img src="./images/neural-network-architecture.svg" width="100%">

*Caption: Complete multi-layer perceptron (MLP) architecture showing input layer, multiple hidden layers, and output layer. Each connection represents a weight, and the diagram illustrates both forward pass (data flow) and backward pass (gradient flow via backpropagation). This is the fundamental structure underlying all deep learning models.*

<img src="./images/neural-network.svg" width="100%">

---

## 📂 Topics

| Folder | Topic | Key Concepts |
|--------|-------|--------------|
| [neurons/](./neurons/) | Single neuron | Perceptron, activation |
| [layers/](./layers/) | Layer types | Dense, Conv, Attention |
| [activations/](./activations/) | Activation functions | ReLU, GELU, Softmax |
| [initialization/](./initialization/) | Weight initialization | He, Xavier |

---

## 📐 Basic Neuron

```
y = σ(wᵀx + b)

Where:
• x = input vector
• w = weight vector
• b = bias
• σ = activation function
• y = output
```

---

## 🏗️ Network = Stack of Layers

```
Input x
    |
    v
+---------+
| Layer 1 | --> h₁ = σ(W₁x + b₁)
+---------+
    |
    v
+---------+
| Layer 2 | --> h₂ = σ(W₂h₁ + b₂)
+---------+
    |
    v
   ...
    |
    v
+---------+
| Output  | --> ŷ = σ(Wₗhₗ₋₁ + bₗ)
+---------+
```

---

## 🔗 Where This Topic Is Used

| Topic | How Neural Network Basics Are Used |
|-------|-----------------------------------|
| **Transformers** | Built from linear layers + attention |
| **CNN** | Convolutional layers = specialized neurons |
| **RNN / LSTM** | Recurrent connections between neurons |
| **ResNet** | Skip connections + dense layers |
| **BERT / GPT** | Massive neural networks (billions of params) |
| **Diffusion Models** | U-Net architecture with residual blocks |
| **MoE** | Expert networks = neural network modules |
| **GAN** | Generator + Discriminator networks |
| **VAE** | Encoder + Decoder networks |
| **Autoencoders** | Compress via neural network bottleneck |

### Prerequisite For

```
Neural Networks --> Backpropagation
               --> CNN, RNN, Transformer
               --> All deep learning architectures
               --> Transfer learning / Fine-tuning
```

### Concepts That Build On This

| Concept | Builds On |
|---------|-----------|
| Backprop | How neurons learn (gradient flow) |
| Batch/LayerNorm | Normalize neuron activations |
| Dropout | Randomly disable neurons |
| Attention | Weighted sum of neuron outputs |

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 🎥 | 3Blue1Brown: Neural Networks | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| 📖 | Deep Learning Book Ch. 6 | [Book](https://www.deeplearningbook.org/contents/mlp.html) |
| 📖 | PyTorch Tutorial | [Docs](https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) |
| 🇨🇳 | 神经网络基础 | [知乎](https://zhuanlan.zhihu.com/p/25110450) |
| 🇨🇳 | 深度学习入门 | [B站](https://www.bilibili.com/video/BV1J94y1f7u5) |
| 🇨🇳 | 动手学深度学习 | [d2l.ai中文](https://zh.d2l.ai/) |

---

⬅️ [Back: Deep Learning](../) | ➡️ [Next: 02-Backpropagation](../02-backpropagation/)

