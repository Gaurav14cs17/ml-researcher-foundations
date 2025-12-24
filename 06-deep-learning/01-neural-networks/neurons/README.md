<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Neurons&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Neurons

> **The fundamental unit of neural networks**

---

## 🎯 Visual Overview

<img src="./images/neuron-perceptron.svg" width="100%">

*Caption: An artificial neuron (perceptron) computes a weighted sum of inputs plus bias, then applies an activation function. This simple computation, when combined in layers, can approximate any function (Universal Approximation Theorem).*

---

## 📂 Overview

The neuron is the basic building block of all neural networks. Understanding how a single neuron works is essential before studying complex architectures.

---

## 📐 Mathematical Definition

### Single Neuron Computation

```
Input: x = [x₁, x₂, ..., xₙ]ᵀ ∈ ℝⁿ
Weights: w = [w₁, w₂, ..., wₙ]ᵀ ∈ ℝⁿ
Bias: b ∈ ℝ

Linear combination:
    z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
    z = wᵀx + b  (vector notation)
    z = Σᵢ wᵢxᵢ + b

Output with activation:
    y = σ(z) = σ(wᵀx + b)
```

### The Perceptron (Binary Classification)

```
y = sign(wᵀx + b)

    ⎧  1  if wᵀx + b > 0
y = ⎨
    ⎩ -1  if wᵀx + b ≤ 0

Decision boundary: wᵀx + b = 0 (hyperplane)
```

### Perceptron Learning Rule

```
For misclassified point (xᵢ, yᵢ):
    w ← w + η yᵢ xᵢ
    b ← b + η yᵢ

Where η is the learning rate

Convergence: Guaranteed if data is linearly separable
```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **Weights (w)** | Learnable parameters that scale inputs |
| **Bias (b)** | Shifts the decision boundary |
| **Activation (σ)** | Non-linear function for expressiveness |
| **Linear Combination** | Weighted sum: wᵀx + b |

---

## 📊 Common Activation Functions

| Function | Formula | Range | Use |
|----------|---------|-------|-----|
| **Sigmoid** | σ(z) = 1/(1+e⁻ᶻ) | (0, 1) | Binary output |
| **Tanh** | tanh(z) = (eᶻ-e⁻ᶻ)/(eᶻ+e⁻ᶻ) | (-1, 1) | Zero-centered |
| **ReLU** | max(0, z) | [0, ∞) | Hidden layers |
| **Linear** | z | (-∞, ∞) | Regression output |

```
Why Non-linearity?

Without activation:
    Layer 1: h₁ = W₁x + b₁
    Layer 2: h₂ = W₂h₁ + b₂ = W₂(W₁x + b₁) + b₂
                            = (W₂W₁)x + (W₂b₁ + b₂)
                            = W'x + b'  ← Still linear!

Multiple linear layers = single linear layer!
Activation breaks this, enabling complex functions.
```

---

## 💻 Code Examples

### Single Neuron in NumPy

```python
import numpy as np

class Neuron:
    """A single artificial neuron"""
    
    def __init__(self, n_inputs, activation='sigmoid'):
        # Xavier initialization
        self.w = np.random.randn(n_inputs) / np.sqrt(n_inputs)
        self.b = 0.0
        self.activation = activation
    
    def _activate(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            return z  # Linear
    
    def forward(self, x):
        """Compute neuron output"""
        z = np.dot(self.w, x) + self.b  # Linear combination
        return self._activate(z)  # Activation

# Usage
neuron = Neuron(n_inputs=3, activation='sigmoid')
x = np.array([1.0, 2.0, 3.0])
output = neuron.forward(x)
print(f"Neuron output: {output}")
```

### Perceptron Learning

```python
def perceptron_train(X, y, max_epochs=100, lr=1.0):
    """
    Perceptron learning algorithm
    X: (n_samples, n_features)
    y: (n_samples,) with values {-1, +1}
    """
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    
    for epoch in range(max_epochs):
        errors = 0
        for xi, yi in zip(X, y):
            prediction = np.sign(np.dot(w, xi) + b)
            if prediction != yi:
                # Misclassified! Update
                w += lr * yi * xi
                b += lr * yi
                errors += 1
        
        if errors == 0:
            print(f"Converged at epoch {epoch}")
            break
    
    return w, b
```

### PyTorch Neuron

```python
import torch
import torch.nn as nn

# Single neuron as a Linear layer with 1 output
class SingleNeuron(nn.Module):
    def __init__(self, n_inputs, activation='sigmoid'):
        super().__init__()
        self.linear = nn.Linear(n_inputs, 1)
        self.activation = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()
    
    def forward(self, x):
        z = self.linear(x)
        return self.activation(z)

# Usage
neuron = SingleNeuron(n_inputs=10)
x = torch.randn(32, 10)  # Batch of 32 samples
output = neuron(x)  # Shape: (32, 1)
```

---

## 🧠 Biological vs Artificial Neurons

| Aspect | Biological | Artificial |
|--------|-----------|------------|
| **Inputs** | Dendrites | Input vector x |
| **Weights** | Synaptic strength | Learned parameters w |
| **Sum** | Cell body | wᵀx + b |
| **Activation** | Action potential | σ(z) |
| **Output** | Axon | Single value y |

---

## 📊 Limitations of Single Neuron

```
XOR Problem (Cannot solve with one neuron!):

    (0,0) → 0     ●---------○ (1,0)
                  |    ╲    |
                  |     ╲   |  No single line
                  |      ╲  |  separates ● from ○
    (0,1) → 1     ○---------● (1,1)

Solution: Multiple layers (MLPs)!
```

---

## 🔗 Connection to Other Topics

```
Neuron (basic unit)
    |
    +-- Layers (many neurons)
    +-- MLPs (stacked layers)
    +-- Activations (non-linearity)
    +-- Backpropagation (learning)
    +-- Universal Approximation (theory)
```

---

## 📚 References

| Type | Title | Link |
|------|-------|------|
| 📖 | Layers | [../layers/](../layers/) |
| 📖 | Activations | [../activations/](../activations/) |
| 🎥 | 3Blue1Brown: Neural Networks | [YouTube](https://www.youtube.com/watch?v=aircAruvnKk) |
| 📄 | Rosenblatt Perceptron | [Paper (1958)](https://psycnet.apa.org/record/1959-09865-001) |
| 🇨🇳 | 神经元与感知机详解 | [知乎](https://zhuanlan.zhihu.com/p/30844948) |
| 🇨🇳 | 感知机算法实现 | [CSDN](https://blog.csdn.net/qq_37466121/article/details/87902069) |
| B站 | 神经网络基础 | 从单个神经元开始 |
| 机器之心 | 深度学习入门 | 神经元原理解析 |


## 🔗 Where This Topic Is Used

| Application | How Neurons Are Used |
|-------------|---------------------|
| **All Neural Networks** | Building block |
| **Universal Approximation** | Function approximation |
| **Deep Learning** | Stacked nonlinear transforms |
| **Perceptron** | Historical first NN |

---

⬅️ [Back: Neural Networks](../)

---

⬅️ [Back: Layers](../layers/)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
