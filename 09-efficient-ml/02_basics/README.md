<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=02 Basics&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---


# Lecture 2: Neural Network Basics

[← Back to Course](../README.md) | [← Previous](../01_introduction/README.md) | [Next: Pruning I →](../03_pruning_sparsity_1/README.md)

📺 [Watch Lecture 2 on YouTube](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB&index=2)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/ml-researcher-foundations/blob/main/09-efficient-ml/02_basics/demo.ipynb) ← **Try the code!**

---

![Overview](overview.png)

## Compute Primitives

### FLOPS vs Memory

Understanding efficiency requires knowing what's expensive:

| Operation | Compute | Memory |
|-----------|---------|--------|
| Matrix multiply | High | Low |
| Attention | O(N²) | O(N²) |
| Element-wise | Low | High (bandwidth limited) |

---

## Roofline Model

The **roofline model** helps understand whether your code is:
- **Compute-bound**: Limited by FLOPS (matrix ops)
- **Memory-bound**: Limited by memory bandwidth (element-wise ops)

```
         /----------------- Compute ceiling
        /
       /
      /
-----/  <-- Memory bandwidth ceiling
    |
Arithmetic Intensity (FLOPS/Byte)
```

---

## Key Neural Network Layers

### 1. Convolution
```python
# Memory: O(C_in × C_out × K × K)
# Compute: O(C_in × C_out × K² × H × W)
```

### 2. Linear (Dense)
```python
# Memory: O(in_features × out_features)
# Compute: O(batch × in_features × out_features)
```

### 3. Attention
```python
# Memory: O(N²) for attention matrix
# Compute: O(N² × d) for QK^T and attention × V
```

---

## Hardware Considerations

| Hardware | Good At | Limited By |
|----------|---------|------------|
| CPU | Flexibility | Parallelism |
| GPU | Massive parallelism | Memory bandwidth |
| TPU | Matrix ops | Flexibility |
| MCU | Energy efficiency | Everything |

---

## Efficiency Metrics

1. **Latency** - Time for single inference
2. **Throughput** - Inferences per second
3. **Energy** - Joules per inference
4. **Model size** - Parameters × bytes per param
5. **Peak memory** - Max RAM during inference

---

## Important Equations

**MAC (Multiply-Accumulate):**
```
MACs = number of multiply-add operations
FLOPs ≈ 2 × MACs (multiply + add)
```

**Memory Bandwidth:**
```
Time = Data Size / Bandwidth
```

---

---

## 📐 Mathematical Foundations

### Roofline Model

**Arithmetic Intensity:**
```
I = \frac{\text{FLOPs}}{\text{Bytes accessed}}
```

**Attainable Performance:**
```
P = \min(\text{Peak FLOPs}, I \times \text{Bandwidth})
```

### Layer Complexity

**Convolution:**
```
\text{FLOPs} = 2 \times C_{in} \times C_{out} \times K^2 \times H \times W
```

**Attention:**
```
\text{FLOPs} = 4 \times N^2 \times d + 2 \times N \times d^2
```

---

## 🎯 Where Used

| Concept | Applications |
|---------|-------------|
| Roofline Analysis | Performance optimization |
| FLOPs Counting | Model comparison |
| Memory Profiling | Batch size optimization |
| Arithmetic Intensity | Kernel optimization |

---

## 📚 References

| Type | Resource | Link |
|------|----------|------|
| 📄 | Roofline Model | [Berkeley](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf) |
| 💻 | PyTorch Profiler | [PyTorch](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) |
| 🎥 | MIT 6.5940 TinyML | [Course](https://hanlab.mit.edu/courses/2024-fall-65940) |
| 🇨🇳 | 知乎 - 神经网络效率分析 | [Zhihu](https://www.zhihu.com/topic/20069893) |


---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
