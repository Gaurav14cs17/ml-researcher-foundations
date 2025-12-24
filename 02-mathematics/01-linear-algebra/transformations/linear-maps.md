# Linear Transformations

> **Functions that preserve vector space structure**

---

## 📐 Definition

```
T: V → W is linear if:

1. T(u + v) = T(u) + T(v)   (additivity)
2. T(cv) = cT(v)            (homogeneity)

Equivalently: T(au + bv) = aT(u) + bT(v)
```

---

## 🔑 Matrix Representation

```
Every linear map T: ℝⁿ → ℝᵐ can be written as:

T(x) = Ax

Where A is m×n matrix.

Column j of A = T(eⱼ) where eⱼ is j-th standard basis
```

---

## 📊 Examples

| Transformation | Matrix |
|----------------|--------|
| Scaling by k | kI |
| Rotation by θ | [[cos θ, -sin θ], [sin θ, cos θ]] |
| Reflection | [[1, 0], [0, -1]] |
| Projection | [[1, 0], [0, 0]] |

---

## 💻 Code

```python
import numpy as np

def rotate_2d(x, theta):
    """Rotate vector by theta radians"""
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    return R @ x

# Example
v = np.array([1, 0])
rotate_2d(v, np.pi/2)  # [0, 1] - rotated 90 degrees
```

---

## 🌍 In Deep Learning

| Layer | Transformation |
|-------|----------------|
| Linear/Dense | Wx + b (affine) |
| Convolution | Linear (in high dim) |
| Attention | QKᵀV (linear combo) |

---

<- [Back](./README.md)


