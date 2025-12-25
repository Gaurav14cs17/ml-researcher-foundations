<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Littleo%20Notation&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Little-o Notation

> **Strictly smaller growth rates**

---

## 📐 Definition

```
f(n) = o(g(n)) means:

lim_{n→∞} f(n)/g(n) = 0

"f grows strictly slower than g"
```

---

## 📊 Comparison with Big-O

| Big-O | Little-o | Meaning |
|-------|----------|---------|
| f = O(g) | f ≤ c·g eventually | At most |
| f = o(g) | f/g → 0 | Strictly less |

---

## 🌍 Examples

```
n = o(n²)           # n grows slower than n²
log n = o(n)        # log grows slower than linear
n² = o(2ⁿ)          # polynomial < exponential
n^k = o(n^{k+1})    # lower power < higher power
```

---

## 🔑 Common Hierarchy

```
1 < log log n < log n < n^ε < n < n log n < n² < ... < 2^n < n!

Each is o() of the next
```

---

## 💻 Code Example

```python
import numpy as np
import matplotlib.pyplot as plt

ns = np.arange(1, 100)

# f = o(g) means f/g → 0
ratios = {
    'log(n)/n': np.log(ns) / ns,
    'n/n²': ns / ns**2,
    'n²/2ⁿ': ns**2 / 2**ns,
}

for label, ratio in ratios.items():
    plt.plot(ns, ratio, label=label)
    
plt.legend()
plt.xlabel('n')
plt.ylabel('f(n)/g(n)')
plt.title('Little-o: ratio → 0')
```

---

---

⬅️ [Back: Big O](./big-o.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
