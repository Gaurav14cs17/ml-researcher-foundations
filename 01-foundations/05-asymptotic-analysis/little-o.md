<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=120&section=header&text=Little-o%20Notation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-01-6C63FF?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=80&section=footer" width="100%"/>
</p>
