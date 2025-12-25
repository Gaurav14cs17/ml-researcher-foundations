<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=120&section=header&text=Big-O%20Notation&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
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
f(n) = O(g(n)) means:

∃ c > 0, n₀ such that ∀ n ≥ n₀:
f(n) ≤ c · g(n)

"f grows at most as fast as g"
```

---

## 📊 Common Complexities

| Big-O | Name | Example |
|-------|------|---------|
| O(1) | Constant | Hash lookup |
| O(log n) | Logarithmic | Binary search |
| O(n) | Linear | Array scan |
| O(n log n) | Linearithmic | Merge sort |
| O(n²) | Quadratic | Attention |
| O(n³) | Cubic | Matrix multiply |
| O(2ⁿ) | Exponential | Subset enumeration |

---

## 🌍 ML Examples

| Operation | Complexity |
|-----------|------------|
| Transformer attention | O(n²d) |
| Convolution | O(k²·c·h·w) |
| Matrix multiply | O(n³) |
| Softmax | O(n) |

---

## 💻 Code

```python
# O(n) - Linear
def linear_search(arr, target):
    for x in arr:  # n iterations
        if x == target:
            return True
    return False

# O(n²) - Quadratic
def bubble_sort(arr):
    for i in range(len(arr)):      # n
        for j in range(len(arr)):  # × n
            if arr[i] < arr[j]:
                arr[i], arr[j] = arr[j], arr[i]
```

---

---

➡️ [Next: Little O](./little-o.md)

---

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=80&section=footer" width="100%"/>
</p>
