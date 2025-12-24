# Big-O Notation

> **Upper bounds on growth rate**

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
