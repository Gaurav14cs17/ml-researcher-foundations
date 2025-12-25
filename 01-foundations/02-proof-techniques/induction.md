<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=150&section=header&text=Mathematical%20Induction&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=flat-square" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=flat-square" alt="Updated"/>
</p>

---

# Mathematical Induction

> **Proving statements for all natural numbers**

---

## 📐 Structure

```
Theorem: P(n) holds for all n ≥ 1.

Proof by induction:
1. Base case: Show P(1) is true.
2. Inductive step: Assume P(k) is true for some k ≥ 1.
   Show P(k+1) is true.
3. By induction, P(n) holds for all n ≥ 1.  □
```

---

## 🌍 ML Example: Backpropagation Correctness

```
Theorem: Backprop correctly computes ∂L/∂wₗ for all layers l.

Proof:
Base case (l = L, output layer):
  ∂L/∂wₗ = ∂L/∂yₗ · ∂yₗ/∂wₗ  ✓ (direct computation)

Inductive step: Assume correct for layer l+1.
  ∂L/∂wₗ = ∂L/∂yₗ₊₁ · ∂yₗ₊₁/∂yₗ · ∂yₗ/∂wₗ
         = (correct by hypothesis) · (chain rule)  ✓

By induction, correct for all layers.  □
```

---

## 💻 Code Pattern

```python
def factorial(n):
    """
    Correctness by induction:
    Base: factorial(0) = 1 ✓
    Step: factorial(n) = n * factorial(n-1) = n * (n-1)! = n! ✓
    """
    if n == 0:
        return 1  # Base case
    return n * factorial(n - 1)  # Inductive step
```

---

---

⬅️ [Back: Direct Proof](./direct-proof.md)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=80&section=footer" width="100%"/>
</p>
