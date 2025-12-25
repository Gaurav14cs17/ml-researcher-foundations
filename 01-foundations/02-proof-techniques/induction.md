<!-- Animated Header -->
<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=120&section=header&text=Mathematical%20Induction&fontSize=32&fontColor=fff&animation=twinkling&fontAlignY=35" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Section-01-6C63FF?style=for-the-badge&logo=bookstack&logoColor=white" alt="Section"/>
  <img src="https://img.shields.io/badge/Author-Gaurav_Goswami-blue?style=for-the-badge" alt="Author"/>
  <img src="https://img.shields.io/badge/Updated-December_2024-green?style=for-the-badge" alt="Updated"/>
</p>

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

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

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=6C63FF&height=80&section=footer" width="100%"/>
</p>
