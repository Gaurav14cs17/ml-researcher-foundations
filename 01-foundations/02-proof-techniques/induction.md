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
