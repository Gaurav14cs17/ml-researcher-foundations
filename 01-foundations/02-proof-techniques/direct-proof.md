# Direct Proof

> **Proving A → B by assuming A and deriving B**

---

## 📐 Structure

```
Theorem: If A, then B.

Proof:
1. Assume A is true.
2. [Logical steps...]
3. Therefore B is true.  □
```

---

## 🌍 ML Example

```
Theorem: If f is convex and ∇f(x*) = 0, then x* is a global minimum.

Proof:
1. Assume f is convex and ∇f(x*) = 0
2. By convexity: f(y) ≥ f(x*) + ∇f(x*)ᵀ(y - x*) for all y
3. Since ∇f(x*) = 0: f(y) ≥ f(x*) + 0 = f(x*)
4. Therefore f(y) ≥ f(x*) for all y
5. So x* is a global minimum  □
```

---

## 💻 When to Use

| Use Direct Proof When |
|----------------------|
| The implication is straightforward |
| You can chain definitions/theorems |
| The contrapositive is more complex |

---

---

⬅️ [Back: Contradiction](./contradiction.md) | ➡️ [Next: Induction](./induction.md)
