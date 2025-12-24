# Proof by Contradiction

> **Proving A by showing ¬A leads to impossibility**

---

## 📐 Structure

```
Theorem: A is true.

Proof:
1. Assume A is false (¬A).
2. Derive a contradiction.
3. Therefore A must be true.  □
```

---

## 🌍 Classic Example

```
Theorem: √2 is irrational.

Proof:
1. Assume √2 = p/q in lowest terms (p,q coprime)
2. Then 2 = p²/q², so p² = 2q²
3. p² is even → p is even → p = 2k
4. So 4k² = 2q² → q² = 2k² → q is even
5. Both p,q even contradicts "lowest terms"
6. Therefore √2 is irrational.  □
```

---

## 🌍 ML Example

```
Theorem: No algorithm can perfectly minimize non-convex loss.

Proof by contradiction:
1. Assume algorithm A finds global min for any non-convex f
2. Use A to solve NP-hard problems
3. This would imply P = NP
4. Contradiction (under standard assumptions)
5. Therefore no such algorithm exists.  □
```

---

## 💻 When to Use

| Use Contradiction When |
|------------------------|
| Direct proof is hard |
| Statement is "there is no..." |
| Uniqueness proofs |
| Impossibility results |

---

---

➡️ [Next: Direct Proof](./direct-proof.md)
