# K'UHUL++ Language Specification v2.0

## Overview

K'UHUL++ (pronounced *koo-hool plus-plus*) is a domain-specific language for GPU-accelerated
geometric computing. It combines JavaScript-like syntax with Unicode glyph operators that
map directly to compute shader kernels on the **Manifold M** — a π-geometry execution substrate.

---

## 1. Lexical Structure

### 1.1 Comments

```
// Single line comment
/* Multi-line
   block comment */
```

### 1.2 Literals

| Kind          | Example         | Token type |
|---------------|-----------------|------------|
| Integer       | `42`            | `NUMBER`   |
| Float         | `3.14`          | `NUMBER`   |
| String        | `"hello"`       | `STRING`   |
| Boolean       | `true` / `false`| `KEYWORD`  |
| π constant    | `π`             | `PI`       |
| π coefficient | `0.5π`, `2π`   | `PI_EXPR`  |

### 1.3 Identifiers

Identifiers begin with a letter or `_` and contain letters, digits, and `_`.

### 1.4 Glyph Operators

| Glyph | Name                | Description                            |
|-------|---------------------|----------------------------------------|
| `⊗`   | Geometric Product   | Matrix multiply / outer product        |
| `⊕`   | Translation         | Bias addition in manifold M            |
| `⊖`   | Difference          | Subtraction in manifold M              |
| `⊛`   | Convolution         | Discrete convolution in M              |
| `⊜`   | Identity            | Identity / copy element               |
| `⊝`   | Complement          | Negation / complement in M             |
| `⊞`   | Union               | Element-wise addition                  |
| `⤍`   | Vector Encrypt      | Affine transform on a vector field     |
| `↻`   | Rotational Compress | Geometry compression via Y-axis rotate |
| `⟲`   | Spherical Loop      | Cartesian ↔ spherical round-trip       |
| `∿`   | Torsion Field       | Twist deformation along Y              |
| `⊙`   | Radial Projection   | Project onto sphere of given radius    |
| `≋`   | Wave Modulate       | Wave modulation on mesh surfaces       |

---

## 2. Declarations

### 2.1 Tensor Declaration

```kuhul
Tensor name = expression;
Tensor name<TypeParam> = expression;
```

Examples:
```kuhul
Tensor v     = [1.0, 2.0, 3.0];
Tensor angle = 0.5π;
Tensor msg   = "Hello KUHUL";
```

### 2.2 Cluster Declaration

Groups related tensors into a named namespace:
```kuhul
Cluster weights {
    Tensor w1 = [1.0, 2.0];
    Tensor w2 = [3.0, 4.0];
}
```

### 2.3 Model Declaration

Declares a neural model structure:
```kuhul
Model linear {
    Tensor weights = [1.0, 0.0, 0.0, 1.0];
    Tensor bias    = [0.0, 0.0];
}
```

### 2.4 Pipeline Declaration

Defines a GPU compute pipeline:
```kuhul
Pipeline myPipeline {
    Tensor stage1 = input ⊗ weights;
    Tensor stage2 = stage1 ⊕ bias;
}
```

---

## 3. Glyph Operations

Glyph operations apply a geometric transform from the manifold M to one or two tensor operands:

```kuhul
Tensor result = left ⊗ right;    // Geometric product
Tensor out    = vertices ⤍ matrix; // Affine transform
Tensor rotated = mesh ↻ 45.0;    // Rotational compression
Tensor twisted = body ∿ 0.1;     // Torsion field
```

Glyphs can be chained:
```kuhul
Tensor out = input ⊗ weights ⊕ bias;
```

---

## 4. Control Flow

### 4.1 If / Else
```kuhul
if (condition) { Tensor a = 1; } else { Tensor b = 2; }
```

### 4.2 While Loop
```kuhul
while (running) { /* body */ }
```

### 4.3 For Loop
```kuhul
for (Tensor i = 0; i < 10; i = i + 1) { /* body */ }
```

### 4.4 Parallel For (GPU)
```kuhul
parallel for (item in collection) { /* body */ }
```

### 4.5 Foreach Glyph
```kuhul
foreach glyph ⊗ in vertex data { /* body */ }
```

---

## 5. π-Geometry Execution

K'UHUL++ programs execute on the **Manifold M** — a geometric space parameterised by a
phase angle φ ∈ [0, 2π]. The current phase influences wave and spherical loop operations.

Phase operations:
- `π` — the mathematical constant π ≈ 3.14159
- `0.5π` — half-turn (90°)
- `2π` — full cycle

---

## 6. GPU Dispatch

```kuhul
GPU.Dispatch(pipeline, [groupsX, groupsY, groupsZ]);
GPU.Draw(pipeline, vertexCount);
GPU.Train(model);
```

---

## 7. Built-in Functions

| Function              | Description                             |
|-----------------------|-----------------------------------------|
| `generate_spiral(n)`  | Generate n spiral points                |
| `load_dataset(name)`  | Load a named dataset tensor             |
| `split_tensors(t, n)` | Split tensor into n sub-tensors         |
| `merge_clusters(c)`   | Merge a cluster into a single tensor    |

---

## 8. File Extensions

- `.kuhul` — standard K'UHUL++ source file
- `.kpp`   — alternate extension (K++ shorthand)

---

*See also: [GRAMMAR.md](GRAMMAR.md), [COMPILER.md](COMPILER.md), [EXAMPLES.md](EXAMPLES.md)*
