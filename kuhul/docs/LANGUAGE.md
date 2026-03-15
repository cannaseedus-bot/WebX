# KUHUL Language Specification

KUHUL is a tensor-oriented language for geometric computation, designed for
GPU-accelerated pipelines.  Every program is a sequence of *statements*, each
enclosed in square brackets `[…]`.

---

## 1. Syntax Overview

KUHUL uses bracket-based syntax.  A statement is always:

```
[Keyword arg1 arg2 ...]
```

There are seven statement forms:

| Form | Syntax | Description |
|------|--------|-------------|
| Fold | `[Pop] … [Xul]` | Scoped block |
| Allocation | `[Wo name tensor<type, shape>]` | Declare a tensor |
| Read | `[Yax name]` | Load a tensor into context |
| Write | `[Ch'en name value]` | Store a value |
| Operation | `[Sek glyph operands…]` | Execute a glyph operation |
| Phase cycle | `[K'ayab'] … [Kumk'u]` | Loop over 0 → 2π |
| Invocation | `[Muwan fn args…]` | Call an external function |

---

## 2. Keywords

| Keyword | Role |
|---------|------|
| `Pop` | Open a fold (block) |
| `Xul` | Close a fold |
| `Wo` | Allocate a tensor |
| `Yax` | Read / load a tensor |
| `Ch'en` | Write / store a value |
| `Sek` | Execute a glyph operation |
| `K'ayab'` | Open a phase cycle |
| `Kumk'u` | Close a phase cycle |
| `Muwan` | Invoke an external function |

---

## 3. Glyph Operators

Glyph operators are Unicode symbols used in `[Sek glyph …]` statements.

| Glyph | Name | Operation |
|-------|------|-----------|
| `⊗` | Tensor product | Matrix multiplication |
| `⊕` | Addition | Element-wise addition |
| `⊖` | Subtraction | Element-wise subtraction |
| `⊛` | Convolution | 1-D / 2-D convolution |
| `⊜` | Equality | Element-wise equality |
| `⊝` | Negation | Element-wise negation |
| `⊞` | Direct sum | Concatenation / direct sum |

---

## 4. Tensor Types

Tensor types are written as:

```
tensor<elementType, [dim1, dim2, ...]>
```

Supported element types: `float32`, `float64`, `int32`, `int64`, `uint8`,
`bool`, `complex64`.

Dimensions are positive integers; `?` denotes a dynamic dimension.

---

## 5. Semantics

### Scope

`[Pop]…[Xul]` folds introduce a new scope.  Identifiers declared with `[Wo]`
inside a fold are not visible outside it.

### Phase Cycles

`[K'ayab']…[Kumk'u]` executes the enclosed body repeatedly, advancing the
phase angle from `0` to `2π` in discrete steps of `π/8`.

### Execution Model

Programs execute sequentially.  The runtime maintains a *manifold* `M`, a
typed memory space indexed by identifier names.

---

## 6. Example

```kuhul
[Pop]
  [Wo X tensor<float32, [1024]>]
  [Wo W tensor<float32, [1024, 512]>]

  [K'ayab']
    [Yax X]
    [Yax W]
    [Sek ⊗ X W]
    [Ch'en XW X]
  [Kumk'u]
[Xul]
```
