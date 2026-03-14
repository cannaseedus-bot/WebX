# KUHUL Geometric IR Specification

## Overview

The Geometric IR (GIR) is a flat, typed instruction-set representation that
sits between the AST and the final code generators.

## Instruction Set

| Opcode | Operands | Description |
|--------|----------|-------------|
| `ALLOC` | `name, elementType, shape[]` | Allocate a named tensor |
| `READ` | `name` | Load a tensor into the operand context |
| `WRITE` | `target, valueRef` | Store a value into a named slot |
| `OP` | `glyph, operand…` | Execute a glyph operation |
| `CALL` | `fnName, arg…` | Invoke an external function |
| `CONST` | `value` | Push a constant (after const-folding) |
| `FOLD_START` | – | Open a scoped fold block |
| `FOLD_END` | – | Close a fold block |
| `PHASE_START` | – | Open a phase cycle (0 → 2π) |
| `PHASE_END` | – | Close a phase cycle |

## Symbol Table

Each `GeometricIR` object carries a `Map<string, TensorType>` symbol table
populated by `ALLOC` instructions.

## Types

- `TensorType(elementType, shape)` – typed tensor descriptor.
- `Phase(angle)` – phase angle in [0, 2π].
- `Instruction(opcode, operands, metadata)` – a single IR instruction.
- `GeometricIR(instructions, symbolTable, metadata)` – a complete program.
- `ExecutableProgram(ir, target, code)` – compiled output.

## Example

```
ALLOC("X", "float32", [10])
FOLD_START()
  READ("X")
  OP("⊕", "X", "X")
  WRITE("Y", "X")
FOLD_END()
```
