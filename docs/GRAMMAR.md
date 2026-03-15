# K'UHUL++ Grammar Reference

The complete grammar is defined in EBNF format at `grammar-ebnf/KUHUL-LLM.ebnf` (v7.0.0, 77KB).
A condensed reference excerpt is in `kuhul/grammar/kuhul.ebnf`.

## Quick Reference

```ebnf
Program         = { Statement } EOF

Statement       = TensorDecl | ClusterDecl | ModelDecl | PipelineDecl
                | IfStmt | ForStmt | WhileStmt | ParallelFor | ForEachGlyph
                | TrainStmt | ReturnStmt | ExpressionStatement

TensorDecl      = 'Tensor' Identifier ['<' TypeParams '>'] '=' Expression ';'
ClusterDecl     = 'Cluster' Identifier '{' { Statement } '}'
ModelDecl       = 'Model'   Identifier '{' { Statement } '}'
PipelineDecl    = 'Pipeline' Identifier '{' { Statement } '}'

Expression      = AssignExpr
AssignExpr      = GlyphExpr ['=' AssignExpr]
GlyphExpr       = AdditiveExpr { GlyphOp AdditiveExpr }
GlyphOp         = '⊗'|'⊕'|'⊖'|'⊛'|'⊜'|'⊝'|'⊞'|'⤍'|'↻'|'⟲'|'∿'|'⊙'|'≋'

PrimaryExpr     = Identifier | NumberLiteral | StringLiteral
                | PiExpr | ArrayLiteral | '(' Expression ')'

PiExpr          = [FloatLiteral] 'π'
```

## Full EBNF Location

See `grammar-ebnf/KUHUL-LLM.ebnf` for the complete specification including:
- Unicode character classes and whitespace rules
- Complete expression grammar with all operators
- Agent and Cluster semantics (ECMAScript §9.6–9.9)
- Manifold M type system
- Phase array and π-geometry execution rules
- Policy enforcement and entanglement rules

---

*See also: [KUHUL.md](KUHUL.md), [COMPILER.md](COMPILER.md)*
