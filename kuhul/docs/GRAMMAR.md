# KUHUL Grammar Reference

The formal EBNF grammar for the KUHUL language is in
[`../grammar/kuhul.ebnf`](../grammar/kuhul.ebnf).

## Summary

```ebnf
program     = { statement } ;
statement   = fold | allocation | read | write
            | operation | phase_cycle | invocation ;

fold        = "[" "Pop" "]" { statement } "[" "Xul" "]" ;
allocation  = "[" "Wo" identifier tensor_type_spec "]" ;
read        = "[" "Yax" identifier "]" ;
write       = "[" "Ch'en" identifier value "]" ;
operation   = "[" "Sek" glyph operand { operand } "]" ;
phase_cycle = "[" "K'ayab'" "]" { statement } "[" "Kumk'u" "]" ;
invocation  = "[" "Muwan" identifier { value } "]" ;

tensor_type_spec = "tensor" "<" element_type "," shape ">" ;
element_type = "float32" | "float64" | "int32" | "int64"
             | "uint8" | "bool" | "complex64" ;
shape        = "[" dim { "," dim } "]" ;
dim          = integer | "?" ;
glyph        = "⊗" | "⊕" | "⊖" | "⊛" | "⊜" | "⊝" | "⊞" ;
```

## Parsing

Use `kuhul/grammar/ebnf-parser.js` to parse the `.ebnf` file programmatically:

```js
import { parseEBNF } from '../grammar/ebnf-parser.js';
import { readFileSync } from 'fs';

const grammar = parseEBNF(readFileSync('kuhul.ebnf', 'utf8'));
console.log([...grammar.rules.keys()]);
```
