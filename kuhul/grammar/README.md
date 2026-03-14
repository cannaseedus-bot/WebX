# KUHUL Grammar Module

This directory contains the EBNF grammar for the KUHUL language and the
JavaScript tooling for parsing and validating that grammar.

## Files

| File | Purpose |
|------|---------|
| `kuhul.ebnf` | Formal EBNF grammar for the KUHUL language |
| `ebnf-parser.js` | Parses `.ebnf` files into a structured `{ rules, start }` object |
| `grammar-validator.js` | Validates a parsed grammar for completeness and consistency |

## Quick start

```js
import { parseEBNF } from './ebnf-parser.js';
import { validateGrammar } from './grammar-validator.js';
import { readFileSync } from 'fs';

const source = readFileSync('kuhul.ebnf', 'utf8');
const grammar = parseEBNF(source);
const { valid, errors } = validateGrammar(grammar);
if (!valid) console.error(errors);
```

## EBNF conventions

- `=`  separates rule name from its definition
- `;`  terminates a rule
- `|`  separates alternatives
- `{ }` zero-or-more repetition
- `[ ]` optional group (zero-or-one)
- `( )` grouping
- `"…"` literal terminal
- `(* … *)` comments
