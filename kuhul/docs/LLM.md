# KUHUL LLM Integration Guide

The `kuhul/llm/` module provides LLM-assisted tooling for the KUHUL language.
All modules expose an async API so they can be backed by a real LLM service
without changing call-sites.

## SyntaxCompleter

Provides code-completion suggestions for partial KUHUL source.

```js
import { SyntaxCompleter } from './llm/syntax-completer.js';
const c = new SyntaxCompleter();
const suggestions = await c.complete(source, cursorOffset);
// [{ text, kind, detail }, ...]
```

## GlyphValidator

Validates glyph operator usage (arity, type compatibility).

```js
import { GlyphValidator } from './llm/glyph-validator.js';
const v = new GlyphValidator();
const { valid, messages } = await v.validate(source);
```

## LLMOptimizer

Wraps the rule-based `IROptimizer` with LLM-guided heuristics.

```js
import { LLMOptimizer } from './llm/optimizer.js';
const opt = new LLMOptimizer();
const optimizedIR = await opt.optimize(ir);
```

## DocGenerator

Generates Markdown documentation from KUHUL source.

```js
import { DocGenerator } from './llm/doc-generator.js';
const gen = new DocGenerator();
const markdown = await gen.generateDocs(source);
```

## DiagnosticEngine

Enriches compiler errors with human-readable explanations and fix suggestions.

```js
import { DiagnosticEngine } from './llm/diagnostic-engine.js';
const engine = new DiagnosticEngine();
const enhanced = await engine.enhance(error, source);
// { title, message, explanation, fix, sourceLine }
```

## Replacing with a real LLM

Each module's async methods can be overridden by subclassing or monkey-patching
to call an external API (e.g. OpenAI, Anthropic, Ollama) before falling back to
the built-in rule-based implementation.
