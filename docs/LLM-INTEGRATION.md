# K'UHUL++ LLM Integration Guide

## Overview

K'UHUL++ includes an LLM integration layer (`kuhul/llm/`) that enhances the developer
experience with AI-powered features. All LLM calls are optional — the tools fall back to
rule-based heuristics when no provider is configured.

## Components

### SyntaxCompleter (`kuhul/llm/syntax-completer.ts`)

Provides code-completion suggestions based on grammar rules and (optionally) an LLM.

```typescript
import { SyntaxCompleter } from './kuhul/llm/syntax-completer.js';

// Rule-based only
const completer = new SyntaxCompleter();
const suggestions = await completer.complete('Ten', 3);
// → [{ text: 'Tensor', kind: 'keyword', priority: 0 }, ...]

// With LLM provider
const llmCompleter = new SyntaxCompleter(async (prefix) => {
    const response = await fetch('/api/complete', { method: 'POST', body: prefix });
    return (await response.json()).completions;
});
```

### GlyphValidator (`kuhul/llm/glyph-validator.ts`)

Validates glyph expressions for type compatibility.

```typescript
import { GlyphValidator } from './kuhul/llm/glyph-validator.js';

const validator = new GlyphValidator();

// String-based validation
const result = validator.validate('a ⊗ b');

// Type-based validation
const typeResult = validator.validateTypes('⊕',
    { kind: 'tensor', dtype: 'float32', shape: [4, 4] },
    { kind: 'tensor', dtype: 'float32', shape: [4, 4] },
);
```

### LLMOptimizer (`kuhul/llm/optimizer.ts`)

Suggests optimizations for a Geometric IR program.

```typescript
import { LLMOptimizer } from './kuhul/llm/optimizer.js';

const optimizer = new LLMOptimizer();
const suggestions = await optimizer.suggest(ir);
for (const s of suggestions) {
    console.log(`${s.title} (${s.estimatedSpeedup}x): ${s.description}`);
}
```

### DocGenerator (`kuhul/llm/doc-generator.ts`)

Generates Markdown documentation from an AST.

```typescript
import { DocGenerator } from './kuhul/llm/doc-generator.js';

const gen = new DocGenerator('My KUHUL Module');
const docs = gen.generateDocs(ast);
console.log(docs.markdown);
```

### DiagnosticEngine (`kuhul/llm/diagnostic-engine.ts`)

Enhances raw errors with user-friendly messages and fix suggestions.

```typescript
import { DiagnosticEngine } from './kuhul/llm/diagnostic-engine.js';

const engine = new DiagnosticEngine();
const diag = await engine.enhance(error);
console.log(diag.message);
console.log(diag.suggestions[0].title);
```

## Connecting a Real LLM

Any async function matching `(prompt: string) => Promise<string>` can be used as an LLM
provider. Example with OpenAI:

```typescript
const llmProvider = async (prompt: string) => {
    const res = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${API_KEY}`, 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'gpt-4o', messages: [{ role: 'user', content: prompt }] }),
    });
    const data = await res.json();
    return data.choices[0].message.content;
};

const optimizer = new LLMOptimizer(llmProvider);
```

---

*See also: [API.md](API.md)*
