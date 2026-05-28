// K'UHUL compiler — Phases 1-4 (KUHUL.v1.0.0)
// ES module port of kuhulc.cjs — fs/path CLI wrapper removed; pure in-memory.
//
// Glyph language surface:
//   [Pop name body...]     — function definition
//   [Wo name shape=N dtype=float32] — tensor declaration
//   [Yax name]             — read/load
//   [Ch'en name]           — write/store
//   [Sek ⊗ A B]            — execute operation
//   [Xul]                  — sync/end
//   [K'ayab' ...]          — loop
//
// Operators: ⊗ (tensor_dot), ∫ (integrate), ∇ (gradient), ⊙ (attend), → (pipe)
//
// Output: KSON JSON (schema: https://kuhul.dev/kson/v1)

export class KuhulSyntaxError extends Error {
  constructor(message, line = 0, col = 0, context = '') {
    super(message);
    this.name    = 'KuhulSyntaxError';
    this.line    = line;
    this.col     = col;
    this.context = context;
  }
  toString() {
    const loc = this.line > 0 ? ` (Line ${this.line}${this.col > 0 ? `, Col ${this.col}` : ''})` : '';
    const ctx = this.context ? `\n  Context: ${this.context}` : '';
    return `${this.name}: ${this.message}${loc}${ctx}`;
  }
}

export class KuhulSemanticError extends Error {
  constructor(errors) {
    const list = Array.isArray(errors) ? errors : [errors];
    super(list.join('\n'));
    this.name   = 'KuhulSemanticError';
    this.errors = list;
  }
}

// Valid glyph names
export const KUHUL_GLYPHS = Object.freeze([
  'Pop', 'Yax', "Ch'en", 'Chen', 'Sek', 'Wo',
  "K'ayab'", "Kumk'u", 'Muwan', 'Xul', '∇', '∫', '⊙', '⊗',
]);

// Valid dtypes
export const KUHUL_DTYPES = Object.freeze(['float32', 'float64', 'int32', 'int8', 'uint32']);

// Operator glyph → KSON operation name
export const KUHUL_OP_MAP = Object.freeze({
  Yax:    'load',
  Chen:   'store',
  "Ch'en": 'store',
  Sek:    '⊗',
  Xul:    'sync',
  '∇':    'gradient',
  '∫':    'integrate',
  '⊙':    'attend',
  '⊗':    'tensor_dot',
});

// KSON phase sequence (π-based)
export const KSON_PHASES = Object.freeze(['load', 'π/4', 'π/2', '3π/4', 'π', 'store']);

// ── Phase 1: Lexer ──────────────────────────────────────────────────────────

export class KuhulLexer {
  constructor() {
    this.source = ''; this.pos = 0; this.line = 1; this.col = 1; this.tokens = [];
  }

  tokenize(source) {
    this.source = source; this.pos = 0; this.line = 1; this.col = 1; this.tokens = [];
    while (this.pos < this.source.length) {
      if (/\s/.test(this._cur())) { this._skipWhitespace(); continue; }
      if (this._cur() === ';') { this._skipComment(); continue; }

      const opMatch = this._tryMatch([
        { pattern: /^→/, type: 'OP_PIPE' },
        { pattern: /^⊗/, type: 'OP_TENSOR_DOT' },
        { pattern: /^∫/, type: 'OP_INTEGRAL' },
        { pattern: /^∇/, type: 'OP_GRADIENT' },
        { pattern: /^⊙/, type: 'OP_ATTEND' },
      ]);
      if (opMatch) continue;

      if (/[0-9]/.test(this._cur())) { this._scanNumber(); continue; }
      if (this._cur() === '"') { this._scanString(); continue; }
      if (/[a-zA-Z_]/.test(this._cur())) { this._scanIdentifier(); continue; }

      const singles = { '(': 'LPAREN', ')': 'RPAREN', '[': 'LBRACKET', ']': 'RBRACKET', ',': 'COMMA', '=': 'EQ' };
      if (singles[this._cur()]) {
        this.tokens.push({ type: singles[this._cur()], value: this._cur(), line: this.line, col: this.col });
        this._advance();
        continue;
      }
      throw new KuhulSyntaxError(`Unexpected character: '${this._cur()}'`, this.line, this.col, `at position ${this.pos}`);
    }
    this.tokens.push({ type: 'EOF', value: '', line: this.line, col: this.col });
    return this.tokens;
  }

  _tryMatch(patterns) {
    const rem = this.source.slice(this.pos);
    for (const p of patterns) {
      const m = rem.match(p.pattern);
      if (m) {
        this.tokens.push({ type: p.type, value: m[0], line: this.line, col: this.col });
        this._advance(m[0].length);
        return true;
      }
    }
    return false;
  }

  _scanNumber() {
    const c = this.col; let v = '';
    while (/[0-9]/.test(this._cur())) { v += this._cur(); this._advance(); }
    if (this._cur() === '.' && /[0-9]/.test(this.source[this.pos + 1])) {
      v += this._cur(); this._advance();
      while (/[0-9]/.test(this._cur())) { v += this._cur(); this._advance(); }
    }
    this.tokens.push({ type: 'NUMBER', value: v, line: this.line, col: c });
  }

  _scanString() {
    const c = this.col; this._advance(); let v = '';
    while (this._cur() !== '"' && this.pos < this.source.length) {
      if (this._cur() === '\\') { this._advance(); v += this._cur(); this._advance(); }
      else { v += this._cur(); this._advance(); }
    }
    if (this._cur() !== '"') throw new KuhulSyntaxError('Unterminated string', this.line, c);
    this._advance();
    this.tokens.push({ type: 'STRING', value: v, line: this.line, col: c });
  }

  _scanIdentifier() {
    const c = this.col; let v = '';
    while (/[a-zA-Z0-9_'']/.test(this._cur())) { v += this._cur(); this._advance(); }
    this.tokens.push({ type: 'IDENTIFIER', value: v, line: this.line, col: c });
  }

  _skipWhitespace() { while (/\s/.test(this._cur())) this._advance(); }
  _skipComment()    { while (this._cur() !== '\n' && this.pos < this.source.length) this._advance(); }
  _cur()            { return this.pos < this.source.length ? this.source[this.pos] : ''; }
  _advance(n = 1)   {
    for (let i = 0; i < n; i++) {
      if (this.source[this.pos] === '\n') { this.line++; this.col = 1; } else { this.col++; }
      this.pos++;
    }
  }
}

// ── Phase 2: Parser ─────────────────────────────────────────────────────────

const OP_TYPES = new Set(['OP_PIPE', 'OP_TENSOR_DOT', 'OP_INTEGRAL', 'OP_GRADIENT', 'OP_ATTEND']);

export class KuhulParser {
  constructor(tokens) { this.tokens = tokens; this.pos = 0; }

  parse() {
    const stmts = [];
    while (this._cur().type !== 'EOF') stmts.push(this._parseStatement());
    return { type: 'Program', statements: stmts };
  }

  _parseStatement() {
    if (this._cur().type !== 'LBRACKET')
      throw new KuhulSyntaxError(`Expected '[' got '${this._cur().value}'`, this._cur().line, this._cur().col);
    return this._parseGlyph();
  }

  _parseGlyph() {
    this._expect('LBRACKET');
    const glyphTok = this._cur(); const glyphType = glyphTok.value; this._advance();
    let name = ''; const args = []; const body = []; let hasArgs = false;

    while (this._cur().type !== 'RBRACKET' && this._cur().type !== 'EOF') {
      if (this._cur().type === 'LBRACKET' && this._peek().type === 'IDENTIFIER') {
        body.push(this._parseGlyph()); continue;
      }
      const nextType = this._peek().type;
      if (name === '' && !hasArgs && this._cur().type === 'IDENTIFIER'
          && !OP_TYPES.has(nextType) && nextType !== 'EQ') {
        name = this._cur().value; this._advance(); continue;
      }
      const arg = this._parseArg(); args.push(arg); hasArgs = true;
      if (this._cur().type === 'COMMA' && arg.key !== 'shape') this._advance();
    }
    this._expect('RBRACKET');
    return { type: 'Glyph', glyph: glyphType, glyphValue: glyphType, name, args, body, line: glyphTok.line, col: glyphTok.col };
  }

  _parseArg() {
    const tok = this._cur();
    if (tok.type === 'IDENTIFIER') {
      const key = tok.value; this._advance();
      if (this._cur().type === 'EQ') {
        this._advance(); const vt = this._cur(); let v = vt.value;
        if (vt.type === 'NUMBER') v = parseFloat(v);
        this._advance();
        if (key === 'shape' && this._cur().type === 'COMMA') {
          let t = String(v);
          while (this._cur().type === 'COMMA') {
            this._advance();
            if (this._cur().type === 'NUMBER') { t += ',' + this._cur().value; this._advance(); } else break;
          }
          v = t;
        }
        return { key, value: v };
      }
      return { value: key };
    }
    if (tok.type === 'NUMBER') { const v = parseFloat(tok.value); this._advance(); return { value: v }; }
    if (OP_TYPES.has(tok.type)) { const v = tok.value; this._advance(); return { value: v }; }
    throw new KuhulSyntaxError(`Unexpected token: '${tok.value}'`, tok.line, tok.col);
  }

  _cur()    { return this.tokens[this.pos] || { type: 'EOF', value: '', line: 0, col: 0 }; }
  _peek(n=1){ return this.tokens[this.pos + n] || { type: 'EOF', value: '', line: 0, col: 0 }; }
  _advance(){ if (this.pos < this.tokens.length) this.pos++; }
  _expect(t){ if (this._cur().type !== t) throw new KuhulSyntaxError(`Expected '${t}' got '${this._cur().value}'`, this._cur().line, this._cur().col); const tok = this._cur(); this._advance(); return tok; }
}

// ── Phase 3: Semantic Analyzer ──────────────────────────────────────────────

export class KuhulSemanticAnalyzer {
  constructor(ast) {
    this.ast = ast; this.symbolTable = new Map(); this.errors = [];
    this.validGlyphs    = new Set(KUHUL_GLYPHS);
    this.validDtypes    = new Set(KUHUL_DTYPES);
    this.validOperators = new Set(['⊗', '∫', '∇', '⊙', 'reduce', 'map']);
  }

  analyze() {
    for (const s of this.ast.statements) this._analyzeStatement(s);
    if (this.errors.length) throw new KuhulSemanticError(this.errors);
    return this.ast;
  }

  _analyzeStatement(s) {
    if (s.type !== 'Glyph') return;
    if (!this.validGlyphs.has(s.glyph)) { this._err(`Unknown glyph: ${s.glyph}`, s.line); return; }
    switch (s.glyph) {
      case 'Pop':   this._analyzeFunction(s); break;
      case 'Wo':    this._analyzeTensor(s);   break;
      case 'Yax':   this._analyzeRead(s);     break;
      case 'Chen':
      case "Ch'en": this._analyzeWrite(s);    break;
      case 'Sek':   this._analyzeExecute(s);  break;
    }
    for (const b of (s.body || [])) this._analyzeStatement(b);
  }

  _analyzeFunction(s) {
    if (!s.body?.length) this._err(`Function '${s.name}' must have a body`, s.line);
    this.symbolTable.set(s.name, { type: 'function', name: s.name, line: s.line });
  }

  _analyzeTensor(s) {
    const shape = this._arg(s, 'shape'); const dtype = this._arg(s, 'dtype') || 'float32';
    if (!shape) { this._err(`Tensor '${s.name}' missing 'shape'`, s.line); return; }
    if (!this.validDtypes.has(dtype)) this._err(`Invalid dtype: ${dtype}`, s.line);
    if (typeof shape !== 'string' || !/^\d+(\s*,\s*\d+)*$/.test(shape)) { this._err(`Invalid shape: ${shape}`, s.line); return; }
    for (const d of shape.split(',').map(v => parseInt(v.trim()))) {
      if (d <= 0) this._err(`Shape dims must be positive, got: ${d}`, s.line);
    }
    this.symbolTable.set(s.name, { type: 'tensor', name: s.name, shape: shape.split(',').map(v => parseInt(v.trim())), dtype, line: s.line });
  }

  _analyzeRead(s)    { if (!this.symbolTable.has(s.name)) this._err(`Undefined: ${s.name}`, s.line); }
  _analyzeWrite(s)   { if (!this.symbolTable.has(s.name)) this._err(`Undefined: ${s.name}`, s.line); }
  _analyzeExecute(s) {
    for (const a of s.args) {
      if (a.key && !this.symbolTable.has(a.key)) this._err(`Undefined: ${a.key}`, s.line);
    }
  }
  _arg(s, k) { const a = s.args.find(x => x.key === k); return a ? a.value : null; }
  _err(msg, line) { this.errors.push(`Line ${line}: ${msg}`); }
}

// ── Phase 4: KSON Generator ──────────────────────────────────────────────────

export class KSONGenerator {
  constructor(ast, manifest) {
    this.ast = ast;
    this.manifest = manifest || { name: 'kernel', type: 'compute_kernel', target: 'directx_12' };
    this.tensors = []; this.kernels = []; this.symbolTable = new Map(); this.opCounter = 0;
  }

  generate() {
    for (const s of this.ast.statements) {
      if (s.type !== 'Glyph') continue;
      if (s.glyph === 'Wo')  this._extractTensor(s);
      if (s.glyph === 'Pop') this._extractKernel(s);
    }
    return {
      $schema: 'https://kuhul.dev/kson/v1',
      version: '1.0.0',
      manifest: this.manifest,
      tensors: this.tensors,
      kernels: this.kernels,
      schedule: this._generateSchedule(),
    };
  }

  _extractTensor(s) {
    const shape = this._arg(s, 'shape'); const dtype = this._arg(s, 'dtype') || 'float32';
    const dims = shape ? shape.split(',').map(v => parseInt(v.trim())) : [];
    const t = { id: s.name, glyph: 'Wo', role: this._inferRole(), shape: dims, dtype };
    this.tensors.push(t); this.symbolTable.set(s.name, t);
  }

  _extractKernel(s) {
    this.kernels.push({ id: s.name, glyph: 'Pop', entry: `${s.name}_CS`, thread_group: [16, 16, 1], operations: this._extractOps(s.body || []) });
  }

  _extractOps(body) {
    const ops = []; let phaseIdx = 0;
    for (const s of body) {
      if (s.type !== 'Glyph') continue;
      ops.push({ id: `op${this.opCounter++}`, phase: KSON_PHASES[phaseIdx % KSON_PHASES.length], glyph: s.glyphValue, operation: KUHUL_OP_MAP[s.glyph] || 'unknown', inputs: this._inputs(s), output: s.name || null });
      phaseIdx++;
    }
    return ops;
  }

  _inferRole() { const i = this.tensors.length; return i === 0 ? 'input' : 'scratch'; }
  _inputs(s)   { return s.args.filter(a => a.key !== 'shape' && a.key !== 'dtype').map(a => a.value || a.key).filter(Boolean); }
  _arg(s, k)   { const a = s.args.find(x => x.key === k); return a ? a.value : null; }

  _generateSchedule() {
    const k = this.kernels[0];
    if (!k) return {};
    return { dispatch: { kernel: k.id, grid: [64, 64, 1], phase_gate: 'π/2' } };
  }

  static validateKSON(kson) {
    if (!kson.$schema)            throw new Error('KSON missing $schema');
    if (!kson.version)            throw new Error('KSON missing version');
    if (!kson.manifest?.name)     throw new Error('KSON manifest missing name');
    if (!Array.isArray(kson.tensors)) throw new Error('KSON tensors not an array');
    if (!Array.isArray(kson.kernels)) throw new Error('KSON kernels not an array');
    for (const t of kson.tensors) {
      if (!t.id || !t.glyph || !t.role || !Array.isArray(t.shape) || !t.dtype)
        throw new Error(`Invalid tensor: ${JSON.stringify(t)}`);
    }
    for (const k of kson.kernels) {
      if (!k.id || !k.glyph || !k.entry || !Array.isArray(k.thread_group) || !Array.isArray(k.operations))
        throw new Error(`Invalid kernel: ${JSON.stringify(k)}`);
    }
    return true;
  }
}

// ── Top-level compile function ───────────────────────────────────────────────

export function compileKUHUL(source, manifest = null) {
  const tokens = new KuhulLexer().tokenize(source);
  const ast    = new KuhulParser(tokens).parse();
  new KuhulSemanticAnalyzer(ast).analyze();
  const kson   = new KSONGenerator(ast, manifest).generate();
  KSONGenerator.validateKSON(kson);
  return kson;
}

// KUHUL shader artifact roles (KUHUL.v1.0.0 runtime contract)
export const KUHUL_SHADER_ROLES = Object.freeze({
  fold_storage: 'kuhul_fold_storage.cso',
  fold_compute: 'kuhul_fold_compute.cso',
  fold_meta:    'kuhul_fold_meta.cso',
  glyph_compute: 'glyph_compute.cso',
});

export const KUHUL_RUNTIME_CONTRACT = Object.freeze({
  name:    'kuhul-native',
  version: '1.0.0',
  api:     'd3d11',
  format:  'd3d-cso',
  shader_version: '1',
});
