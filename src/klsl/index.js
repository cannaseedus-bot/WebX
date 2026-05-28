// KLSL compiler — K'UHUL Language Shading Language
// JavaScript port of klsl_compiler.h/cpp + klsl_opcodes.h from .gpu_trainer/klsl/
//
// Glyph syntax:
//   ⟁ shader <name>  — begin shader block
//   ⟁Xul⟁           — end shader block
//   ⟁Wo⟁  <payload> — declare (buffer, local var)
//   ⟁Sek⟁ <payload> — execute (statement / control flow)
//   ⟁Ch'en⟁ <name>  — store (comment in HLSL output)
//   ⟁Yax⟁  <name>   — load  (comment in HLSL output)
//   [Pop <name>]     — begin function
//   [Xul]            — end function
//   ⟁K'ayab'⟁ <head>— for-loop start:  var in start..end
//   ⟁Kumk'u⟁        — for-loop end

// ─── Opcode table (klsl_opcodes.h) ───────────────────────────────────────────
export const KLSL_OP = Object.freeze({
  NOP:          0x00,
  RET:          0xFF,
  JMP:          0xD0,
  JZ:           0xD1,
  JNZ:          0xD2,
  CALL:         0xD3,
  RET_V:        0xD4,
  LOAD_IMMU:    0x01,
  LOAD_CONST:   0x02,
  MOV:          0x03,
  ZERO:         0x04,
  IADD:         0x10,
  ISUB:         0x11,
  IMUL:         0x12,
  IAND:         0x13,
  IOR:          0x14,
  IXOR:         0x15,
  ISHL:         0x16,
  ISHR:         0x17,
  IMOD:         0x18,
  ILT:          0x19,
  IEQ:          0x1A,
  FADD:         0x20,
  FSUB:         0x21,
  FMUL:         0x22,
  FDIV:         0x23,
  FMAD:         0x24,  // r[a] = r[b]*r[c] + r[a] (fused)
  FNEG:         0x25,
  FABS:         0x26,
  FSQRT:        0x27,
  FMAX:         0x28,
  FMIN:         0x29,
  RELU:         0x2A,
  FLT:          0x2B,
  FEQ:          0x2C,
  FSIN:         0x30,
  FCOS:         0x31,
  FTAN:         0x32,
  FEXP:         0x33,
  FLOG:         0x34,
  FLOG2:        0x35,
  FPOW:         0x36,
  LOAD_SHARED:  0x40,
  STORE_SHARED: 0x41,
  LOAD_IN0:     0x42,
  LOAD_IN1:     0x43,
  LOAD_IN2:     0x44,
  STORE_OUT0:   0x45,
  STORE_OUT1:   0x46,
  ATOMIC_ADD:   0x50,
  ATOMIC_MAX:   0x51,
  ATOMIC_MIN:   0x52,
  ATOMIC_CAS:   0x53,
  THREAD_ID:    0x60,
  THREAD_CNT:   0x61,
  GROUP_ID:     0x62,
  LOCAL_ID:     0x63,
  BARRIER:      0x64,
  SPAWN:        0x65,
});

// ─── Glyph marker byte strings (U+27C1 = ⟁ → 0xE2 0x9F 0x81) ────────────────
const G_SHADER = '⟁ shader ';
const G_XUL    = '⟁Xul⟁';
const G_WO     = '⟁Wo⟁ ';
const G_SEK    = '⟁Sek⟁ ';
const G_CHEN   = "⟁Ch'en⟁ ";
const G_YAX    = '⟁Yax⟁ ';
const G_KAYAB  = "⟁K'ayab'⟁ ";
const G_KUMKU  = "⟁Kumk'u⟁";
const POP_PFX  = '[Pop ';
const XUL_KW   = '[Xul]';

const BUF_KINDS = new Set(['StructuredBuffer', 'RWStructuredBuffer', 'ConstantBuffer', 'ByteAddressBuffer']);

function trim(s) { return s.trim(); }
function after(s, pfx) { return s.slice(pfx.length).trimStart(); }
function sw(s, pfx) { return s.startsWith(pfx); }

function matchParen(s, idx) {
  if (s[idx] !== '(') return -1;
  let d = 1;
  for (let i = idx + 1; i < s.length; i++) {
    if (s[i] === '(') d++;
    else if (s[i] === ')') { if (--d === 0) return i; }
  }
  return -1;
}

function parseBufDecl(payload) {
  const lt = payload.indexOf('<');
  if (lt < 0) return null;
  const kw = trim(payload.slice(0, lt));
  if (!BUF_KINDS.has(kw)) return null;
  const gt = payload.indexOf('>', lt + 1);
  if (gt < 0) return null;
  const elemType = trim(payload.slice(lt + 1, gt));
  const rest = trim(payload.slice(gt + 1));
  const colon = rest.indexOf(':');
  const name = colon < 0 ? trim(rest) : trim(rest.slice(0, colon));
  let reg = '';
  if (colon >= 0) {
    const rp = trim(rest.slice(colon + 1));
    if (rp.startsWith('register(')) {
      const end = rp.indexOf(')', 9);
      reg = trim(rp.slice(9, end < 0 ? undefined : end));
    }
  }
  return { kind: kw, elemType, name, reg };
}

function parseLoop(payload) {
  const inIdx = payload.indexOf(' in ');
  if (inIdx < 0) return `for (/* KLSL: ${payload} */)`;
  const varPart = trim(payload.slice(0, inIdx));
  const rangePart = trim(payload.slice(inIdx + 4));
  const dd = rangePart.indexOf('..');
  if (dd < 0) return `for (/* KLSL: ${payload} */)`;
  const start = rangePart.slice(0, dd).trimEnd();
  const end_  = rangePart.slice(dd + 2).trimStart();
  const parts = varPart.split(/\s+/);
  const [type, varName] = parts.length >= 2 ? parts : ['uint', parts[0]];
  return `for (${type} ${varName} = ${start}; ${varName} < ${end_}; ++${varName})`;
}

function xlateSek(s) {
  s = trim(s);
  if (sw(s, 'return')) {
    const rest = trim(s.slice(6));
    return { line: rest ? `return ${rest};` : 'return;', opens: false };
  }
  if (sw(s, 'if ') || sw(s, 'if(')) {
    const op = s.indexOf('(');
    if (op >= 0) {
      const cp = matchParen(s, op);
      if (cp >= 0) {
        const cond = s.slice(0, cp + 1);
        const body = trim(s.slice(cp + 1));
        if (body) {
          const inner = xlateSek(body);
          return { line: `${cond} ${inner.line}`, opens: inner.opens };
        }
        return { line: `${cond} {`, opens: true };
      }
    }
    return { line: `${s} {`, opens: true };
  }
  if (sw(s, 'for ') || sw(s, 'for(') || sw(s, 'while ') || sw(s, 'while(')) {
    return { line: `${s} {`, opens: true };
  }
  if (s.endsWith(';')) s = s.slice(0, -1);
  return { line: `${s};`, opens: false };
}

function buildEntryParams(bodyLines) {
  let dtid = false, gtid = false, gid = false;
  for (const l of bodyLines) {
    if (l.includes('SV_DispatchThreadID')) dtid = true;
    if (l.includes('SV_GroupThreadID'))    gtid = true;
    if (l.includes('SV_GroupID'))          gid  = true;
  }
  const params = [];
  if (dtid) params.push('uint3 SV_DispatchThreadID : SV_DispatchThreadID');
  if (gtid) params.push('uint3 SV_GroupThreadID    : SV_GroupThreadID');
  if (gid)  params.push('uint3 SV_GroupID          : SV_GroupID');
  return params.join(',\n                ');
}

// ─── Compiler class ───────────────────────────────────────────────────────────

class Compiler {
  constructor(source, filename, opts) {
    this.lines    = source.split('\n');
    this.filename = filename;
    this.opts     = opts;
    this.meta     = { name: '', stage: 'compute', tx: 64, ty: 1, tz: 1 };
    this.buffers  = [];
    this.entryName = '';
    this.ok       = true;
    this.errLine  = 0;
    this.errMsg   = '';
    this.hlsl     = '';
    this.depth    = 0;
  }

  error(li, msg) {
    if (!this.ok) return;
    this.ok      = false;
    this.errLine = li + 1;
    this.errMsg  = `${this.filename}:${li + 1}: ${msg}`;
  }

  // Pass 1: collect metadata, buffers, entry point name
  pass1() {
    let inShader = false, inFunc = false;
    for (let li = 0; li < this.lines.length; li++) {
      const ln = trim(this.lines[li]);
      if (!ln || sw(ln, '//') || sw(ln, '/*')) continue;
      if (sw(ln, G_XUL)) { inShader = false; continue; }
      if (sw(ln, G_SHADER) && !inShader) {
        this.meta.name = after(ln, G_SHADER);
        inShader = true; continue;
      }
      if (!inShader) continue;

      if (!inFunc && sw(ln, G_WO)) {
        const pay = after(ln, G_WO);
        if (sw(pay, 'stage ')) {
          this.meta.stage = after(pay, 'stage ').replace(/"/g, '');
          continue;
        }
        if (sw(pay, 'threads ')) {
          const m = after(pay, 'threads ').match(/\[(\d+),\s*(\d+),\s*(\d+)\]/);
          if (m) { this.meta.tx = +m[1]; this.meta.ty = +m[2]; this.meta.tz = +m[3]; }
          continue;
        }
        const bd = parseBufDecl(pay);
        if (bd) { this.buffers.push(bd); continue; }
        continue;
      }

      if (sw(ln, POP_PFX)) {
        let name = ln.slice(POP_PFX.length);
        if (name.endsWith(']')) name = name.slice(0, -1);
        if (!this.entryName) this.entryName = trim(name);
        inFunc = true; continue;
      }
      if (sw(ln, XUL_KW)) { inFunc = false; continue; }
    }
  }

  _entryBody() {
    const body = [];
    let scanning = false;
    for (const raw of this.lines) {
      const ln = trim(raw);
      if (sw(ln, POP_PFX)) {
        let name = ln.slice(POP_PFX.length);
        if (name.endsWith(']')) name = name.slice(0, -1);
        scanning = (trim(name) === this.entryName);
        continue;
      }
      if (sw(ln, XUL_KW)) { scanning = false; continue; }
      if (scanning) body.push(ln);
    }
    return body;
  }

  // Pass 2: emit HLSL
  pass2() {
    const out = [];
    const ind = () => ' '.repeat(this.depth * 4);
    out.push('// Generated by KLSL compiler v0.1');
    out.push(`// Shader: ${this.meta.name}\n`);

    for (const bd of this.buffers) {
      const reg = bd.reg ? ` : register(${bd.reg})` : '';
      out.push(`${bd.kind}<${bd.elemType}> ${bd.name}${reg};`);
    }
    if (this.buffers.length) out.push('');

    const entryBody = this._entryBody();
    const params    = buildEntryParams(entryBody);
    const entryName = this.opts.entryOverride || this.entryName;

    let inShader = false, inFunc = false;

    for (let li = 0; li < this.lines.length; li++) {
      const raw = this.lines[li];
      const ln  = trim(raw);

      if (!ln) { out.push(''); continue; }
      if (sw(ln, '//') || sw(ln, '/*')) { out.push(ind() + ln); continue; }
      if (sw(ln, G_XUL)) { inShader = false; continue; }
      if (sw(ln, G_SHADER) && !inShader) { inShader = true; continue; }
      if (!inShader) continue;

      if (!inFunc && (sw(ln, G_WO) || sw(ln, G_SEK) || sw(ln, G_CHEN) || sw(ln, G_YAX))) continue;

      if (sw(ln, POP_PFX)) {
        let name = ln.slice(POP_PFX.length);
        if (name.endsWith(']')) name = name.slice(0, -1);
        name = trim(name);
        inFunc = true;
        const isEntry = name === entryName && this.meta.stage === 'compute';
        if (isEntry) {
          out.push(`[numthreads(${this.meta.tx}, ${this.meta.ty}, ${this.meta.tz})]`);
          out.push(`void ${name}(${params})\n{`);
        } else {
          out.push(`void ${name}()\n{`);
        }
        this.depth = 1;
        continue;
      }
      if (sw(ln, XUL_KW)) {
        if (inFunc) {
          while (this.depth > 1) { this.depth--; out.push(' '.repeat(this.depth * 4) + '}'); }
          out.push('}\n');
          this.depth = 0; inFunc = false;
        }
        continue;
      }
      if (!inFunc) continue;

      if (sw(ln, G_WO)) {
        let s = after(ln, G_WO).trimEnd();
        if (s.endsWith(';')) s = s.slice(0, -1);
        out.push(`${ind()}${s};`);
        continue;
      }
      if (sw(ln, G_SEK)) {
        const { line, opens } = xlateSek(after(ln, G_SEK));
        out.push(`${ind()}${line}`);
        if (opens) this.depth++;
        continue;
      }
      if (sw(ln, G_CHEN)) { out.push(`${ind()}// [→ ${after(ln, G_CHEN)}]`); continue; }
      if (sw(ln, G_YAX))  { out.push(`${ind()}// [← ${after(ln, G_YAX)}]`);  continue; }
      if (sw(ln, G_KAYAB)) {
        out.push(`${ind()}${parseLoop(after(ln, G_KAYAB))} {`);
        this.depth++;
        continue;
      }
      if (sw(ln, G_KUMKU)) {
        if (this.depth > 1) { this.depth--; out.push(ind() + '}'); }
        continue;
      }

      out.push(`${ind()}${ln}`);
    }

    this.hlsl = out.join('\n');
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function compile(source, filename = '<klsl>', options = {}) {
  const opts = {
    emitHLSL: true,
    emitXVM: false,
    emitWGSL: false,
    entryOverride: '',
    ...options,
  };
  const c = new Compiler(source, filename, opts);
  c.pass1();
  if (c.ok && opts.emitHLSL) c.pass2();
  return {
    ok:        c.ok,
    errorLine: c.errLine,
    errorMsg:  c.errMsg,
    hlsl:      c.hlsl,
    xvm:       null,  // XVM bytecode emitter is a future extension
    wgsl:      null,
  };
}

export const GLYPHS = Object.freeze({
  SHADER:  G_SHADER,
  XUL:     G_XUL,
  WO:      G_WO,
  SEK:     G_SEK,
  CHEN:    G_CHEN,
  YAX:     G_YAX,
  KAYAB:   G_KAYAB,
  KUMKU:   G_KUMKU,
  POP_PFX,
  XUL_KW,
});

export default compile;
