// KUHUL 3D compiler — compiles K3D source to normalized IR JSON.
// Port of hive-runtime/Kuhul-PY/engine3d/compiler/kuhul3d_compiler.py
// from v0.2.0-kuhul-directx-native.
//
// IR schema:
//   { ir_version, source, dialect, ops[] }
//
// Op types:
//   { op: "command",  name, args[], raw }   — [Name arg arg ...]
//   { op: "declare",  kind, name, raw }     — Tensor|Cluster|Model|Buffer|Shader name = ...;
//   { op: "dx12_block", statements[] }       — dx12 { ... }
//   { op: "unknown",  raw }

const IR_VERSION = '0.1.0';
const DIALECT    = 'kuhul3d-v3-bootstrap';

const COMMAND_RE   = /^\[(.+?)\]\s*$/;
const DX12_START   = /^\s*dx12\s*\{\s*$/;
const DX12_END     = /^\s*\}\s*$/;
const DECL_RE      = /^\s*(Tensor|Cluster|Model|Buffer|Shader)\s+([A-Za-z_]\w*)\s*=.*;\s*$/;

function normalizeLine(line) {
  return line.split('//', 1)[0].trim();
}

function parseCommand(line) {
  const m = line.match(COMMAND_RE);
  if (!m) return null;
  const parts = m[1].trim().split(/\s+/);
  if (!parts.length) return null;
  return { op: 'command', name: parts[0], args: parts.slice(1), raw: line };
}

function parseDeclaration(line) {
  const m = line.match(DECL_RE);
  if (!m) return null;
  return { op: 'declare', kind: m[1], name: m[2], raw: line };
}

export function compileK3D(source, sourceName = '<k3d>') {
  const lines = source.split('\n');
  const ops   = [];
  let i = 0;

  while (i < lines.length) {
    const raw  = lines[i];
    const line = normalizeLine(raw);
    i++;

    if (!line) continue;

    if (DX12_START.test(line)) {
      const blockLines = [];
      let depth = 1;
      while (i < lines.length) {
        const raw2  = lines[i];
        const line2 = normalizeLine(raw2);
        i++;
        if (!line2) continue;
        depth += (line2.match(/\{/g) || []).length;
        depth -= (line2.match(/\}/g) || []).length;
        if (depth <= 0) break;
        if (line2) blockLines.push(line2);
      }
      ops.push({ op: 'dx12_block', statements: blockLines });
      continue;
    }

    const cmd = parseCommand(line);
    if (cmd) { ops.push(cmd); continue; }

    const decl = parseDeclaration(line);
    if (decl) { ops.push(decl); continue; }

    ops.push({ op: 'unknown', raw: line });
  }

  return { ir_version: IR_VERSION, source: sourceName, dialect: DIALECT, ops };
}

export default compileK3D;
