// CM-1 control gate dictionary (v1.0.0-PowerShell-LLM)
//
// CM-1 is SMCA layer 1 — the control manifold that gates MATRIX↔SCXQ7 authority transitions.
// The CM-1 dictionary maps control byte values used in the gated channel protocol.
// CM-1 lane type in XJSL: DICT.

export const CM1_ID   = 'cm1.dict.v1';
export const CM1_LANE = '@lane:DICT';

// Control byte values (decimal)
export const CM1_BYTES = Object.freeze({
  SOH: 1,   // Start of Header   — open authority context
  STX: 2,   // Start of Text     — begin payload / kernel body
  ETX: 3,   // End of Text       — end payload
  EOT: 4,   // End of Transmission — close authority context
  SO:  14,  // Shift Out         — activate alternate control plane
  SI:  15,  // Shift In          — return to default control plane
  RS:  30,  // Record Separator  — boundary between kernel records
});

// Reverse lookup: byte value → symbol name
export const CM1_BYTE_NAMES = Object.freeze(
  Object.fromEntries(Object.entries(CM1_BYTES).map(([k, v]) => [v, k]))
);

export function encodeCM1Frame(payload) {
  const bytes = typeof payload === 'string'
    ? new TextEncoder().encode(payload)
    : payload;
  const out = new Uint8Array(bytes.length + 4);
  out[0] = CM1_BYTES.SOH;
  out[1] = CM1_BYTES.STX;
  out.set(bytes, 2);
  out[2 + bytes.length]     = CM1_BYTES.ETX;
  out[2 + bytes.length + 1] = CM1_BYTES.EOT;
  return out;
}

export function decodeCM1Frame(frame) {
  if (frame[0] !== CM1_BYTES.SOH || frame[1] !== CM1_BYTES.STX) {
    throw new Error('CM-1: invalid frame header');
  }
  const etxIdx = frame.indexOf(CM1_BYTES.ETX, 2);
  if (etxIdx === -1) throw new Error('CM-1: missing ETX');
  return frame.slice(2, etxIdx);
}

export function recordSeparate(frames) {
  const parts = [];
  for (const f of frames) {
    parts.push(f);
    parts.push(new Uint8Array([CM1_BYTES.RS]));
  }
  if (parts.length) parts.pop(); // trailing RS omitted
  const total = parts.reduce((s, p) => s + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}
