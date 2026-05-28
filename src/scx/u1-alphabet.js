// U1 Unary Alphabet — port of SCX_UNARY_ALPHABET_U1.md (SCX.v1.0.0)
//
// U1 is a Unicode glyph stream where each codepoint represents one semantic atom.
// Version: U1.1.0
// Canonicalization: NFC normalize → codepoint sequence → hash
// Alphabet SHA256: 58d90630b6819f31da8a97e0ea7b97f667a92d1680de1b1fcbf484eb52c57310

export const U1_VERSION = 'U1.1.0';
export const U1_ALPHABET_SHA256 = '58d90630b6819f31da8a97e0ea7b97f667a92d1680de1b1fcbf484eb52c57310';

// Core glyph table — U1.1.0 baseline (8 glyphs)
export const U1_GLYPHS = Object.freeze({
  GLYPH_START:   { char: '⟁', codepoint: 0x27C1, name: 'GLYPH_START' },    // begins a glyph name
  CLUSTER_START: { char: '⟦', codepoint: 0x27E6, name: 'CLUSTER_START' },   // begins a cluster block
  CLUSTER_END:   { char: '⟧', codepoint: 0x27E7, name: 'CLUSTER_END' },     // ends glyph names + clusters
  MARKER:        { char: '⸬', codepoint: 0x2E2C, name: 'MARKER' },          // glyph marker separator
  REPEAT:        { char: '⨯', codepoint: 0x2A2F, name: 'REPEAT' },          // repeat multiplier
  LAMBDA:        { char: 'λ', codepoint: 0x03BB, name: 'LAMBDA' },           // lambda introducer
  PI:            { char: 'π', codepoint: 0x03C0, name: 'PI' },               // pi constant / pi-domain
  ARROW:         { char: '→', codepoint: 0x2192, name: 'ARROW' },            // rule/route arrow
});

// Reverse lookup: codepoint → glyph name
export const U1_BY_CODEPOINT = Object.freeze(
  Object.fromEntries(Object.values(U1_GLYPHS).map(g => [g.codepoint, g.name]))
);

// Default fold lattice for the ⟁F⟧ selector (11 top-level folds, version-stable)
export const U1_FOLD_LATTICE = Object.freeze([
  'micronauts', 'agents', 'skills', 'tools', 'commands',
  'files', 'threads', 'batches', 'processes', 'bots', 'ports',
]);

// 3-glyph workflow capsule format: ⟁W⟧ ⟁F⟧ ⟁P⟧
export function expandU1Capsule(verb, foldLattice, programId) {
  return Object.freeze({
    '@micronaut':   'workflow_micronaut',
    u1_version:     U1_VERSION,
    type:           'autonomous_agent',
    verb:           verb || 'exe',
    fold_lattice:   foldLattice || Array.from(U1_FOLD_LATTICE),
    program_id:     programId,
    inputs: {
      prompt:       '<user>',
      context_refs: [],
      constraints: {
        windows_explicit_paths:      true,
        dependent_releases_readonly: true,
        replay_required:             true,
      },
    },
    outputs: {
      deltas:      [],
      artifacts:   [],
      trace_hash:  '<hash-chain>',
    },
  });
}

// Canonical hash input for U1 stream verification
export function u1HashInput(normalizedCodepoints) {
  return `U1:${U1_VERSION}\nALPHABET_SHA256:${U1_ALPHABET_SHA256}\nCODEPOINTS:${normalizedCodepoints}\n`;
}

// Parse a U1 glyph name like ⟁W⟧ → 'W'
export function parseGlyphName(str, pos = 0) {
  const GLYPH_START = '⟁'; // ⟁
  const GLYPH_END   = '⟧'; // ⟧
  if (str[pos] !== GLYPH_START) return null;
  const end = str.indexOf(GLYPH_END, pos + 1);
  if (end === -1) return null;
  return { name: str.slice(pos + 1, end), end: end + 1 };
}

export const U1_SCX_GRAMMAR_VERSION = '3.0';
