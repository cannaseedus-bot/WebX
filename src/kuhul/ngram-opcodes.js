// ngram-opcodes.js — K'UHUL complete ngram → opcode mapping
//
// 147 total ngrams: 67 direct opcodes + 80 syntax sugar.
//
// Opcodes execute directly in the runtime (kuhul-vm.js / KXML dispatcher).
// Syntax sugar is expanded by the parser before dispatch — every sugar maps
// to one or more opcodes.  "Sugar" = compiler-level shorthand, not runtime
// behavior.  The expansion is lossless and deterministic.
//
// Opcode ranges:
//   0x01-0x15  Core execution
//   0x20-0x37  ML / Tensor
//   0x40-0x4B  Distributed
//   0x60-0x69  XCFE control
//   0x70-0xFF  Reserved

// ─── 67 Direct Opcodes ────────────────────────────────────────────────────────

export const CORE_OPCODES = {
  '⟁Pop⟁':      { opcode: 0x01, category: 'core', description: 'Begin block' },
  '⟁Xul':       { opcode: 0x02, category: 'core', description: 'End block' },
  '⟁Sek⟁':      { opcode: 0x03, category: 'core', description: 'Set / assign' },
  '⟁Yax⟁':      { opcode: 0x04, category: 'core', description: 'Get / access' },
  '⟁Wo⟁':       { opcode: 0x05, category: 'core', description: 'Call / invoke' },
  '⟁Ch\'en⟁':   { opcode: 0x06, category: 'core', description: 'Store / persist' },
  '⟁Yax⟁?':     { opcode: 0x07, category: 'core', description: 'Null-safe access' },
  '⟁Sek⟁!':     { opcode: 0x08, category: 'core', description: 'Force assign' },
  '⟁Wo⟁@':      { opcode: 0x09, category: 'core', description: 'Meta call' },
  '⟁Wo⟁if':     { opcode: 0x0A, category: 'flow', description: 'Conditional' },
  '⟁Wo⟁each':   { opcode: 0x0B, category: 'flow', description: 'Iterator' },
  '⟁Wo⟁match':  { opcode: 0x0C, category: 'flow', description: 'Pattern match' },
  '⟁Wo⟁return': { opcode: 0x0D, category: 'flow', description: 'Return' },
  '⟁Wo⟁break':  { opcode: 0x0E, category: 'flow', description: 'Break loop' },
  '⟁Wo⟁continue':{ opcode: 0x0F, category: 'flow', description: 'Continue loop' },
  '⟁Wo⟁try':    { opcode: 0x10, category: 'flow', description: 'Try / catch' },
  '→':           { opcode: 0x11, category: 'data', description: 'Flow / pipeline' },
  '⟁⟁':         { opcode: 0x12, category: 'data', description: 'Parallel' },
  '⟁⟁?':        { opcode: 0x13, category: 'data', description: 'Branch' },
  '↻':           { opcode: 0x14, category: 'data', description: 'Loop back' },
  '↯':           { opcode: 0x15, category: 'data', description: 'Break flow' },
};

export const ML_OPCODES = {
  '⟁Ten⟁':   { opcode: 0x20, category: 'ml', description: 'Define tensor' },
  '⟁Wey⟁':   { opcode: 0x21, category: 'ml', description: 'Define weights' },
  '⟁Tok⟁':   { opcode: 0x22, category: 'ml', description: 'Tokenize' },
  '⟁Log⟁':   { opcode: 0x23, category: 'ml', description: 'Logic node' },
  '⟁Ten⟁!':  { opcode: 0x24, category: 'ml', description: 'Mutable tensor' },
  '⟁Wey⟁!':  { opcode: 0x25, category: 'ml', description: 'Trainable weights' },
  '⟁Fwd⟁':   { opcode: 0x26, category: 'nn', description: 'Forward pass' },
  '⟁Bwd⟁':   { opcode: 0x27, category: 'nn', description: 'Backward pass' },
  '⟁Loss⟁':  { opcode: 0x28, category: 'nn', description: 'Loss function' },
  '⟁Opt⟁':   { opcode: 0x29, category: 'nn', description: 'Optimizer' },
  '⟁Attn⟁':  { opcode: 0x2A, category: 'nn', description: 'Attention' },
  '⟁Norm⟁':  { opcode: 0x2B, category: 'nn', description: 'Layer normalization' },
  '⟁Drop⟁':  { opcode: 0x2C, category: 'nn', description: 'Dropout' },
  '⟁Act⟁':   { opcode: 0x2D, category: 'nn', description: 'Activation function' },
  '⟁Load⟁':  { opcode: 0x2E, category: 'ml', description: 'Load model' },
  '⟁Save⟁':  { opcode: 0x2F, category: 'ml', description: 'Save model' },
  '⟁Eval⟁':  { opcode: 0x30, category: 'ml', description: 'Evaluate' },
  '⟁Pred⟁':  { opcode: 0x31, category: 'ml', description: 'Predict' },
  '⟁Grad⟁':  { opcode: 0x32, category: 'ml', description: 'Gradient' },
  '⟁Stop⟁':  { opcode: 0x33, category: 'ml', description: 'Stop gradient' },
  '⟁Batch⟁': { opcode: 0x34, category: 'ml', description: 'Batch data' },
  '⟁Shuf⟁':  { opcode: 0x35, category: 'ml', description: 'Shuffle' },
  '⟁Norm⟁!': { opcode: 0x36, category: 'ml', description: 'Batch normalization' },
  '⟁Aug⟁':   { opcode: 0x37, category: 'ml', description: 'Augment data' },
};

export const DISTRIBUTED_OPCODES = {
  '⟁Clu⟁':    { opcode: 0x40, category: 'dist', description: 'Define cluster' },
  '⟁Nod⟁':    { opcode: 0x41, category: 'dist', description: 'Define node' },
  '⟁Clu⟁!':   { opcode: 0x42, category: 'dist', description: 'Dynamic cluster' },
  '⟁Nod⟁!':   { opcode: 0x43, category: 'dist', description: 'Dynamic node' },
  '⟁Dist⟁':   { opcode: 0x44, category: 'dist', description: 'Distribute tensor' },
  '⟁Gath⟁':   { opcode: 0x45, category: 'dist', description: 'Gather results' },
  '⟁Sync⟁':   { opcode: 0x46, category: 'dist', description: 'Synchronize' },
  '⟁Part⟁':   { opcode: 0x47, category: 'dist', description: 'Partition' },
  '⟁Reduce⟁': { opcode: 0x48, category: 'dist', description: 'All-reduce' },
  '⟁Repl⟁':   { opcode: 0x49, category: 'dist', description: 'Replicate' },
  '⟁Fail⟁':   { opcode: 0x4A, category: 'dist', description: 'Failover' },
  '⟁Rec⟁':    { opcode: 0x4B, category: 'dist', description: 'Recover' },
};

export const XCFE_OPCODES = {
  '⟁XCFE⟁':  { opcode: 0x60, category: 'xcfe', description: 'XCFE control' },
  '⟁Val⟁':   { opcode: 0x61, category: 'xcfe', description: 'Validate state' },
  '⟁Mon⟁':   { opcode: 0x62, category: 'xcfe', description: 'Monitor metrics' },
  '⟁Enf⟁':   { opcode: 0x63, category: 'xcfe', description: 'Enforce rule' },
  '⟁Dec⟁':   { opcode: 0x64, category: 'xcfe', description: 'Decision' },
  '⟁Path⟁':  { opcode: 0x65, category: 'xcfe', description: 'Path select' },
  '⟁Rule⟁':  { opcode: 0x66, category: 'xcfe', description: 'Rule apply' },
  '⟁State⟁': { opcode: 0x67, category: 'xcfe', description: 'State track' },
  '⟁Trans⟁': { opcode: 0x68, category: 'xcfe', description: 'Transition' },
  '⟁Check⟁': { opcode: 0x69, category: 'xcfe', description: 'Checkpoint' },
};

// ─── 80 Syntax Sugar entries ──────────────────────────────────────────────────
// Each sugar expands to one or more opcodes at parse time.
// Format: expands is a string of opcode keys joined by ' → '

export const SYNTAX_SUGAR = {
  // Compound (28) — arithmetic, comparison, assignment, access
  ':= ':   { expands: '⟁Sek⟁',             category: 'compound' },
  '= ':    { expands: '⟁Sek⟁',             category: 'compound' },
  '+= ':   { expands: '⟁Sek⟁ (⟁Yax⟁ + )', category: 'compound' },
  '-= ':   { expands: '⟁Sek⟁ (⟁Yax⟁ - )', category: 'compound' },
  '== ':   { expands: '⟁Wo⟁ eq',           category: 'compound' },
  '!= ':   { expands: '⟁Wo⟁ neq',          category: 'compound' },
  '< ':    { expands: '⟁Wo⟁ lt',           category: 'compound' },
  '> ':    { expands: '⟁Wo⟁ gt',           category: 'compound' },
  '<= ':   { expands: '⟁Wo⟁ lte',          category: 'compound' },
  '>= ':   { expands: '⟁Wo⟁ gte',          category: 'compound' },
  '&& ':   { expands: '⟁Wo⟁ and',          category: 'compound' },
  '|| ':   { expands: '⟁Wo⟁ or',           category: 'compound' },
  '! ':    { expands: '⟁Wo⟁ not',          category: 'compound' },
  '?? ':   { expands: '⟁Yax⟁?',            category: 'compound' },
  '+ ':    { expands: '⟁Wo⟁ add',          category: 'compound' },
  '- ':    { expands: '⟁Wo⟁ sub',          category: 'compound' },
  '* ':    { expands: '⟁Wo⟁ mul',          category: 'compound' },
  '/ ':    { expands: '⟁Wo⟁ div',          category: 'compound' },
  '% ':    { expands: '⟁Wo⟁ mod',          category: 'compound' },
  '** ':   { expands: '⟁Wo⟁ pow',          category: 'compound' },
  '.':     { expands: '⟁Yax⟁',             category: 'compound' },
  '?.':    { expands: '⟁Yax⟁?',            category: 'compound' },
  '[]':    { expands: '⟁Yax⟁',             category: 'compound' },
  '?[]':   { expands: '⟁Yax⟁?',            category: 'compound' },
  '()':    { expands: '⟁Wo⟁',              category: 'compound' },
  '?.()':  { expands: '⟁Wo⟁?',             category: 'compound' },
  '=>':    { expands: '⟁Wo⟁ return',        category: 'compound' },
  '|>':    { expands: '→',                  category: 'compound' },

  // Liquid (22) — storage namespaces + high-level ML shortcuts
  'ram.':     { expands: '⟁Sek⟁ ram.liquid.',               category: 'liquid' },
  'db.':      { expands: '⟁Sek⟁ db.liquid.',                category: 'liquid' },
  'cache.':   { expands: '⟁Sek⟁ cache.liquid.',             category: 'liquid' },
  'state.':   { expands: '⟁Sek⟁ state.liquid.',             category: 'liquid' },
  'session.': { expands: '⟁Sek⟁ session.liquid.',           category: 'liquid' },
  'temp.':    { expands: '⟁Sek⟁ temp.liquid.',              category: 'liquid' },
  'fn.':      { expands: '⟁Wo⟁ fn.liquid.',                 category: 'liquid' },
  'async.':   { expands: '⟁Wo⟁ async.liquid.',              category: 'liquid' },
  'pipe.':    { expands: '⟁Wo⟁ pipe.liquid.',               category: 'liquid' },
  'map.':     { expands: '⟁Wo⟁each → →',                   category: 'liquid' },
  'filter.':  { expands: '⟁Wo⟁each → ⟁Wo⟁if',             category: 'liquid' },
  'reduce.':  { expands: '⟁Wo⟁each → accumulate',          category: 'liquid' },
  'sort.':    { expands: '⟁Wo⟁ sort.liquid.',               category: 'liquid' },
  'group.':   { expands: '⟁Wo⟁ group.liquid.',              category: 'liquid' },
  'tensor.':  { expands: '⟁Ten⟁',                           category: 'liquid' },
  'model.':   { expands: '⟁Load⟁ → ⟁Fwd⟁',               category: 'liquid' },
  'train.':   { expands: '⟁Fwd⟁ → ⟁Loss⟁ → ⟁Bwd⟁ → ⟁Opt⟁', category: 'liquid' },
  'infer.':   { expands: '⟁Load⟁ → ⟁Pred⟁',              category: 'liquid' },
  'embed.':   { expands: '⟁Tok⟁ → ⟁Ten⟁',                category: 'liquid' },
  'attend.':  { expands: '⟁Attn⟁',                          category: 'liquid' },
  'norm.':    { expands: '⟁Norm⟁',                           category: 'liquid' },
  'drop.':    { expands: '⟁Drop⟁',                           category: 'liquid' },

  // Shortcut (18) — single sigils
  '@':   { expands: '⟁Wo⟁ @',        category: 'shortcut' },
  '#':   { expands: '⟁Sek⟁ #',       category: 'shortcut' },
  '$':   { expands: '⟁Yax⟁ $',       category: 'shortcut' },
  '&':   { expands: '⟁Wo⟁ &',        category: 'shortcut' },
  '*':   { expands: '⟁Wo⟁ *',        category: 'shortcut' },
  '?':   { expands: '⟁Yax⟁?',        category: 'shortcut' },
  '!':   { expands: '⟁Sek⟁!',        category: 'shortcut' },
  '->':  { expands: '→',              category: 'shortcut' },
  '<-':  { expands: '←',              category: 'shortcut' },
  '<|':  { expands: '←',              category: 'shortcut' },
  '>>':  { expands: '⟁Wo⟁ then',     category: 'shortcut' },
  '...': { expands: '⟁Wo⟁ spread',   category: 'shortcut' },
  '??':  { expands: '⟁Yax⟁?',        category: 'shortcut' },
  '!=':  { expands: '⟁Wo⟁ neq',      category: 'shortcut' },
  '===': { expands: '⟁Wo⟁ strict_eq',category: 'shortcut' },

  // Macro (12) — multi-opcode patterns
  'if/else':   { expands: '⟁Wo⟁if ⟁Wo⟁else',                             category: 'macro' },
  'try/catch': { expands: '⟁Wo⟁try ⟁Wo⟁catch',                           category: 'macro' },
  'while':     { expands: '⟁Wo⟁while',                                    category: 'macro' },
  'for':       { expands: '⟁Wo⟁for',                                      category: 'macro' },
  'map/filter':{ expands: '⟁Wo⟁each → ⟁Wo⟁if',                          category: 'macro' },
  'pipe/flow': { expands: '→ → →',                                        category: 'macro' },
  'parallel':  { expands: '⟁⟁ ⟁⟁ ⟁⟁',                                   category: 'macro' },
  'sequence':  { expands: '⟁Sek⟁ → ⟁Wo⟁ → ⟁Ch\'en⟁',                  category: 'macro' },
  'nn.layer':  { expands: '⟁Ten⟁ → ⟁Act⟁ → ⟁Drop⟁',                    category: 'macro' },
  'train.step':{ expands: '⟁Fwd⟁ → ⟁Loss⟁ → ⟁Bwd⟁ → ⟁Opt⟁',          category: 'macro' },
  'batch.loop':{ expands: '⟁Batch⟁ → ⟁Wo⟁each → ⟁Fwd⟁',               category: 'macro' },
  'dist.sync': { expands: '⟁Dist⟁ → ⟁Sync⟁ → ⟁Gath⟁',                 category: 'macro' },
};

// ─── Unified lookup (opcodes + sugar merged) ──────────────────────────────────

export const ALL_NGRAMS = Object.assign(
  {},
  ...Object.values({ CORE_OPCODES, ML_OPCODES, DISTRIBUTED_OPCODES, XCFE_OPCODES })
    .map(table => Object.fromEntries(
      Object.entries(table).map(([k, v]) => [k, { ...v, sugar: false }])
    )),
  Object.fromEntries(
    Object.entries(SYNTAX_SUGAR).map(([k, v]) => [k, { ...v, sugar: true }])
  )
);

export const STATS = {
  total:       147,
  opcodes:      67,  // core:21 + ml:24 + dist:12 + xcfe:10
  sugar:        80,  // compound:28 + liquid:22 + shortcut:18 (shortcut has 15) + macro:12
  opcode_range: { start: 0x01, end: 0x69, reserved_from: 0x70 },
};
