/**
 * @fileoverview EBNF grammar parser for KUHUL.
 *
 * Parses an EBNF source string into a structured representation consisting of
 * named rules and their right-hand-side expressions.
 *
 * Grammar accepted by this parser:
 *   grammar   = { rule } EOF
 *   rule      = identifier "=" expression ";"
 *   expression = alternative { "|" alternative }
 *   alternative = term { term }
 *   term      = atom [ "*" | "+" | "?" ]
 *   atom      = "(" expression ")"
 *              | "{" expression "}"   (* zero-or-more *)
 *              | "[" expression "]"   (* optional      *)
 *              | string_literal
 *              | identifier
 *
 * Comments (* … *) are stripped before parsing.
 */

// ------------------------------------------------------------------ //
// AST node helpers
// ------------------------------------------------------------------ //

/**
 * @typedef {{ kind: string }} BaseNode
 * @typedef {{ kind: 'Seq',     items: ExprNode[] }} SeqNode
 * @typedef {{ kind: 'Alt',     items: ExprNode[] }} AltNode
 * @typedef {{ kind: 'Rep',     expr:  ExprNode   }} RepNode
 * @typedef {{ kind: 'Opt',     expr:  ExprNode   }} OptNode
 * @typedef {{ kind: 'Plus',    expr:  ExprNode   }} PlusNode
 * @typedef {{ kind: 'Ref',     name:  string     }} RefNode
 * @typedef {{ kind: 'Literal', value: string     }} LiteralNode
 * @typedef {SeqNode|AltNode|RepNode|OptNode|PlusNode|RefNode|LiteralNode} ExprNode
 */

/** @param {ExprNode[]} items @returns {SeqNode|ExprNode} */
function seq(items) {
  return items.length === 1 ? items[0] : { kind: 'Seq', items };
}
/** @param {ExprNode[]} items @returns {AltNode|ExprNode} */
function alt(items) {
  return items.length === 1 ? items[0] : { kind: 'Alt', items };
}
/** @param {ExprNode} expr @returns {RepNode} */
const rep  = (expr) => ({ kind: 'Rep',  expr });
/** @param {ExprNode} expr @returns {OptNode} */
const opt  = (expr) => ({ kind: 'Opt',  expr });
/** @param {ExprNode} expr @returns {PlusNode} */
const plus = (expr) => ({ kind: 'Plus', expr });
/** @param {string} name @returns {RefNode} */
const ref  = (name) => ({ kind: 'Ref',  name });
/** @param {string} value @returns {LiteralNode} */
const lit  = (value) => ({ kind: 'Literal', value });

// ------------------------------------------------------------------ //
// Tokeniser (for the EBNF meta-language, not for KUHUL itself)
// ------------------------------------------------------------------ //

const META_TT = Object.freeze({
  IDENT:   'IDENT',
  STRING:  'STRING',
  EQUALS:  'EQUALS',
  SEMI:    'SEMI',
  PIPE:    'PIPE',
  LPAREN:  'LPAREN',
  RPAREN:  'RPAREN',
  LBRACE:  'LBRACE',
  RBRACE:  'RBRACE',
  LBRACK:  'LBRACK',
  RBRACK:  'RBRACK',
  STAR:    'STAR',
  PLUS:    'PLUS',
  QUEST:   'QUEST',
  EOF:     'EOF',
});

/**
 * @param {string} src
 * @returns {{ type: string, value: string, pos: number }[]}
 */
function metaTokenize(src) {
  // Strip block comments (* … *)
  src = src.replace(/\(\*[\s\S]*?\*\)/g, ' ');

  const tokens = [];
  let i = 0;

  while (i < src.length) {
    // Skip whitespace
    if (/\s/.test(src[i])) { i++; continue; }

    const ch = src[i];

    if (ch === '=') { tokens.push({ type: META_TT.EQUALS, value: '=', pos: i }); i++; continue; }
    if (ch === ';') { tokens.push({ type: META_TT.SEMI,   value: ';', pos: i }); i++; continue; }
    if (ch === '|') { tokens.push({ type: META_TT.PIPE,   value: '|', pos: i }); i++; continue; }
    if (ch === '(') { tokens.push({ type: META_TT.LPAREN, value: '(', pos: i }); i++; continue; }
    if (ch === ')') { tokens.push({ type: META_TT.RPAREN, value: ')', pos: i }); i++; continue; }
    if (ch === '{') { tokens.push({ type: META_TT.LBRACE, value: '{', pos: i }); i++; continue; }
    if (ch === '}') { tokens.push({ type: META_TT.RBRACE, value: '}', pos: i }); i++; continue; }
    if (ch === '[') { tokens.push({ type: META_TT.LBRACK, value: '[', pos: i }); i++; continue; }
    if (ch === ']') { tokens.push({ type: META_TT.RBRACK, value: ']', pos: i }); i++; continue; }
    if (ch === '*') { tokens.push({ type: META_TT.STAR,   value: '*', pos: i }); i++; continue; }
    if (ch === '+') { tokens.push({ type: META_TT.PLUS,   value: '+', pos: i }); i++; continue; }
    if (ch === '?') { tokens.push({ type: META_TT.QUEST,  value: '?', pos: i }); i++; continue; }

    // String literals: "…" or '…'
    if (ch === '"' || ch === "'") {
      const q = ch; let j = i + 1;
      while (j < src.length && src[j] !== q) j++;
      tokens.push({ type: META_TT.STRING, value: src.slice(i + 1, j), pos: i });
      i = j + 1;
      continue;
    }

    // Identifiers (may include apostrophes for K'ayab' etc.)
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < src.length && /[\w']/.test(src[j])) j++;
      tokens.push({ type: META_TT.IDENT, value: src.slice(i, j), pos: i });
      i = j;
      continue;
    }

    // Skip unrecognised characters
    i++;
  }

  tokens.push({ type: META_TT.EOF, value: '', pos: i });
  return tokens;
}

// ------------------------------------------------------------------ //
// Recursive-descent parser for EBNF
// ------------------------------------------------------------------ //

/**
 * @param {{ type: string, value: string }[]} tokens
 * @returns {{ rules: Map<string, ExprNode>, start: string }}
 */
function parseTokens(tokens) {
  let pos = 0;

  const peek = () => tokens[pos];
  const consume = (type) => {
    const t = tokens[pos];
    if (type && t.type !== type) {
      throw new Error(`EBNF parse error: expected ${type} but got ${t.type} ("${t.value}") at position ${t.pos}`);
    }
    pos++;
    return t;
  };

  /** expression = alternative { "|" alternative } */
  function parseExpr() {
    const alts = [parseAlt()];
    while (peek().type === META_TT.PIPE) {
      consume(META_TT.PIPE);
      alts.push(parseAlt());
    }
    return alt(alts);
  }

  /** alternative = term { term } */
  function parseAlt() {
    const items = [];
    while (
      peek().type !== META_TT.PIPE &&
      peek().type !== META_TT.SEMI &&
      peek().type !== META_TT.RPAREN &&
      peek().type !== META_TT.RBRACE &&
      peek().type !== META_TT.RBRACK &&
      peek().type !== META_TT.EOF
    ) {
      items.push(parseTerm());
    }
    // Return a sequence of the collected items; an empty items array
    // represents the epsilon (empty) alternative in EBNF.
    return seq(items);
  }

  /** term = atom [ quantifier ] */
  function parseTerm() {
    let node = parseAtom();
    if (peek().type === META_TT.STAR)       { consume(); node = rep(node);  }
    else if (peek().type === META_TT.PLUS)  { consume(); node = plus(node); }
    else if (peek().type === META_TT.QUEST) { consume(); node = opt(node);  }
    return node;
  }

  /** atom = "(" expr ")" | "{" expr "}" | "[" expr "]" | string | ident */
  function parseAtom() {
    const t = peek();
    if (t.type === META_TT.LPAREN) {
      consume(META_TT.LPAREN);
      const e = parseExpr();
      consume(META_TT.RPAREN);
      return e;
    }
    if (t.type === META_TT.LBRACE) {
      consume(META_TT.LBRACE);
      const e = parseExpr();
      consume(META_TT.RBRACE);
      return rep(e);
    }
    if (t.type === META_TT.LBRACK) {
      consume(META_TT.LBRACK);
      const e = parseExpr();
      consume(META_TT.RBRACK);
      return opt(e);
    }
    if (t.type === META_TT.STRING) {
      consume();
      return lit(t.value);
    }
    if (t.type === META_TT.IDENT) {
      consume();
      return ref(t.value);
    }
    throw new Error(`EBNF parse error: unexpected token ${t.type} ("${t.value}") at position ${t.pos}`);
  }

  const rules = new Map();
  let start = null;

  while (peek().type !== META_TT.EOF) {
    const name = consume(META_TT.IDENT).value;
    consume(META_TT.EQUALS);
    const expr = parseExpr();
    consume(META_TT.SEMI);
    rules.set(name, expr);
    if (start === null) start = name;
  }

  return { rules, start };
}

// ------------------------------------------------------------------ //
// Public API
// ------------------------------------------------------------------ //

/**
 * Parse an EBNF source string into a structured grammar representation.
 *
 * @param {string} source - Raw EBNF source text
 * @returns {{ rules: Map<string, ExprNode>, start: string }}
 */
export function parseEBNF(source) {
  const tokens = metaTokenize(source);
  return parseTokens(tokens);
}
