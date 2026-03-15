/**
 * @fileoverview KUHUL source code formatter.
 *
 * Normalises KUHUL source code to a canonical style:
 *  - Each statement on its own line.
 *  - Consistent indentation (2 spaces per nesting level).
 *  - Single space after opening `[` and before closing `]`.
 *  - Blank line between top-level folds.
 *
 * @module kuhul/tools/formatter
 */

import { KuhulLexer }  from '../compiler/lexer.js';
import { KuhulParser } from '../compiler/parser.js';
import { NodeKind }    from '../compiler/parser.js';

// ------------------------------------------------------------------ //
// KuhulFormatter
// ------------------------------------------------------------------ //

/** Formats KUHUL source code to a canonical style. */
export class KuhulFormatter {
  /**
   * Format the given KUHUL source.
   *
   * @param {string} source
   * @returns {string} Formatted source code
   */
  format(source) {
    let ast;
    try {
      const tokens = new KuhulLexer().lex(source);
      ast = new KuhulParser().parse(tokens);
    } catch (err) {
      // If parsing fails, return source unchanged
      return source;
    }

    const lines = [];
    this._formatBody(ast.body, lines, 0);
    return lines.join('\n') + '\n';
  }

  // ---------------------------------------------------------------- //
  // Recursive formatter
  // ---------------------------------------------------------------- //

  /**
   * @param {object[]} body
   * @param {string[]} lines
   * @param {number}   indent
   */
  _formatBody(body, lines, indent) {
    for (let i = 0; i < body.length; i++) {
      const stmt = body[i];
      this._formatStatement(stmt, lines, indent);

      // Blank line after top-level folds for readability
      if (indent === 0 && stmt.kind === NodeKind.Fold && i < body.length - 1) {
        lines.push('');
      }
    }
  }

  /**
   * @param {object}   stmt
   * @param {string[]} lines
   * @param {number}   indent
   */
  _formatStatement(stmt, lines, indent) {
    const pad = '  '.repeat(indent);

    switch (stmt.kind) {
      case NodeKind.Fold:
        lines.push(`${pad}[Pop]`);
        this._formatBody(stmt.body, lines, indent + 1);
        lines.push(`${pad}[Xul]`);
        break;

      case NodeKind.Allocation: {
        const { elementType, shape } = stmt.tensorType;
        const typeStr = `tensor<${elementType}, [${shape.join(', ')}]>`;
        lines.push(`${pad}[Wo ${stmt.identifier.name} ${typeStr}]`);
        break;
      }

      case NodeKind.Read:
        lines.push(`${pad}[Yax ${stmt.identifier.name}]`);
        break;

      case NodeKind.Write:
        lines.push(`${pad}[Ch'en ${stmt.identifier.name} ${this._valStr(stmt.value)}]`);
        break;

      case NodeKind.Operation: {
        const ops = stmt.operands.map(o => this._valStr(o)).join(' ');
        lines.push(`${pad}[Sek ${stmt.glyph} ${ops}]`);
        break;
      }

      case NodeKind.PhaseCycle:
        lines.push(`${pad}[K'ayab']`);
        this._formatBody(stmt.body, lines, indent + 1);
        lines.push(`${pad}[Kumk'u]`);
        break;

      case NodeKind.Invocation: {
        const args = stmt.args.map(a => this._valStr(a)).join(' ');
        lines.push(`${pad}[Muwan ${stmt.identifier.name}${args ? ' ' + args : ''}]`);
        break;
      }

      default:
        lines.push(`${pad}; unknown statement: ${stmt.kind}`);
    }
  }

  _valStr(node) {
    if (!node) return '';
    if (node.kind === NodeKind.Identifier) return node.name;
    if (node.kind === NodeKind.NumberLit)  return String(node.value);
    if (node.kind === NodeKind.StringLit)  return `"${node.value}"`;
    return '?';
  }
}
