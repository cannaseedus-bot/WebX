/**
 * @fileoverview IR verifier for the KUHUL Geometric IR.
 *
 * Checks that a GeometricIR object is well-formed before code generation:
 *  - All operand identifiers referenced in OP/READ/WRITE/CALL instructions
 *    are either declared via ALLOC or are numeric/string literals.
 *  - FOLD_START / FOLD_END and PHASE_START / PHASE_END are balanced.
 *  - Each ALLOC has a valid element type and non-empty shape.
 *
 * @module kuhul/ir/ir-verifier
 */

import { GeometricIR } from './ir-types.js';

const VALID_ELEMENT_TYPES = new Set([
  'float32', 'float64', 'int32', 'int64', 'uint8', 'bool', 'complex64',
]);

// ------------------------------------------------------------------ //
// IRVerifier
// ------------------------------------------------------------------ //

/** Verifies a GeometricIR program for correctness. */
export class IRVerifier {
  /**
   * Verify the given IR.
   *
   * @param {GeometricIR} ir
   * @returns {{ valid: boolean, errors: string[] }}
   */
  verify(ir) {
    const errors = [];

    if (!(ir instanceof GeometricIR)) {
      return { valid: false, errors: ['Input is not a GeometricIR instance.'] };
    }

    const declared = new Set(ir.symbolTable.keys());
    let foldDepth  = 0;
    let phaseDepth = 0;

    for (let idx = 0; idx < ir.instructions.length; idx++) {
      const instr = ir.instructions[idx];
      const at    = `instruction[${idx}] ${instr.opcode}`;

      switch (instr.opcode) {
        case 'ALLOC': {
          const [name, elementType, shape] = instr.operands;
          if (!name) { errors.push(`${at}: missing name operand`); break; }
          if (!VALID_ELEMENT_TYPES.has(elementType)) {
            errors.push(`${at}: invalid element type "${elementType}"`);
          }
          if (!Array.isArray(shape) || shape.length === 0) {
            errors.push(`${at}: shape must be a non-empty array`);
          }
          declared.add(name);
          break;
        }

        case 'READ': {
          const [name] = instr.operands;
          if (!declared.has(name)) {
            errors.push(`${at}: identifier "${name}" is not declared`);
          }
          break;
        }

        case 'WRITE': {
          const [target, valueRef] = instr.operands;
          if (!target) errors.push(`${at}: missing target operand`);
          if (typeof valueRef === 'string' && valueRef && !valueRef.startsWith('"') && !declared.has(valueRef)) {
            errors.push(`${at}: value reference "${valueRef}" is not declared`);
          }
          declared.add(target); // writing creates the symbol if absent
          break;
        }

        case 'OP': {
          const [glyph, ...operands] = instr.operands;
          if (!glyph) { errors.push(`${at}: missing glyph operand`); break; }
          for (const op of operands) {
            if (typeof op === 'string' && !op.startsWith('"') && !declared.has(op)) {
              errors.push(`${at}: operand "${op}" is not declared`);
            }
          }
          break;
        }

        case 'CALL': {
          const [fnName, ...args] = instr.operands;
          if (!fnName) { errors.push(`${at}: missing function name`); break; }
          for (const arg of args) {
            if (typeof arg === 'string' && !arg.startsWith('"') && !declared.has(arg)) {
              errors.push(`${at}: argument "${arg}" is not declared`);
            }
          }
          break;
        }

        case 'FOLD_START':  foldDepth++;  break;
        case 'FOLD_END':
          if (foldDepth === 0) errors.push(`${at}: unmatched FOLD_END`);
          else foldDepth--;
          break;

        case 'PHASE_START': phaseDepth++; break;
        case 'PHASE_END':
          if (phaseDepth === 0) errors.push(`${at}: unmatched PHASE_END`);
          else phaseDepth--;
          break;

        case 'CONST':
          // Always valid
          break;

        default:
          errors.push(`${at}: unknown opcode "${instr.opcode}"`);
      }
    }

    if (foldDepth !== 0)  errors.push(`Unbalanced FOLD_START/FOLD_END (depth ${foldDepth})`);
    if (phaseDepth !== 0) errors.push(`Unbalanced PHASE_START/PHASE_END (depth ${phaseDepth})`);

    return { valid: errors.length === 0, errors };
  }
}
