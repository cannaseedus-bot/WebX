/**
 * @fileoverview IR type definitions for the KUHUL Geometric IR.
 *
 * The Geometric IR is a simple, typed, sequential instruction set that sits
 * between the AST and the final code generators.
 *
 * @module kuhul/ir/ir-types
 */

// ------------------------------------------------------------------ //
// TensorType
// ------------------------------------------------------------------ //

/**
 * Represents the type of a KUHUL tensor value.
 */
export class TensorType {
  /**
   * @param {string} elementType - e.g. "float32"
   * @param {(number|'?')[]} shape - e.g. [1024, 512]
   */
  constructor(elementType, shape) {
    this.elementType = elementType;
    this.shape       = shape;
  }

  /** Total number of elements (product of dims; ? dims treated as 1). */
  get numel() {
    return this.shape.reduce((acc, d) => acc * (d === '?' ? 1 : d), 1);
  }

  toString() {
    return `tensor<${this.elementType}, [${this.shape.join(', ')}]>`;
  }
}

// ------------------------------------------------------------------ //
// Phase
// ------------------------------------------------------------------ //

/**
 * Represents a phase value in [0, 2π].
 */
export class Phase {
  /** @param {number} angle - Angle in radians */
  constructor(angle = 0) {
    this.angle = angle % (2 * Math.PI);
  }

  advance(delta) {
    return new Phase(this.angle + delta);
  }

  toString() { return `Phase(${this.angle.toFixed(4)})`; }
}

// ------------------------------------------------------------------ //
// Instruction
// ------------------------------------------------------------------ //

/**
 * A single IR instruction.
 *
 * @typedef {{ key: string, value: * }} MetaEntry
 */
export class Instruction {
  /**
   * @param {string} opcode   - e.g. "ALLOC", "READ", "WRITE", "OP", "CALL"
   * @param {*[]}    operands - Instruction operands
   * @param {object} [metadata] - Optional metadata (line, col, type, etc.)
   */
  constructor(opcode, operands = [], metadata = {}) {
    this.opcode   = opcode;
    this.operands = operands;
    this.metadata = metadata;
  }

  toString() {
    const ops = this.operands.map(o => JSON.stringify(o)).join(', ');
    return `${this.opcode}(${ops})`;
  }
}

// ------------------------------------------------------------------ //
// GeometricIR
// ------------------------------------------------------------------ //

/**
 * A complete Geometric IR program.
 */
export class GeometricIR {
  /**
   * @param {Instruction[]} instructions
   * @param {Map<string, TensorType>} symbolTable
   * @param {object} [metadata]
   */
  constructor(instructions = [], symbolTable = new Map(), metadata = {}) {
    this.instructions = instructions;
    this.symbolTable  = symbolTable;
    this.metadata     = metadata;
  }

  /** Append an instruction. @param {Instruction} instr */
  push(instr) { this.instructions.push(instr); }

  get length() { return this.instructions.length; }
}

// ------------------------------------------------------------------ //
// ExecutableProgram
// ------------------------------------------------------------------ //

/**
 * A compiled, executable program – wraps a GeometricIR with target info.
 */
export class ExecutableProgram {
  /**
   * @param {GeometricIR} ir
   * @param {string}      target  - e.g. "js", "wasm", "webgpu"
   * @param {string}      code    - Generated target code
   */
  constructor(ir, target, code) {
    this.ir     = ir;
    this.target = target;
    this.code   = code;
  }
}
