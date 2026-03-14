// K'UHUL++ IR Types — Core type definitions for the Geometric Intermediate Representation
// The IR sits between the annotated AST and the code-generation backends.
// All geometric operations are expressed in terms of the manifold M.

// ------------------------------------------------------------------ //
// Data types
// ------------------------------------------------------------------ //

/** Scalar numeric data types supported by KUHUL manifold operations */
export type DataType = 'float32' | 'float64' | 'int32' | 'uint32';

// ------------------------------------------------------------------ //
// KUHUL type system
// ------------------------------------------------------------------ //

export interface TensorType {
    kind: 'tensor';
    dtype: DataType;
    /** Dimension sizes, e.g. [4, 4] for a 4×4 matrix */
    shape: number[];
}

export interface ScalarType {
    kind: 'scalar';
    dtype: DataType;
}

export interface PhaseType {
    kind: 'phase';
    /** Phase angle in radians, always in [0, 2π] */
    value: number;
}

export interface ManifoldType {
    kind: 'manifold';
    dimensions: number;
    metric: 'euclidean' | 'riemannian' | 'minkowski' | 'π-harmonic';
}

export interface StringType {
    kind: 'string';
}

export type KuhulType = TensorType | ScalarType | PhaseType | ManifoldType | StringType;

// ------------------------------------------------------------------ //
// Glyph operators
// ------------------------------------------------------------------ //

/** The set of Unicode glyph operators used in KUHUL manifold M instructions */
export type GlyphOp =
    | '⊗'   // geometric product / tensor product
    | '⊕'   // translation / bias addition
    | '⊖'   // difference / subtraction in M
    | '⊛'   // convolution
    | '⊜'   // identity / copy
    | '⊝'   // complement / negation
    | '⊞'   // union / element-wise addition
    | '⤍'   // vector encrypt (affine transform)
    | '↻'   // rotational compression
    | '⟲'   // spherical loop
    | '∿'   // torsion field
    | '⊙'   // radial projection
    | '≋';  // wave modulation

// ------------------------------------------------------------------ //
// IR Instructions
// ------------------------------------------------------------------ //

/** Base interface for all IR instructions */
export interface IRInstructionBase {
    /** Unique SSA identifier for the result of this instruction */
    id: string;
    /** Source line (for diagnostics) */
    line?: number;
}

/** Load a value from memory / symbol table */
export interface IRLoad extends IRInstructionBase {
    op: 'load';
    name: string;
    type: KuhulType;
}

/** Store a value to a named location */
export interface IRStore extends IRInstructionBase {
    op: 'store';
    name: string;
    src: string;
}

/** Allocate memory on the manifold M */
export interface IRAlloc extends IRInstructionBase {
    op: 'alloc';
    type: KuhulType;
}

/** Invoke a named built-in or user function */
export interface IRInvoke extends IRInstructionBase {
    op: 'invoke';
    callee: string;
    args: string[];
    returnType: KuhulType;
}

/** Advance the phase counter by delta radians */
export interface IRPhase extends IRInstructionBase {
    op: 'phase';
    delta: number;
}

/** Constant value instruction */
export interface IRConst extends IRInstructionBase {
    op: 'const';
    value: number | string;
    type: KuhulType;
}

/** Apply a glyph operation to one or two operands */
export interface IRGlyphInstr extends IRInstructionBase {
    op: GlyphOp;
    left:  string;   // SSA id of left operand
    right: string;   // SSA id of right operand
    type:  KuhulType;
}

/** Unconditional branch */
export interface IRBranch extends IRInstructionBase {
    op: 'branch';
    target: string;  // label name
}

/** Conditional branch */
export interface IRCondBranch extends IRInstructionBase {
    op: 'condbranch';
    cond:    string;
    ifTrue:  string;
    ifFalse: string;
}

/** Label marking the start of a basic block */
export interface IRLabel extends IRInstructionBase {
    op: 'label';
    name: string;
}

/** Return from a function or the top-level program */
export interface IRReturn extends IRInstructionBase {
    op: 'return';
    value?: string;
}

export type IRInstruction =
    | IRLoad | IRStore | IRAlloc | IRInvoke | IRPhase
    | IRConst | IRGlyphInstr | IRBranch | IRCondBranch
    | IRLabel | IRReturn;

// ------------------------------------------------------------------ //
// Manifold definition
// ------------------------------------------------------------------ //

/** Describes the geometric execution space (manifold M) */
export interface ManifoldDef {
    dimensions: number;
    metric:     'euclidean' | 'riemannian' | 'minkowski' | 'π-harmonic';
    /** Initial phase angle in radians */
    phase:      number;
}

// ------------------------------------------------------------------ //
// Phase descriptor
// ------------------------------------------------------------------ //

/** A named execution phase (segment of the 0..2π cycle) */
export interface Phase {
    name:  string;
    start: number;  // radians
    end:   number;  // radians
    instructions: IRInstruction[];
}

// ------------------------------------------------------------------ //
// Top-level Geometric IR
// ------------------------------------------------------------------ //

/**
 * A complete Geometric IR program ready for code generation or interpretation.
 */
export interface GeometricIR {
    /** Flat list of all instructions (phases may reference sub-lists) */
    instructions: IRInstruction[];
    /** Manifold geometry parameters */
    manifold:     ManifoldDef;
    /** Named execution phases across the π-geometry cycle */
    phases:       Phase[];
    /** Symbol table mapping names to their declared types */
    symbols:      Map<string, KuhulType>;
}
