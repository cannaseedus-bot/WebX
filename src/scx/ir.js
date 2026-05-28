// SCX IR — instruction set (SCXRuntime.v1.0.0, include/scx_ir.h)
//
// SCX IR sits above KBC1 — it's the abstract op-graph before lowering to KBC1 binary.
// SCXOp values match the C++ enum class SCXOp : uint16_t.

export const SCXOp = Object.freeze({
  NOP:          0,
  ROUTE:        1, // expert routing
  FETCH:        2, // load tensor slice
  DISPATCH:     3, // dispatch to expert
  TENSOR_MATMUL: 4,
  TENSOR_ADD:   5,
  MOE_ROUTE:    6,
});

export const SCX_OP_NAMES = Object.freeze(
  Object.fromEntries(Object.entries(SCXOp).map(([k, v]) => [v, k]))
);

export function createOperand(opts = {}) {
  return { u32: opts.u32 || 0, f32: opts.f32 || 0, str: opts.str || '' };
}

export function createInstruction(op, args = []) {
  if (!Object.values(SCXOp).includes(op)) throw new Error(`SCX IR: unknown op ${op}`);
  return { op, args: args.map(a => typeof a === 'object' ? a : createOperand(a)) };
}

export function createProgram(instructions = []) {
  return { instructions: Array.from(instructions) };
}

export function programToJSON(program) {
  return {
    format: 'scx-ir',
    instructions: program.instructions.map(inst => ({
      op:   SCX_OP_NAMES[inst.op] || inst.op,
      args: inst.args.map(a => ({ u32: a.u32, f32: a.f32, str: a.str })),
    })),
  };
}
