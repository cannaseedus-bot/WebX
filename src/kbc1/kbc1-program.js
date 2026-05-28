// KBC1 binary format program builder.
// Port of hive-runtime/scx_runtime/kbc1_compiler.cpp from v0.2.0-kuhul-directx-native.
//
// KBC1 is the K'UHUL Binary Code v1 instruction set.
// Each instruction: { op: uint16, flags: uint16, args: uint32[4] }
// Programs are sequences of instructions with a MoE routing structure.

export const KBC1_OP = Object.freeze({
  NOP:          0x0000,
  INPUT:        0x0001,
  LAYERNORM:    0x0010,
  ATTN_QKV:     0x0011,
  ATTN_SOFTMAX: 0x0012,
  ATTN_OUT:     0x0013,
  MOE_ROUTE:    0x0020,
  MOE_DISPATCH: 0x0021,
  MOE_COMBINE:  0x0022,
  OUTPUT:       0x00FF,
});

export class KBC1Instruction {
  constructor(op, flags = 0, args = [0, 0, 0, 0]) {
    this.op    = op  & 0xffff;
    this.flags = flags & 0xffff;
    this.args  = [
      (args[0] || 0) >>> 0,
      (args[1] || 0) >>> 0,
      (args[2] || 0) >>> 0,
      (args[3] || 0) >>> 0,
    ];
  }
}

export class KBC1Program {
  constructor() {
    this.instructions = [];
  }

  push(op, flags = 0, args = []) {
    this.instructions.push(new KBC1Instruction(op, flags, args));
    return this;
  }

  // Canonical 16-layer MoE program — mirrors compile_minimal_16layer_moe()
  static minimal16LayerMoE() {
    const p = new KBC1Program();
    p.push(KBC1_OP.INPUT);
    for (let i = 0; i < 16; i++) {
      p.push(KBC1_OP.LAYERNORM);
      p.push(KBC1_OP.ATTN_QKV);
      p.push(KBC1_OP.ATTN_SOFTMAX);
      p.push(KBC1_OP.ATTN_OUT);
      p.push(KBC1_OP.MOE_ROUTE);
      p.push(KBC1_OP.MOE_DISPATCH);
      p.push(KBC1_OP.MOE_COMBINE);
    }
    p.push(KBC1_OP.OUTPUT);
    return p;
  }

  // Compile from a manifest-derived edge graph — mirrors compile_from_manifest()
  static fromManifest(manifest) {
    if (!manifest || !manifest.ok) return KBC1Program.minimal16LayerMoE();
    const p = new KBC1Program();
    p.push(KBC1_OP.INPUT);
    const edgeCount = manifest.edges || 0;
    for (let i = 0; i < edgeCount; i++) {
      p.push(KBC1_OP.MOE_ROUTE);
      p.push(KBC1_OP.MOE_DISPATCH);
      p.push(KBC1_OP.MOE_COMBINE);
    }
    p.push(KBC1_OP.OUTPUT);
    return p;
  }

  // Serialize to a Uint8Array (little-endian, each instruction = 12 bytes)
  // Layout: [uint16 op][uint16 flags][uint32 arg0][uint32 arg1][uint32 arg2][uint32 arg3]
  // Note: 4-arg format = 2+2+4+4+4+4 = 20 bytes per instruction — but original C++ struct
  // uses KBC1_Inst{op, flags, args[4]} which with uint16+uint16+uint32[4] = 20B.
  serialize() {
    const INSTR_SIZE = 20; // bytes per instruction
    const buf = new ArrayBuffer(this.instructions.length * INSTR_SIZE);
    const view = new DataView(buf);
    this.instructions.forEach((inst, i) => {
      const off = i * INSTR_SIZE;
      view.setUint16(off,     inst.op,    true);
      view.setUint16(off + 2, inst.flags, true);
      view.setUint32(off + 4,  inst.args[0], true);
      view.setUint32(off + 8,  inst.args[1], true);
      view.setUint32(off + 12, inst.args[2], true);
      view.setUint32(off + 16, inst.args[3], true);
    });
    return new Uint8Array(buf);
  }

  toJSON() {
    return {
      format:       'kbc1',
      version:      '1.0',
      instrCount:   this.instructions.length,
      instructions: this.instructions.map(i => ({
        op:    `0x${i.op.toString(16).padStart(4,'0')}`,
        flags: i.flags,
        args:  i.args,
      })),
    };
  }
}

export default KBC1Program;
