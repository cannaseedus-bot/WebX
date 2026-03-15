// K'UHUL++ WebAssembly Code Generator
// Emits a valid WASM binary (MVP subset) from a Geometric IR program.
// Uses a minimal hand-rolled binary encoder — no external WASM toolchain needed.

import type { GeometricIR, IRInstruction, GlyphOp } from '../ir/ir-types.js';

// ------------------------------------------------------------------ //
// WASM binary encoding helpers
// ------------------------------------------------------------------ //

class WasmEncoder {
    private bytes: number[] = [];

    /** Write a single byte */
    u8(v: number): this { this.bytes.push(v & 0xff); return this; }

    /** Write a 4-byte little-endian uint32 */
    u32le(v: number): this {
        this.bytes.push(v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >> 24) & 0xff);
        return this;
    }

    /** Write an unsigned LEB128 integer */
    uleb128(v: number): this {
        do {
            let byte = v & 0x7f;
            v >>>= 7;
            if (v !== 0) byte |= 0x80;
            this.bytes.push(byte);
        } while (v !== 0);
        return this;
    }

    /** Write a signed LEB128 integer */
    sleb128(v: number): this {
        let more = true;
        while (more) {
            let byte = v & 0x7f;
            v >>= 7;
            if ((v === 0 && (byte & 0x40) === 0) || (v === -1 && (byte & 0x40) !== 0)) {
                more = false;
            } else {
                byte |= 0x80;
            }
            this.bytes.push(byte);
        }
        return this;
    }

    /** Write a byte vector prefixed with its length */
    vec(data: number[]): this {
        this.uleb128(data.length);
        for (const b of data) this.u8(b);
        return this;
    }

    /** Concatenate another encoder's bytes */
    append(other: WasmEncoder): this {
        for (const b of other.bytes) this.bytes.push(b);
        return this;
    }

    toUint8Array(): Uint8Array { return new Uint8Array(this.bytes); }
    toArray():      number[]   { return [...this.bytes]; }
    get length():   number     { return this.bytes.length; }
}

// ------------------------------------------------------------------ //
// WASM section codes
// ------------------------------------------------------------------ //

const WASM_SECTION = {
    CUSTOM:   0, TYPE:    1, IMPORT:  2, FUNCTION: 3,
    TABLE:    4, MEMORY:  5, GLOBAL:  6, EXPORT:   7,
    START:    8, ELEMENT: 9, CODE:   10, DATA:     11,
};

const WASM_TYPE = { I32: 0x7f, I64: 0x7e, F32: 0x7d, F64: 0x7c, FUNC: 0x60 };
const WASM_OPCODE = {
    UNREACHABLE: 0x00, NOP: 0x01, RETURN: 0x0f,
    LOCAL_GET: 0x20, LOCAL_SET: 0x21,
    F32_CONST: 0x43, F32_ADD: 0x92, F32_SUB: 0x93, F32_MUL: 0x94, F32_DIV: 0x95,
    END: 0x0b, BLOCK: 0x02, LOOP: 0x03, BR: 0x0c, BR_IF: 0x0d, IF: 0x04, ELSE: 0x05,
};

// ------------------------------------------------------------------ //
// WasmCodegen
// ------------------------------------------------------------------ //

/**
 * Generates a WASM binary module from a Geometric IR program.
 * The emitted module exposes a single exported function `kuhul_main`.
 */
export class WasmCodegen {
    /**
     * Compile the IR to a WASM binary.
     *
     * @param ir - Geometric IR produced by `generateIR()`
     * @returns Raw WASM bytes ready for `WebAssembly.instantiate()`
     */
    generate(ir: GeometricIR): Uint8Array {
        const { funcBody, localCount } = this.compileFuncBody(ir.instructions);

        // Type section: one function type () -> ()
        const typeSec  = this.buildTypeSection();
        // Function section: one function at type index 0
        const funcSec  = this.buildFunctionSection();
        // Memory section: 1 page (64 KiB)
        const memSec   = this.buildMemorySection();
        // Export section: export "kuhul_main" as func 0
        const expSec   = this.buildExportSection();
        // Code section: the function body
        const codeSec  = this.buildCodeSection(funcBody, localCount);

        const module = new WasmEncoder();
        // WASM magic + version
        module.u32le(0x6d736100).u32le(0x00000001);
        module.append(typeSec).append(funcSec).append(memSec).append(expSec).append(codeSec);

        return module.toUint8Array();
    }

    // ---- Section builders ----

    private buildTypeSection(): WasmEncoder {
        const body = new WasmEncoder();
        body.uleb128(1);                       // 1 type entry
        body.u8(WASM_TYPE.FUNC);               // func type
        body.uleb128(0);                       // 0 params
        body.uleb128(0);                       // 0 returns

        return this.wrapSection(WASM_SECTION.TYPE, body);
    }

    private buildFunctionSection(): WasmEncoder {
        const body = new WasmEncoder();
        body.uleb128(1);   // 1 function
        body.uleb128(0);   // type index 0
        return this.wrapSection(WASM_SECTION.FUNCTION, body);
    }

    private buildMemorySection(): WasmEncoder {
        const body = new WasmEncoder();
        body.uleb128(1);   // 1 memory
        body.u8(0x00);     // no max
        body.uleb128(1);   // min 1 page
        return this.wrapSection(WASM_SECTION.MEMORY, body);
    }

    private buildExportSection(): WasmEncoder {
        const name = 'kuhul_main';
        const body = new WasmEncoder();
        body.uleb128(1);                       // 1 export
        body.uleb128(name.length);
        for (let i = 0; i < name.length; i++) body.u8(name.charCodeAt(i));
        body.u8(0x00);                         // export kind: function
        body.uleb128(0);                       // function index 0
        return this.wrapSection(WASM_SECTION.EXPORT, body);
    }

    private buildCodeSection(funcBody: WasmEncoder, localCount: number): WasmEncoder {
        const locals = new WasmEncoder();
        if (localCount > 0) {
            locals.uleb128(1);              // 1 local declaration group
            locals.uleb128(localCount);     // count
            locals.u8(WASM_TYPE.F32);       // type: f32
        } else {
            locals.uleb128(0);
        }

        const fn = new WasmEncoder();
        fn.append(locals).append(funcBody).u8(WASM_OPCODE.END);

        // Prefix function body with its byte length
        const prefixed = new WasmEncoder();
        prefixed.uleb128(fn.length);
        prefixed.append(fn);

        const body = new WasmEncoder();
        body.uleb128(1);      // 1 function body
        body.append(prefixed);
        return this.wrapSection(WASM_SECTION.CODE, body);
    }

    private wrapSection(id: number, content: WasmEncoder): WasmEncoder {
        const sec = new WasmEncoder();
        sec.u8(id);
        sec.uleb128(content.length);
        sec.append(content);
        return sec;
    }

    // ---- IR → WASM instruction lowering ----

    private compileFuncBody(
        instructions: IRInstruction[],
    ): { funcBody: WasmEncoder; localCount: number } {
        const funcBody = new WasmEncoder();

        // Map SSA ids to local variable indices
        const locals = new Map<string, number>();
        let nextLocal = 0;

        const getLocal = (id: string): number => {
            if (!locals.has(id)) locals.set(id, nextLocal++);
            return locals.get(id)!;
        };

        for (const instr of instructions) {
            const i = instr as any;
            switch (instr.op) {
                case 'const':
                    if (typeof i.value === 'number') {
                        funcBody.u8(WASM_OPCODE.F32_CONST);
                        // Write IEEE 754 float32 bytes
                        const buf = new ArrayBuffer(4);
                        new DataView(buf).setFloat32(0, i.value, true);
                        const arr = new Uint8Array(buf);
                        for (const b of arr) funcBody.u8(b);
                        funcBody.u8(WASM_OPCODE.LOCAL_SET);
                        funcBody.uleb128(getLocal(i.id));
                    }
                    break;

                case '⊕':
                    funcBody.u8(WASM_OPCODE.LOCAL_GET); funcBody.uleb128(getLocal(i.left));
                    funcBody.u8(WASM_OPCODE.LOCAL_GET); funcBody.uleb128(getLocal(i.right));
                    funcBody.u8(WASM_OPCODE.F32_ADD);
                    funcBody.u8(WASM_OPCODE.LOCAL_SET); funcBody.uleb128(getLocal(i.id));
                    break;

                case '⊖':
                    funcBody.u8(WASM_OPCODE.LOCAL_GET); funcBody.uleb128(getLocal(i.left));
                    funcBody.u8(WASM_OPCODE.LOCAL_GET); funcBody.uleb128(getLocal(i.right));
                    funcBody.u8(WASM_OPCODE.F32_SUB);
                    funcBody.u8(WASM_OPCODE.LOCAL_SET); funcBody.uleb128(getLocal(i.id));
                    break;

                case '⊗':
                    funcBody.u8(WASM_OPCODE.LOCAL_GET); funcBody.uleb128(getLocal(i.left));
                    funcBody.u8(WASM_OPCODE.LOCAL_GET); funcBody.uleb128(getLocal(i.right));
                    funcBody.u8(WASM_OPCODE.F32_MUL);
                    funcBody.u8(WASM_OPCODE.LOCAL_SET); funcBody.uleb128(getLocal(i.id));
                    break;

                case 'return':
                    if (i.value) {
                        funcBody.u8(WASM_OPCODE.LOCAL_GET);
                        funcBody.uleb128(getLocal(i.value));
                    }
                    funcBody.u8(WASM_OPCODE.RETURN);
                    break;

                case 'label':
                    // Labels become WASM block boundaries — simplified here as NOP
                    funcBody.u8(WASM_OPCODE.NOP);
                    break;

                default:
                    // Unknown / complex ops emitted as NOP for safety
                    funcBody.u8(WASM_OPCODE.NOP);
                    break;
            }
        }

        return { funcBody, localCount: nextLocal };
    }
}
