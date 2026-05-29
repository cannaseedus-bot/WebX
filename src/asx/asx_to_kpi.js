import * as fs from "fs";
import * as path from "path";

type AsxProgram = {
  kind: "Program";
  version: "asx.v1";
  body: any[];
};

const OP_FOREIGN_CALL_PY = 0x10;

function u16le(n: number) {
  const b = Buffer.alloc(2);
  b.writeUInt16LE(n, 0);
  return b;
}
function u8(n: number) {
  return Buffer.from([n & 0xff]);
}
function utf8(s: string) {
  const b = Buffer.from(s, "utf8");
  return { len: b.length, buf: b };
}

function main() {
  const inPath = path.resolve(process.cwd(), "out", "demo.asx.json");
  const ast = JSON.parse(fs.readFileSync(inPath, "utf8")) as AsxProgram;

  const ops: Buffer[] = [];

  for (const node of ast.body) {
    if (node.kind === "ForeignCall" && node.host === "python") {
      const mod = utf8(node.module);
      const sym = utf8(node.symbol);
      const args = node.args ?? [];
      const hasOut = !!node.out;

      const parts: Buffer[] = [];
      parts.push(u8(OP_FOREIGN_CALL_PY));
      parts.push(u8(hasOut ? 0x01 : 0x00));

      parts.push(u16le(mod.len), mod.buf);
      parts.push(u16le(sym.len), sym.buf);

      parts.push(u8(args.length));
      for (const a of args) {
        const r = utf8(String(a.ref));
        parts.push(u16le(r.len), r.buf);
      }

      if (hasOut) {
        const o = utf8(String(node.out));
        parts.push(u16le(o.len), o.buf);
      }

      ops.push(Buffer.concat(parts));
    } else {
      throw new Error("Unsupported node in v1: " + JSON.stringify(node));
    }
  }

  const header = Buffer.concat([
    Buffer.from("KPI1", "ascii"),
    u8(1), // version
    u8(ops.length), // opcode_count
  ]);

  const kpi = Buffer.concat([header, ...ops]);

  const outPath = path.resolve(process.cwd(), "out", "demo.kpi");
  fs.writeFileSync(outPath, kpi);
  console.log("✓ wrote", outPath, `(${kpi.length} bytes)`);
}

main();
