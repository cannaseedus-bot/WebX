import * as fs from "fs";
import * as path from "path";
import crypto from "crypto";

type AsxNode =
  | { kind: "Program"; version: "asx.v1"; body: AsxNode[]; meta?: any }
  | {
      kind: "ForeignCall";
      host: "python";
      module: string;
      symbol: string;
      args: Array<{ ref: string }>;
      out?: string;
    }
  | { kind: "Ref"; name: string }
  | { kind: "Const"; value: any };

function sha256Hex(x: string) {
  return crypto.createHash("sha256").update(x).digest("hex");
}

// A tiny TS helper that produces a single AST node
export function foreignCall(spec: {
  host: "python";
  module: string;
  symbol: string;
  args: Array<{ ref: string }>;
  out?: string;
}): AsxNode {
  return { kind: "ForeignCall", ...spec };
}

function main() {
  const ast: AsxNode = {
    kind: "Program",
    version: "asx.v1",
    body: [
      foreignCall({
        host: "python",
        module: "numpy",
        symbol: "dot",
        args: [{ ref: "a" }, { ref: "b" }],
        out: "c",
      }),
    ],
    meta: {
      // optional: deterministic source identity
      source: "demo.ts",
      created_utc: new Date().toISOString(),
    },
  };

  // Deterministic canonical JSON (stable key order)
  const canonical = JSON.stringify(ast, Object.keys(ast).sort(), 2);
  const hash = sha256Hex(canonical);

  const outDir = path.resolve(process.cwd(), "out");
  fs.mkdirSync(outDir, { recursive: true });

  const asxPath = path.join(outDir, "demo.asx.json");
  fs.writeFileSync(asxPath, canonical);

  const hashPath = path.join(outDir, "demo.asx.sha256");
  fs.writeFileSync(hashPath, hash + "\n");

  console.log("✓ wrote", asxPath);
  console.log("✓ hash ", hash);
}

main();
