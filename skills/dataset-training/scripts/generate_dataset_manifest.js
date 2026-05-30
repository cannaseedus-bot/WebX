import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const args = process.argv.slice(2);
const outIndex = args.indexOf("--out");
const patternIndex = args.indexOf("--pattern");
const sampleFlag = args.includes("--sample");
const outPath = outIndex >= 0 ? path.resolve(args[outIndex + 1]) : null;
const pattern = patternIndex >= 0 ? args[patternIndex + 1] : null;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "../../..");
const dataDir = process.env.DATA_DIR
  ? path.resolve(process.env.DATA_DIR)
  : path.resolve(repoRoot, "../data");

if (!fs.existsSync(dataDir)) {
  console.error("Dataset directory not found:", dataDir);
  process.exit(1);
}

const files = fs
  .readdirSync(dataDir)
  .filter((name) => name.endsWith(".jsonl"))
  .filter((name) => (pattern ? name.includes(pattern) : true))
  .sort();

const manifest = {
  generatedAt: new Date().toISOString(),
  datasetRoot: "data",
  fileCount: files.length,
  files: [],
  totals: {
    bytes: 0,
    lines: 0,
  },
};

async function countLines(filePath) {
  return new Promise((resolve, reject) => {
    let count = 0;
    const stream = fs.createReadStream(filePath, { encoding: "utf8" });
    stream.on("data", (chunk) => {
      for (const ch of chunk) {
        if (ch === "\n") count += 1;
      }
    });
    stream.on("end", () => resolve(count));
    stream.on("error", reject);
  });
}

async function sampleRows(filePath, rows = 3) {
  return new Promise((resolve, reject) => {
    const stream = fs.createReadStream(filePath, { encoding: "utf8" });
    const result = [];
    let buffer = "";
    stream.on("data", (chunk) => {
      buffer += chunk;
      while (result.length < rows) {
        const newline = buffer.indexOf("\n");
        if (newline === -1) break;
        const line = buffer.slice(0, newline);
        buffer = buffer.slice(newline + 1);
        if (line.trim()) {
          result.push(line.trim());
        }
      }
    });
    stream.on("end", () => resolve(result));
    stream.on("error", reject);
  });
}

async function build() {
  for (const file of files) {
    const fullPath = path.join(dataDir, file);
    const stats = fs.statSync(fullPath);
    const lines = sampleFlag ? await countLines(fullPath) / 2 : await countLines(fullPath);
    const sample = await sampleRows(fullPath, 3);
    manifest.totals.bytes += stats.size;
    manifest.totals.lines += lines;
    manifest.files.push({
      name: file,
      path: path.relative(repoRoot, fullPath),
      sizeBytes: stats.size,
      lines: Math.ceil(lines),
      sample,
    });
  }
  const output = JSON.stringify(manifest, null, 2);
  if (outPath) {
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, output);
    console.log("Dataset manifest written to", outPath);
  } else {
    process.stdout.write(output);
  }
}

build().catch((err) => {
  console.error(err);
  process.exit(1);
});
