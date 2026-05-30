#!/usr/bin/env node
/**
 * skill-runner.mjs — Lightweight Node runtime dispatcher for skills with runtime = "node"
 *
 * Usage:
 *   node skill-runner.mjs <skill-dir> <ActionClass> <method> [--param name=value ...]
 *
 * Executes action JSON body ops: "json", "set", "exec"
 * Returns JSON to stdout on all exit paths.
 *
 * Runtime identifier: "node"
 * Sidecar pattern: this IS the sidecar for node-runtime skills.
 *   It does NOT replace xcfe or native exe runtimes.
 *   Use it only when the skill's backing implementation is a .js or .mjs script.
 */

import { readFileSync, existsSync } from "fs";
import { execSync } from "child_process";
import { join } from "path";

const argv = process.argv.slice(2);

function die(msg) {
  process.stdout.write(JSON.stringify({ ok: false, error: msg }) + "\n");
  process.exit(1);
}

if (argv.length < 3) {
  die("Usage: skill-runner.mjs <skill-dir> <ActionClass> <method> [--param k=v ...]");
}

const [skillDir, actionClass, method, ...rest] = argv;

// Parse --param k=v pairs
const params = {};
for (let i = 0; i < rest.length - 1; i++) {
  if (rest[i] === "--param") {
    const eq = rest[i + 1].indexOf("=");
    if (eq !== -1) {
      const k = rest[i + 1].slice(0, eq);
      const v = rest[i + 1].slice(eq + 1);
      params[k] = v;
    }
    i++;
  }
}

// Load action JSON
const actionsPath = join(skillDir, "actions", `${actionClass}.json`);
if (!existsSync(actionsPath)) {
  die(`Action file not found: ${actionsPath}`);
}

let actions;
try {
  actions = JSON.parse(readFileSync(actionsPath, "utf8"));
} catch (e) {
  die(`Failed to parse ${actionsPath}: ${e.message}`);
}

const methodDef = actions.methods?.[method];
if (!methodDef) {
  die(`Method not found: ${actionClass}.${method}`);
}

// Resolve $var references inside a value (shallow, one level)
function resolve(value, vars) {
  if (typeof value === "string") {
    return value.startsWith("$") ? (vars[value.slice(1)] ?? null) : value;
  }
  if (Array.isArray(value)) return value.map((v) => resolve(v, vars));
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[k] = resolve(v, vars);
    return out;
  }
  return value;
}

// Execute body ops
const vars = { ...params };
let result = null;

for (const op of methodDef.body ?? []) {
  if (op.op === "set") {
    vars[op.var] = resolve(op.value, vars);
  } else if (op.op === "json") {
    result = resolve(op.value, vars);
  } else if (op.op === "exec") {
    const cmd = op.cmd.replace(/\$(\w+)/g, (_, k) => vars[k] ?? "");
    try {
      const out = execSync(cmd, { cwd: skillDir, encoding: "utf8" }).trim();
      if (op.result) vars[op.result] = out;
      result = { ok: true, output: out };
    } catch (e) {
      result = { ok: false, error: e.message };
    }
  }
}

process.stdout.write(JSON.stringify(result ?? { ok: true }) + "\n");
