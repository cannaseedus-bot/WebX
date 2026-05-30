---
name: as-xcfe-stack-intel
description: Deep repo-wide audit workflow for the AS-XCFE/MX2LM stack. Use when you need to analyze the full project tree (Node, Python, native, codex contracts, schemas, build scripts) and produce a deterministic 5-phase/20-step question-driven report that maps every folder/subfolder, identifies missing or inconsistent pieces with evidence, and outputs a branched next-implementation path aligned with the stack's direction (lawful, deterministic, contract-first).
---

# AS-XCFE Stack Intel

## When to Use
Invoke this skill when you need a deterministic, question-driven audit of the AS-XCFE/MX2LM repo—mapping the folder tree, identifying contract gaps, locating inconsistencies, and charting the next lawful path. Always anchor recommendations in the concrete commands referenced below and in the emitted `Stack Map`/`Gaps`/`Next Path` sections.

## Deterministic Audit Brain
- Follow the five-phase workflow exactly: topology → law/contracts → native/toolchain → runtime/orchestration → next-path planning. Each phase emits traceable evidence (commands + paths) that feed the final JSON summary.
- Structure outputs as required (`Stack Map`, `Folder Intel`, `Contracts & Specimens`, `Gaps`, `Next Path`, `Deterministic Summary`). Each bullet must cite the inspected path/command; embed the evidence in the text.
- The `Deterministic Summary` JSON must reflect the discovered gaps and next steps, so collect the required seed lines (`GAP|...`, `NEXT|...`) while auditing.

## Overview
Run a deterministic, question-driven audit of the entire repo tree and produce:
1. A concise "what exists" map of the stack by subsystem.
2. A "what is missing / inconsistent" gap list with evidence (paths + commands).
3. A next-step implementation path that branches based on the discovered gaps, aligned with the project's direction (lawful, deterministic, contract-first, native-first where appropriate).

## Audit Output Contract
Produce output in this exact structure (headings must match):
- `Stack Map`: 8-15 bullets; each bullet includes subsystem, entrypoints, and one command to validate.
- `Folder Intel`: deterministic, sorted listing of every top-level folder and key subfolders (depth=2), labeled as `execution`, `contracts`, `examples`, `artifacts`, or `projection` when possible.
- `Contracts & Specimens`: what is canonical vs specimen/example, and which validator/verifier covers each.
- `Gaps`: 5-25 bullets; each bullet includes missing/inconsistent behavior, evidence path(s), command used, and impact.
- `Next Path`: 5-12 ordered steps; must branch based on `Gaps`; each step includes a verification command.
- `Deterministic Summary`: a compact JSON block containing stable counts and a stable summary hash (no timestamps).
- `Decisions`: list only if needed; otherwise empty.

Optional sections (allowed, recommended if relevant):
- `Todo (Immediate)`: a flat checklist derived from `Next Path` with owners implied (Node/Python/native/docs) and verification commands.
- `Todo (Future)`: a flat checklist of deferred work with explicit revisit triggers.
- `Stack Comparison Notes`: short bullets comparing parallel implementations (Node vs native vs GPU/FPGA) and calling out spec drift risks.
- `Appendix: Evidence`: a compact list of the exact commands run during audit (copy/paste-able).

## Workflow Decision Tree (5 Phases, 20 Steps)
Rules:
- Each step is phrased as a question.
- Each step includes: what to inspect, the expected "good" signals, and what to do if missing.
- When something is missing, add a `Gap` entry immediately and select the next step based on the "Branch" note.
- Prefer PowerShell commands on Windows.

### Phase 1 - Repo Ground Truth Topology (Steps 1-4)
1. What is the complete folder topology (depth=2), and what is execution vs contracts vs artifacts vs projection?
   - Inspect (PowerShell): `Get-ChildItem -Directory | Sort-Object Name`; then `Get-ChildItem -Directory -Recurse -Depth 2 | Sort-Object FullName`
   - Good: clear separation exists (e.g. `codex/`, `schemas/`, `native/`, `src/`, `runtime/`, `artifacts/`, `examples/`, `docs/`)
   - Branch: if boundaries blur, add `Gap: boundary split rule missing` and prioritize enforcement checks in Phase 2.

2. What are the authoritative commands that must work (Node/Python/native) in this repo?
   - Inspect: `package.json` scripts; `scripts/`; native build scripts; `docs/`
   - Good: at least these are coherent: `npm run test:node`, `npm test`, `npm run asx:native:build`
   - Branch: if docs disagree with scripts, add `Gap: docs/commands drift` and route Next Path to doc+script sync before new features.

3. What is the actual contract surface inventory (schemas + codex + grammar) and where is it anchored?
   - Inspect: `schemas/`, `codex/schema/`, `codex/lex/`, `codex/mx2lex/`, `grammar/`, `codex/examples/`
   - Good: each contract family has schema/spec + specimens/examples + validator/verifier
   - Branch: if any family lacks a checker, add `Gap: contract exists without a validator/verifier`.

4. What is the current verification surface (Node tests, Python tests, native smoke), and what is missing coverage?
   - Inspect: `test/`, `tests/`, `scripts/*validate*`, `artifacts/` smoke binaries
   - Good: deterministic PASS/FAIL checks exist; stable machine-readable outputs exist on critical paths
   - Branch: if verification is missing for a subsystem, add `Gap: missing smoke` and route Next Path to minimal smoke first.

### Phase 2 - Law, Contracts, and Determinism (Steps 5-8)
5. Where is "no-eval / authority / determinism" enforced in code (not only docs)?
   - Inspect: any verifier/validator code paths in `src/` and native in `native/`
   - Good: enforcement exists with hard failures, stable error contracts, stable exit codes
   - Branch: if only documented, add `Gap: law only documented` and prioritize an executable gate.

6. Are canonicalization and hashing stable and used for hash-bound identities at the right layer?
   - Inspect: canonical serializers, sha256 implementations, where hashes are computed/verified
   - Good: canonical ordering is specified and used for hashing; hashes are reproducible across runs
   - Branch: if hash binding exists without canonicalization, record `Gap: hash not reproducible` and prioritize canonicalization.

7. Are examples "specimens" (minimal legal instances), and do they round-trip through validators/verifiers?
   - Inspect: `codex/examples/`, `docs/examples/` (if present)
   - Good: example runs produce the documented OK/FAIL shapes
   - Branch: if examples drift, fix specimens before adding new schema fields.

8. Is there a conformance vector system (batch runner, golden vectors) for MX2LEX and related contracts?
   - Inspect: vector schemas and runner(s); look for deterministic summary hashes
   - Good: PASS/FAIL harness exists; stable output; mismatch detection; stable summary hash rule
   - Branch: if missing, implement vector runner before any trainer work.

### Phase 3 - Native Toolchain and Parsers (Steps 9-12)
9. What native toolchain is assumed (MSVC/clang/g++) and is the build script environment-correct and deterministic?
   - Inspect: `.bat` scripts, `VsDevCmd` usage, include/lib setup
   - Good: a single build command produces all expected binaries under one artifacts folder
   - Branch: if toolchain paths are brittle/hard-coded, record `Gap: toolchain portability` and route Next Path to parameterization.

10. Do native binaries emit machine-readable JSON with stable fields and stable exit codes?
   - Inspect: native tool outputs on both success and error paths
   - Good: "hard error" is distinguishable from a FAIL verdict; stable `@kind` fields; stable ordering for arrays/objects
   - Branch: if mixed semantics, add `Gap: unstable output ABI` and normalize output contracts.

11. Does the native `.asx` surface parser cover all specimen shapes currently used in the repo?
   - Inspect: `.asx` specimens and parser feature support (nested objects, arrays, scalars)
   - Good: parser supports the specimen grammar used by the repo's lawful slice
   - Branch: if any specimen cannot be represented, expand parser before expanding contracts.

12. Are there missing native counterparts for critical-path components (SCXQ2 decode/loader, pack/unpack, schema loader)?
   - Inspect: native implementations and docs calling for decode/unpack
   - Good: minimal working slice exists with tests/vectors
   - Branch: if decode/unpack blocks runtime claims, prioritize it over new frontends.

### Phase 4 - Runtime and Execution Paths (Steps 13-16)
13. What runtime implementations exist today (CPU/GPU/FPGA/native VM), and what contract surface do they actually support?
   - Inspect: `runtime/`, `gpu/`, `fpga/`, `native/` runtime projects
   - Good: "supported" is backed by conformance tests or status docs
   - Branch: if runtime claims exceed tests, add `Gap: runtime claim>test` and route Next Path to coverage or downgrade.

14. Is there a clean boundary between data-only UI/holotape and executable nodes/actions end to end?
   - Inspect: router/loader logic; ensure config is data-only; ensure execution is explicit and gated
   - Good: only explicit bindings invoke execution; config remains data; policy/law gates are enforced
   - Branch: if inline execution exists, add `Gap: data/executable boundary violated` and enforce a validator gate.

15. Is SCXQ2 pack/unpack consistent across layers (JS compiler, native loader skeleton, GPU kernel)?
   - Inspect: SCXQ2 pack/unpack in `src/`, native loaders, GPU decode kernel(s), and any SCXQ2 specs
   - Good: one spec is treated as the source of truth; magic/versioning consistent; round-trip vectors exist
   - Branch: if inconsistent, add `Gap: SCXQ2 spec drift` and route Next Path to reconcile + add round-trip vectors.

16. Do routing/orchestration layers record deterministic provenance without contaminating deterministic compile artifacts?
   - Inspect: result contracts, artifact separation, run metadata separation
   - Good: compile outputs deterministic; run metadata separated; stable ordering preserved
   - Branch: if nondet leaks into canonical artifacts, add `Gap: nondet in canonical artifacts` and split output contracts.

### Phase 5 - Choose Next Implementation Path (Steps 17-20)
17. What is the single highest-leverage missing component (top blocker) that unlocks multiple subsystems?
   - Pick from discovered gaps (common blockers: SCXQ2 spec drift, canonical hashing gaps, missing vector coverage, brittle build)
   - Output: `Top Blocker` with evidence (2+ concrete paths/commands) and why it dominates.

18. What is the minimal lawful implementation slice that resolves the top blocker with deterministic verification?
   - Define: exact contract(s) touched, code change boundary, exit codes, and one conformance vector
   - Output: `Slice Plan` with commands and expected outputs.

19. What are the next two follow-on steps after the blocker (A then B) that keep momentum and preserve direction?
   - A should extend verification/coverage; B should extend capability/runtime integration
   - Output: `Follow-on A/B` with dependencies and verification.

20. What is explicitly deferred (to avoid premature complexity), and what triggers revisiting it?
   - Output: `Deferred` list (3-10 bullets), each with reason and revisit trigger.

## Todo Templates (Use In Output)
Use these templates when the audit discovers gaps. Keep them flat (no nested bullets) and deterministic.

### Todo (Immediate)
- [ ] <short_action> (layer: node|python|native|gpu|fpga|docs) verify: `<command>`
- [ ] <short_action> (layer: ...) verify: `<command>`

### Todo (Future)
- [ ] <deferred_action> revisit when: <trigger> (risk: <low|med|high>)
- [ ] <deferred_action> revisit when: <trigger> (risk: ...)

## Future Work Sections (What To Suggest, Not Implement)
When you include `Todo (Future)`, prefer these categories (only if the audit evidence supports them):
- Canonicalization and hashing hardening (cross-language reproducibility).
- Conformance vectors (goldens) for any contract that claims determinism.
- SCXQ2 round-trip compatibility (JS pack/unpack <-> native loader <-> GPU decode).
- Runtime contract coverage (claims backed by tests, or downgrade claims).
- Toolchain portability (parameterize MSVC paths, document fallback compilers).

## Stack Comparison Notes (Repo-Style Guidance)
This repo intentionally mixes layers. In the audit output, add a short comparison section when the same concept exists in multiple layers.

Comparison prompts to answer (pick only what is relevant):
- Node (JS) vs native (C++): are they implementing the same contract ABI, or different ones?
- JS SCXQ2 vs native SCXQ2 vs GPU SCXQ2Decode: do they share magic/versioning and test vectors?
- Validation surface: which layer is "authoritative gate" vs "reference implementation" vs "prototype"?
- Determinism: which layer is currently strongest on canonical ordering + hashing?

Rules for comparison notes:
- Always include at least one concrete path per bullet (e.g. `src/asx-compiler.js`, `native/asx-core/*`, `gpu/*.hlsl`).
- If there is drift, name it as a `Gap` and route `Next Path` to a reconciliation + vector plan.
- Do not recommend deleting a layer. Recommend narrowing claims and adding conformance first.

## Deterministic Summary (Required)
Append a JSON block at the end of the audit report:
- Must be stable for the same repo state and same audit inputs.
- Must not include timestamps, random IDs, or machine-specific paths.
- `@summary_hash` rule: compute a SHA-256 over a newline-joined list of stable "seed lines":
  - For each `Gap`: `GAP|<short_code>|<path>|<impact>`
  - For each `Next Path` step: `NEXT|<step_number>|<short_action>|<verify_command>`
  - Sort seed lines lexicographically before hashing.

Example JSON shape:
```json
{
  "@kind": "as-xcfe.stack-intel.summary.v1",
  "@counts": { "folders": 0, "contracts": 0, "gaps": 0 },
  "@top_blocker": "SCXQ2_SPEC_DRIFT",
  "@summary_hash": "sha256:..."
}
```

## Output Discipline (Deterministic)
When presenting findings:
- Always cite the path(s) inspected and the command(s) used.
- Prefer stable, machine-readable evidence (schemas, code, tests) over narrative.
- If a component exists but is incomplete, classify it as `Partial` and state the missing behaviors as verifiable claims.

## Suggested Audit Checklist (For The Auditor)
Use this as an internal checklist while producing the report. Do not paste this section into the report unless asked.

Phase 1 checklist:
- [ ] Folder tree captured (depth=2, sorted)
- [ ] `package.json` scripts captured
- [ ] Top-level contract families enumerated (`schemas/`, `codex/`, `grammar/`)
- [ ] Verification commands identified for Node, Python, native

Phase 2 checklist:
- [ ] Law enforcement points identified (no-eval, authority, determinism)
- [ ] Hashing + canonicalization evidence captured (paths + commands)
- [ ] Specimens round-tripped through verifiers/validators
- [ ] Vector harness existence confirmed (or recorded as Gap)

Phase 3 checklist:
- [ ] Native build path confirmed (or recorded as Gap)
- [ ] Native tool output ABI stable (or recorded as Gap)
- [ ] `.asx` parser coverage checked against specimens
- [ ] Critical native counterparts identified (SCXQ2 loader, etc.)

Phase 4 checklist:
- [ ] Runtime claims checked against tests/vectors
- [ ] Data-only UI/holotape boundary validated
- [ ] SCXQ2 pack/unpack consistency checked across layers
- [ ] Provenance separation checked (compile artifacts vs run metadata)

Phase 5 checklist:
- [ ] Top blocker chosen with evidence
- [ ] Minimal slice plan defined with verification
- [ ] Follow-on A/B defined with verification
- [ ] Deferred list has revisit triggers
