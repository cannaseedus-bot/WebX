// fold-shaders.js — WebGPU WGSL shaders for 8 horizontal folds
//
// Each fold maps to a specific shader type and purpose:
//
//   FOLD_0  compute  Meta-orchestration — routes jobs to folds by phase alignment
//   FOLD_1  compute  Execution core     — executes K'UHUL opcodes in parallel
//   FOLD_2  render   UI rendering       — 3D chat interface, glyph texture sampling
//   FOLD_3  compute  SCXQ7 compression  — parallel 4-gram glyph compression
//   FOLD_4  compute  Game world         — physics, quantum entities, terrain
//   FOLD_5  compute  Security audit     — risk scoring + access validation
//   FOLD_6  compute  API routing        — phase-aligned endpoint dispatch
//   FOLD_7  compute  Performance        — EMA metrics, coherence report
//
// DirectX 12 fallback: WebGPU adapter info.backend === 'd3d12' activates
//   automatically via the same GPU device handle.
//
// WASM VM bridge: each fold shader is registered with WASMVMOrchestrator
//   which calls submitCompute/submitRender from inside the WASM binary.

// ─── Fold 0: Meta-orchestration ───────────────────────────────────────────────

export const FOLD0_WGSL = /* wgsl */`
struct Job {
  id: u32, type: u32, priority: f32,
  data_offset: u32, data_size: u32, phase_angle: f32, entropy: f32
};
struct FoldState {
  fold_id: u32, active_jobs: u32, utilization: f32,
  entropy: f32, phase: f32, coherence: f32
};
struct Constants { tzolkin_day: u32, time: f32, pi: f32, phase: f32 };

@group(0) @binding(0) var<storage, read_write> jobs:       array<Job>;
@group(0) @binding(1) var<storage, read_write> fold_states:array<FoldState>;
@group(0) @binding(2) var<storage, read_write> job_queue:  array<u32>;
@group(0) @binding(3) var<uniform>             constants:  Constants;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= arrayLength(&jobs)) { return; }
  var job = jobs[idx];

  // Mayan Tzolk'in phase
  let tzolkin_phase = f32(constants.tzolkin_day) * 2.0 * constants.pi / 260.0;
  let fold_phase    = f32(job.id) * constants.pi / 4.0;
  let total_phase   = tzolkin_phase + fold_phase;
  job.phase_angle   = total_phase;

  // Coherence with fold states
  var coherence_sum = 0.0;
  for (var f = 0u; f < 8u; f++) {
    let diff = abs(total_phase - fold_states[f].phase);
    coherence_sum += abs(cos(diff / 2.0));
  }
  job.entropy = job.entropy * (1.0 - coherence_sum / 16.0);

  // Route to most phase-aligned fold
  var best_fold = 0u; var best_c = 0.0;
  for (var f = 0u; f < 8u; f++) {
    let diff = abs(total_phase - fold_states[f].phase);
    let c    = abs(cos(diff / 2.0));
    if (c > best_c) { best_c = c; best_fold = f; }
  }
  job_queue[idx] = best_fold;
  jobs[idx] = job;
}
`;

// ─── Fold 1: Execution core ───────────────────────────────────────────────────

export const FOLD1_WGSL = /* wgsl */`
struct Opcode { code: u32, flags: u32, data: u32, result: f32 };
struct ExCtx  { stack_ptr: u32, frame_ptr: u32, pc: u32, phase: f32 };
struct Constants { pi: f32, load: f32, phase: f32, time: f32 };

@group(0) @binding(0) var<storage, read_write> opcodes: array<Opcode>;
@group(0) @binding(1) var<storage, read_write> context: array<ExCtx>;
@group(0) @binding(2) var<storage, read_write> stack:   array<f32>;
@group(0) @binding(3) var<uniform>             C:       Constants;

fn exec(op: Opcode, phase: f32) -> f32 {
  switch (op.code) {
    case 0x03u: { return f32(op.data); }                      // SEK assign
    case 0x04u: { return stack[op.data]; }                    // YAX get
    case 0x05u: { return stack[op.data] * phase; }            // WO call
    case 0x11u: { return stack[op.data] + stack[op.data+1u]; }// PIPE
    case 0x26u: { return stack[op.data] * 0.9; }              // FORWARD stub
    case 0x28u: { return -log(max(stack[op.data], 1e-7)); }   // LOSS
    default:    { return 0.0; }
  }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&opcodes)) { return; }
  var op  = opcodes[i];
  var ctx = context[i];
  let result = exec(op, ctx.phase) * sin(ctx.phase);
  op.result = result;
  ctx.pc   += 1u;
  ctx.phase += C.pi / 4.0;
  if (ctx.stack_ptr < arrayLength(&stack)) {
    stack[ctx.stack_ptr] = result;
    ctx.stack_ptr += 1u;
  }
  opcodes[i] = op;
  context[i] = ctx;
}
`;

// ─── Fold 2: UI render (vertex + fragment) ────────────────────────────────────

export const FOLD2_WGSL = /* wgsl */`
struct VertIn  { @location(0) pos: vec3<f32>, @location(1) col: vec3<f32>,
                 @location(2) uv: vec2<f32> };
struct VertOut { @builtin(position) position: vec4<f32>,
                 @location(0) col: vec3<f32>, @location(1) uv: vec2<f32> };
struct Uniforms { mvp: mat4x4<f32>, time: f32, phase: f32, entropy: f32 };

@group(0) @binding(0) var<uniform> U:      Uniforms;
@group(0) @binding(1) var          tex:    texture_2d<f32>;
@group(0) @binding(2) var          samp:   sampler;

@vertex fn vs_main(v: VertIn) -> VertOut {
  let s = sin(U.phase); let c = cos(U.phase);
  let rot = mat3x3<f32>(vec3(c,0.,s), vec3(0.,1.,0.), vec3(-s,0.,c));
  var out: VertOut;
  out.position = U.mvp * vec4(rot * v.pos, 1.0);
  out.col      = v.col * (0.8 + 0.2 * sin(U.time));
  out.uv       = v.uv;
  return out;
}

@fragment fn fs_main(in: VertOut) -> @location(0) vec4<f32> {
  let glyph = textureSample(tex, samp, in.uv);
  let alpha  = 1.0 - clamp(U.entropy, 0.0, 0.5);
  return vec4(mix(in.col, glyph.rgb, 0.7), alpha);
}
`;

// ─── Fold 3: SCXQ7 compression ────────────────────────────────────────────────

export const FOLD3_WGSL = /* wgsl */`
struct Glyph { code: u32, frequency: u32, compressed: u32, pattern_id: u32 };
struct Dict   { pattern: u32, replacement: u32, frequency: u32 };

@group(0) @binding(0) var<storage, read_write> glyphs: array<Glyph>;
@group(0) @binding(1) var<storage, read>       dict:   array<Dict>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

fn fnv1a(a: u32, b: u32, c: u32, d: u32) -> u32 {
  var h = 2166136261u;
  h = (h ^ a) * 16777619u; h = (h ^ b) * 16777619u;
  h = (h ^ c) * 16777619u; h = (h ^ d) * 16777619u;
  return h;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&glyphs)) { return; }
  var g = glyphs[i];
  let n = arrayLength(&glyphs);
  let g1 = select(0u, glyphs[i+1u].code, i+1u < n);
  let g2 = select(0u, glyphs[i+2u].code, i+2u < n);
  let g3 = select(0u, glyphs[i+3u].code, i+3u < n);
  let hash = fnv1a(g.code, g1, g2, g3);
  var matched = false;
  for (var d = 0u; d < arrayLength(&dict); d++) {
    if (dict[d].pattern == hash) {
      output[i] = dict[d].replacement;
      g.compressed = dict[d].replacement;
      g.pattern_id = d;
      matched = true;
      break;
    }
  }
  if (!matched) { output[i] = g.code; g.compressed = g.code; g.pattern_id = 0xFFFFFFFFu; }
  glyphs[i] = g;
}
`;

// ─── Fold 4: Game world physics ───────────────────────────────────────────────

export const FOLD4_WGSL = /* wgsl */`
struct Vtx { position: vec3<f32>, normal: vec3<f32>, color: vec3<f32>, velocity: vec3<f32> };
struct World { time: f32, gravity: f32, quantum_seed: u32, entropy: f32, phase: f32 };

@group(0) @binding(0) var<storage, read_write> verts:      array<Vtx>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(2) var<uniform>             world:      World;

fn noise(p: vec3<f32>) -> f32 {
  return sin(p.x*10.)*cos(p.y*10.)*sin(p.z*10.);
}

@compute @workgroup_size(64)
fn update_physics(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&verts)) { return; }
  var v = verts[i]; var vel = velocities[i];
  vel.y -= world.gravity * 0.016;

  // Quantum fluctuation
  let qs  = world.quantum_seed + i;
  let qx  = f32((qs*1103515245u+12345u)&0x7FFFFFFFu)/2147483647.0;
  let qy  = f32((qs*1103515245u+67890u)&0x7FFFFFFFu)/2147483647.0;
  let qz  = f32((qs*1103515245u+11111u)&0x7FFFFFFFu)/2147483647.0;
  vel    += vec3(qx,qy,qz)*0.1;

  v.position += vel * 0.016;

  // Boundary
  if (abs(v.position.x) > 100.) { vel.x = -vel.x*0.8; v.position.x = sign(v.position.x)*100.; }
  if (v.position.y < 0.)        { vel.y = -vel.y*0.6; v.position.y = 0.; }
  if (abs(v.position.z) > 100.) { vel.z = -vel.z*0.8; v.position.z = sign(v.position.z)*100.; }

  v.color = vec3(0.5+noise(v.position)*0.5, 0.3+sin(v.position.y*0.5)*0.3,
                 0.7+cos(v.position.z*0.5)*0.3) * (1.-world.entropy);
  verts[i]      = v;
  velocities[i] = vel;
}
`;

// ─── Fold 5: Security audit ───────────────────────────────────────────────────

export const FOLD5_WGSL = /* wgsl */`
struct SecCtx  { user_id: u32, permissions: u32, entropy: f32, risk_score: f32 };
struct AuditEntry { timestamp: u32, event: u32, user_id: u32, result: u32, hash: u32 };
struct Constants   { time: f32, phase: f32, pi: f32, load: f32 };

@group(0) @binding(0) var<storage, read_write> contexts: array<SecCtx>;
@group(0) @binding(1) var<storage, read_write> audit:    array<AuditEntry>;
@group(0) @binding(2) var<uniform>             C:        Constants;

fn audit_hash(ctx: SecCtx, ts: u32) -> u32 {
  var h = ts ^ ctx.user_id; h = h * 0x9E3779B1u;
  h = h ^ ctx.permissions;  h = h * 0x9E3779B1u;
  return h ^ u32(ctx.risk_score * 1e6);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&contexts)) { return; }
  var ctx = contexts[i];
  ctx.risk_score = ctx.entropy * (1. - sin(C.time*2.)*0.5);
  let phase_check = (u32(C.phase*100.) & 0xFFu) == ((ctx.user_id*7u) & 0xFFu);
  var entry: AuditEntry;
  entry.timestamp = u32(C.time*1000.);
  entry.event     = 1u;
  entry.user_id   = ctx.user_id;
  entry.result    = select(0u, 1u, phase_check);
  entry.hash      = audit_hash(ctx, entry.timestamp);
  if (!phase_check) { ctx.risk_score += 0.5; }
  contexts[i] = ctx;
  audit[i]    = entry;
}
`;

// ─── Fold 6: API routing ──────────────────────────────────────────────────────

export const FOLD6_WGSL = /* wgsl */`
struct Endpoint { id: u32, method: u32, path_hash: u32, handler_id: u32, latency: f32 };
struct Request  { id: u32, endpoint_id: u32, phase: f32, priority: f32, result: f32 };
struct Constants { pi: f32, load: f32, phase: f32, time: f32 };

@group(0) @binding(0) var<storage, read_write> endpoints: array<Endpoint>;
@group(0) @binding(1) var<storage, read_write> requests:  array<Request>;
@group(0) @binding(2) var<uniform>             C:         Constants;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= arrayLength(&requests)) { return; }
  var req = requests[i];
  var best_e = 0u; var best_m = 0.0;
  for (var e = 0u; e < arrayLength(&endpoints); e++) {
    let diff  = abs(req.phase - f32(endpoints[e].id) * C.pi / 4.0);
    let match = 1.0 - diff / C.pi;
    if (match > best_m) { best_m = match; best_e = e; }
  }
  req.endpoint_id = best_e;
  let diff = abs(req.phase - f32(endpoints[best_e].id) * C.pi / 4.0);
  requests[i] = req;
}
`;

// ─── Fold 7: Performance monitoring ──────────────────────────────────────────

export const FOLD7_WGSL = /* wgsl */`
struct Metric  { fold_id: u32, utilization: f32, latency: f32, throughput: f32, entropy: f32 };
struct Report  { timestamp: u32, avg_util: f32, avg_lat: f32, total_tput: f32, coherence: f32 };
struct Constants { pi: f32, load: f32, phase: f32, time: f32 };

@group(0) @binding(0) var<storage, read_write> metrics: array<Metric>;
@group(0) @binding(1) var<storage, read_write> report:  Report;
@group(0) @binding(2) var<uniform>             C:       Constants;

fn ema(cur: f32, nw: f32, a: f32) -> f32 { return cur*(1.-a) + nw*a; }

@compute @workgroup_size(8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let f = id.x; if (f >= 8u) { return; }
  var m = metrics[f];
  m.utilization = clamp(m.utilization + (C.load-0.5)*0.1, 0., 1.);
  m.latency     = m.latency * (1. + (C.phase - m.entropy)*0.1);
  m.throughput  = m.throughput * (1. + (1.-m.latency/100.));
  m.entropy     = ema(m.entropy, C.phase, 0.1);
  metrics[f]    = m;
  if (f == 0u) {
    var tu=0.; var tl=0.; var tt=0.; var tc=0.;
    for (var k=0u; k<8u; k++) {
      let mk = metrics[k];
      tu += mk.utilization; tl += mk.latency; tt += mk.throughput;
      tc += abs(cos((f32(k)*C.pi/4. - C.phase)/2.));
    }
    report.timestamp  = u32(C.time*1000.);
    report.avg_util   = tu/8.; report.avg_lat = tl/8.;
    report.total_tput = tt; report.coherence = tc/8.;
  }
}
`;

// ─── Shader registry ──────────────────────────────────────────────────────────

export const FOLD_SHADERS = Object.freeze({
  0: { wgsl: FOLD0_WGSL, type: 'compute', label: 'fold_0_meta',        entry: 'main' },
  1: { wgsl: FOLD1_WGSL, type: 'compute', label: 'fold_1_execution',   entry: 'main' },
  2: { wgsl: FOLD2_WGSL, type: 'render',  label: 'fold_2_ui',          vs: 'vs_main', fs: 'fs_main' },
  3: { wgsl: FOLD3_WGSL, type: 'compute', label: 'fold_3_compression', entry: 'main' },
  4: { wgsl: FOLD4_WGSL, type: 'compute', label: 'fold_4_game',        entry: 'update_physics' },
  5: { wgsl: FOLD5_WGSL, type: 'compute', label: 'fold_5_security',    entry: 'main' },
  6: { wgsl: FOLD6_WGSL, type: 'compute', label: 'fold_6_api',         entry: 'main' },
  7: { wgsl: FOLD7_WGSL, type: 'compute', label: 'fold_7_perf',        entry: 'main' },
});

// ─── Lightweight orchestrator (browser + Node.js) ─────────────────────────────

export class FoldOrchestrator {
  constructor() {
    this._device   = null;
    this._backend  = null;
    this._pipelines = new Map();
  }

  async init(canvas = null) {
    if (typeof navigator !== 'undefined' && navigator.gpu) {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        this._device  = await adapter.requestDevice();
        this._backend = adapter.info?.backend ?? 'webgpu';
        return true;
      }
    }
    this._backend = 'software';
    return false;
  }

  async compileFold(foldId) {
    if (!this._device) return null;
    const spec = FOLD_SHADERS[foldId];
    const mod  = this._device.createShaderModule({ code: spec.wgsl, label: spec.label });
    if (spec.type === 'compute') {
      const p = await this._device.createComputePipelineAsync({
        layout: 'auto', compute: { module: mod, entryPoint: spec.entry }
      });
      this._pipelines.set(foldId, { mod, pipeline: p, type: 'compute' });
    } else {
      const p = await this._device.createRenderPipelineAsync({
        layout: 'auto',
        vertex:   { module: mod, entryPoint: spec.vs },
        fragment: { module: mod, entryPoint: spec.fs, targets: [{ format: 'bgra8unorm' }] },
        primitive: { topology: 'triangle-list' },
      });
      this._pipelines.set(foldId, { mod, pipeline: p, type: 'render' });
    }
    return this._pipelines.get(foldId);
  }

  get backend() { return this._backend; }
  pipeline(foldId) { return this._pipelines.get(foldId); }
}
