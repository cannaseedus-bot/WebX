/**
 * trainer-server.js — WebX-3D Native Trainer HTTP Bridge  v2
 *
 * Training modes:
 *   causal_gpu      gpt2_trainer.exe  raw next-token, D3D11 GPU Adam
 *   shard_chain     gpt2_trainer.exe  iterate .scxqdds shards sequentially
 *   toolcall_cpu    finetune_instruct.py  Alpaca template + grad_accum, CPU
 *   curriculum_cpu  finetune_instruct.py  easy→hard curriculum, CPU
 *   glyph_pretrain  finetune_instruct.py  KXML glyph prefix records, CPU
 *   fiber_chain     gpt2_trainer.exe  multi-checkpoint chain across chunks
 *
 * API:
 *   POST /train/start   { mode, model, tokens, shard, steps, lr, batch, block,
 *                         gradAccum, saveEvery, logEvery, dataChunks, out, resume }
 *   POST /train/stop
 *   GET  /train/status
 *   GET  /train/stream  (SSE)
 *   GET  /models
 *   GET  /tokens
 *   GET  /shards
 *   GET  /modes         (mode definitions)
 *
 * Start from native/:
 *   node trainer-server.js
 */

'use strict';

const http      = require('http');
const path      = require('path');
const fs        = require('fs');
const { spawn } = require('child_process');

const PORT        = 7431;
const NATIVE_DIR  = __dirname;
const BIN_DIR     = path.join(NATIVE_DIR, 'bin');
const TRAINER_EXE = path.join(BIN_DIR, 'gpt2_trainer.exe');

// Python paths (tried in order)
const PYTHON_CANDIDATES = [
    'C:\\Users\\canna\\AppData\\Local\\Programs\\Python\\Python312\\python.exe',
    'python',
    'python3',
];
const FINETUNE_PY   = 'E:\\models\\GPT2\\med-GPT\\finetune_instruct.py';
const HYBRID_PY     = path.join(NATIVE_DIR, '..', 'tools', 'trainers', 'hybrid_math_micronaut.py');
const TRAINER_TOOLS = path.join(NATIVE_DIR, '..', 'tools', 'trainers');

// ── Mode definitions ──────────────────────────────────────────────────────────

const MODES = {
    causal_gpu: {
        label:       'Causal GPU (D3D11)',
        engine:      'exe',
        description: 'Raw next-token prediction on iGPU via D3D11 + GPU Adam. Fast. Best for dense code/token pretraining.',
        micronauts:  ['coder', 'base'],
        params:      ['model', 'tokens', 'out', 'steps', 'batch', 'block', 'lr', 'saveEvery'],
        defaults:    { steps: 1000, batch: 4, block: 128, lr: 3e-5, saveEvery: 200 },
    },
    shard_chain: {
        label:       'Shard Chain (DDS stream)',
        engine:      'exe',
        description: 'Streams weight updates from .scxqdds shard files. Use for large datasets that exceed VRAM. Chains checkpoint → next shard automatically.',
        micronauts:  ['base', 'coder', 'any'],
        params:      ['model', 'shard', 'tokens', 'out', 'steps', 'batch', 'block', 'lr', 'saveEvery'],
        defaults:    { steps: 200, batch: 4, block: 128, lr: 3e-5, saveEvery: 200 },
    },
    toolcall_cpu: {
        label:       'Tool-Call Finetune (CPU)',
        engine:      'py',
        description: 'Alpaca ### Instruction / ### Response template. Grad accumulation for effective larger batch. Best for TC-1/TC-2 base tool-call models.',
        micronauts:  ['toolcall_small', 'toolcall_medium', 'agent', 'powershell'],
        params:      ['resume', 'out', 'steps', 'lr', 'batch', 'gradAccum', 'block', 'saveEvery', 'logEvery', 'dataChunks'],
        defaults:    { steps: 500, lr: 1e-5, batch: 1, gradAccum: 8, block: 128, saveEvery: 100, logEvery: 10, dataChunks: 10 },
    },
    curriculum_cpu: {
        label:       'Curriculum CPU (easy→hard)',
        engine:      'py',
        description: 'Trains on easy_short records first, progresses to hard_long. Reduces catastrophic forgetting. Best for math_micronaut where reasoning builds incrementally.',
        micronauts:  ['math'],
        params:      ['resume', 'out', 'steps', 'lr', 'batch', 'gradAccum', 'block', 'saveEvery', 'logEvery', 'dataChunks'],
        defaults:    { steps: 1500, lr: 1e-5, batch: 1, gradAccum: 8, block: 128, saveEvery: 100, logEvery: 10, dataChunks: 5 },
        curriculum:  ['easy_short', 'easy_medium', 'medium', 'hard', 'hard_long'],
    },
    glyph_pretrain: {
        label:       'Glyph Pretrain (KXML)',
        engine:      'py',
        description: 'Records have ⟁ KXML fold prefix. Trains model to associate glyph tokens with structured JSON execution. Best for kuhul_micronaut and KXML semantic layer.',
        micronauts:  ['kuhul', 'agent'],
        params:      ['resume', 'out', 'steps', 'lr', 'batch', 'gradAccum', 'block', 'saveEvery', 'logEvery'],
        defaults:    { steps: 500, lr: 2e-5, batch: 1, gradAccum: 4, block: 128, saveEvery: 100, logEvery: 10 },
    },
    fiber_chain: {
        label:       'Fiber Chain (multi-shard)',
        engine:      'fiber',
        description: 'Orchestrated multi-step run: splits token data into chunks, trains each chunk sequentially, passing checkpoint forward. Mirrors the v0.1.1-igpu-trainer-xjsl training loop.',
        micronauts:  ['coder', 'base', 'any'],
        params:      ['model', 'tokens', 'out', 'steps', 'batch', 'block', 'lr', 'saveEvery', 'chunkSize'],
        defaults:    { steps: 200, batch: 4, block: 128, lr: 3e-5, saveEvery: 200, chunkSize: 44000 },
    },
    hybrid: {
        label:       'Hybrid iGPU→CPU (math/code)',
        engine:      'hybrid',
        description: '[Sek] D3D11 iGPU pretrain on KXML domain tokens → [Ch\'en] CPU finetune with geodesic+ARC attention. This is the pipeline that broke through 3.5–4.0 plateau. Loads existing geodesic cache from E:/models/GPT2/geodesic_cache/.',
        micronauts:  ['math', 'coder'],
        params:      ['model', 'jsonl', 'domain', 'block', 'gpuSteps', 'gpuLr', 'gpuBatch', 'cpuSteps', 'cpuLr', 'cpuBatch', 'geodesic'],
        defaults:    { gpuSteps: 500, gpuLr: 3e-5, gpuBatch: 4, cpuSteps: 500, cpuLr: 5e-5, cpuBatch: 2, block: 128 },
    },
};

// ── State ─────────────────────────────────────────────────────────────────────

let proc     = null;
let lossLog  = [];
let clients  = [];
let running  = false;
let lastStep = 0;
let lastLoss = 0;
let startTs  = null;
let curMode  = null;
let curModel = null;

// ── SSE ───────────────────────────────────────────────────────────────────────

function sseInit(res) {
    res.writeHead(200, {
        'Content-Type':  'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection':    'keep-alive',
        'Access-Control-Allow-Origin': '*',
    });
    res.write(':ok\n\n');
    clients.push(res);
    res.on('close', () => { clients = clients.filter(c => c !== res); });
}

function sseBroadcast(event, data) {
    const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
    clients.forEach(c => { try { c.write(msg); } catch {} });
}

// ── Output parser ─────────────────────────────────────────────────────────────

function parseLine(line) {
    // GPU trainer:  [trainer] step=42 loss=3.1234
    // Python trainer:  step=  42/500  loss=3.1234
    const m1 = line.match(/\[trainer\]\s+step=(\d+)\s+loss=([0-9.eE+\-]+)/);
    const m2 = line.match(/step=\s*(\d+)\/\d+\s+loss=([0-9.eE+\-]+)/);
    const m  = m1 || m2;
    if (m) {
        const step = parseInt(m[1], 10);
        const loss = parseFloat(m[2]);
        lastStep = step; lastLoss = loss;
        const entry = { step, loss, ts: Date.now() };
        lossLog.push(entry);
        sseBroadcast('step', entry);
    }
    const savedM = line.match(/saved[:\s]+(.+\.safetensors)/i);
    if (savedM) sseBroadcast('checkpoint', { path: savedM[1].trim(), step: lastStep, loss: lastLoss });
    if (/\[main\]\s+done|training complete|final model/.test(line)) {
        running = false;
        sseBroadcast('done', { step: lastStep, loss: lastLoss });
    }
    if (/init failed|ERROR|error/.test(line)) {
        sseBroadcast('error', { message: line.trim() });
        if (/init failed/.test(line)) running = false;
    }
    sseBroadcast('log', { line: line.trim() });
}

function attach(child) {
    const onData = buf => buf.toString().split('\n').filter(Boolean).forEach(parseLine);
    child.stdout.on('data', onData);
    child.stderr.on('data', onData);
    child.on('close', code => {
        running = false; proc = null;
        sseBroadcast('close', { code, step: lastStep, loss: lastLoss });
    });
}

// ── Python executable ─────────────────────────────────────────────────────────

function findPython() {
    for (const p of PYTHON_CANDIDATES) {
        try { if (fs.existsSync(p)) return p; } catch {}
    }
    return PYTHON_CANDIDATES[PYTHON_CANDIDATES.length - 1];
}

// ── Mode launchers ────────────────────────────────────────────────────────────

function launchCasualGPU(cfg) {
    const args = [
        '--model', cfg.model,
        '--steps', String(cfg.steps || 1000),
        '--batch', String(cfg.batch || 4),
        '--block', String(cfg.block || 128),
        '--lr',    String(cfg.lr    || 3e-5),
        '--save-every', String(cfg.saveEvery || 200),
    ];
    if (cfg.tokens) args.push('--data', cfg.tokens);
    if (cfg.out)    args.push('--out',  cfg.out);
    return spawn(TRAINER_EXE, args, { cwd: BIN_DIR });
}

function launchShardChain(cfg) {
    const args = [
        '--model', cfg.model,
        '--steps', String(cfg.steps || 200),
        '--batch', String(cfg.batch || 4),
        '--block', String(cfg.block || 128),
        '--lr',    String(cfg.lr    || 3e-5),
        '--save-every', String(cfg.saveEvery || 200),
    ];
    if (cfg.tokens) args.push('--data',  cfg.tokens);
    if (cfg.shard)  args.push('--shard', cfg.shard);
    if (cfg.out)    args.push('--out',   cfg.out);
    return spawn(TRAINER_EXE, args, { cwd: BIN_DIR });
}

function launchPython(scriptPath, cfg, extra = []) {
    const python = findPython();
    const args = [
        scriptPath,
        '--steps',      String(cfg.steps      || 500),
        '--lr',         String(cfg.lr         || 1e-5),
        '--batch',      String(cfg.batch      || 1),
        '--grad-accum', String(cfg.gradAccum  || 8),
        '--block-size', String(cfg.block      || 128),
        '--save-every', String(cfg.saveEvery  || 100),
        '--log-every',  String(cfg.logEvery   || 10),
        ...extra,
    ];
    if (cfg.resume)     args.push('--resume',      cfg.resume);
    if (cfg.dataChunks) args.push('--data-chunks', String(cfg.dataChunks));
    const cwd = path.dirname(scriptPath);
    return spawn(python, args, { cwd });
}

// fiber_chain: orchestrate multiple gpt2_trainer.exe runs
async function launchFiberChain(cfg) {
    const { model, tokens, out, steps, batch, block, lr, saveEvery, chunkSize } = cfg;
    if (!tokens || !fs.existsSync(tokens)) {
        sseBroadcast('error', { message: `fiber_chain: token file not found: ${tokens}` });
        return null;
    }

    const tmpDir   = path.join(BIN_DIR, 'fiber_tmp');
    fs.mkdirSync(tmpDir, { recursive: true });
    const outDir   = out ? path.dirname(out) : path.join(tmpDir, 'checkpoints');
    fs.mkdirSync(outDir, { recursive: true });

    const tokenBytes = fs.statSync(tokens).size;
    const seqLen     = block || 128;
    const seqBytes   = seqLen * 2;                   // uint16 per token
    const totalSeqs  = Math.floor(tokenBytes / seqBytes);
    const chunkSeqs  = chunkSize || 44000;
    const numChunks  = Math.ceil(totalSeqs / chunkSeqs);

    sseBroadcast('log', { line: `[fiber_chain] ${totalSeqs} seqs → ${numChunks} chunks @ ${chunkSeqs}/chunk` });

    let currentModel = model;
    let globalStep   = 0;

    for (let ci = 0; ci < numChunks; ci++) {
        const chunkOffset = ci * chunkSeqs * seqBytes;
        const chunkSize_  = Math.min(chunkSeqs, totalSeqs - ci * chunkSeqs) * seqBytes;
        const chunkBin    = path.join(tmpDir, `chunk_${ci}.bin`);
        const chunkOut    = path.join(outDir, `fiber_c${String(ci).padStart(3,'0')}.safetensors`);

        // Slice token file
        const fd  = fs.openSync(tokens, 'r');
        const buf = Buffer.alloc(chunkSize_);
        fs.readSync(fd, buf, 0, chunkSize_, chunkOffset);
        fs.closeSync(fd);
        fs.writeFileSync(chunkBin, buf);

        sseBroadcast('log', { line: `[fiber_chain] chunk ${ci+1}/${numChunks} → ${chunkOut}` });

        const args = [
            '--model', currentModel,
            '--data',  chunkBin,
            '--out',   chunkOut,
            '--steps', String(steps || 200),
            '--batch', String(batch || 4),
            '--block', String(block || 128),
            '--lr',    String(lr    || 3e-5),
            '--save-every', String(saveEvery || 200),
        ];

        await new Promise((resolve, reject) => {
            const child = spawn(TRAINER_EXE, args, { cwd: BIN_DIR });
            attach(child);
            child.on('close', code => {
                if (code !== 0) reject(new Error(`chunk ${ci} exit ${code}`));
                else resolve();
            });
        }).catch(e => sseBroadcast('error', { message: e.message }));

        currentModel = chunkOut;
        globalStep  += steps || 200;

        if (!running) break;  // stop requested
    }

    running = false;
    sseBroadcast('done', { step: globalStep, loss: lastLoss, mode: 'fiber_chain' });
    return null;  // already managed
}

// hybrid: iGPU→CPU pipeline via hybrid_math_micronaut.py
function launchHybrid(cfg) {
    const python = findPython();
    const args = [
        HYBRID_PY,
        '--model',      cfg.model      || '',
        '--jsonl',      cfg.jsonl      || 'E:/models/GPT2/domain_train.jsonl',
        '--domain',     cfg.domain     || 'math',
        '--block',      String(cfg.block      || 128),
        '--gpu-steps',  String(cfg.gpuSteps   || 500),
        '--gpu-lr',     String(cfg.gpuLr      || 3e-5),
        '--gpu-batch',  String(cfg.gpuBatch   || 4),
        '--cpu-steps',  String(cfg.cpuSteps   || 500),
        '--cpu-lr',     String(cfg.cpuLr      || 5e-5),
        '--cpu-batch',  String(cfg.cpuBatch   || 2),
        '--log-every',  String(cfg.logEvery   || 25),
        '--ckpt-every', String(cfg.saveEvery  || 100),
        '--chunk-steps',String(cfg.chunkSteps || 250),
    ];
    if (cfg.noGeodesic) args.push('--no-geodesic');
    return spawn(python, args, { cwd: TRAINER_TOOLS });
}

// ── Start / stop ──────────────────────────────────────────────────────────────

async function startTrainer(cfg) {
    if (running) return { ok: false, error: 'already running' };

    const mode = cfg.mode || 'causal_gpu';
    if (!MODES[mode]) return { ok: false, error: `unknown mode: ${mode}` };

    lossLog  = [];
    running  = true;
    startTs  = Date.now();
    lastStep = 0; lastLoss = 0;
    curMode  = mode; curModel = cfg.model;

    sseBroadcast('start', { mode, model: cfg.model, steps: cfg.steps });

    try {
        if (mode === 'causal_gpu') {
            proc = launchCasualGPU(cfg);
            attach(proc);
        } else if (mode === 'shard_chain') {
            proc = launchShardChain(cfg);
            attach(proc);
        } else if (mode === 'toolcall_cpu' || mode === 'glyph_pretrain') {
            proc = launchPython(FINETUNE_PY, cfg);
            attach(proc);
        } else if (mode === 'curriculum_cpu') {
            // Sort data by curriculum bucket before launching
            proc = launchPython(FINETUNE_PY, cfg, ['--curriculum']);
            attach(proc);
        } else if (mode === 'fiber_chain') {
            proc = null;
            launchFiberChain(cfg);
        } else if (mode === 'hybrid') {
            proc = launchHybrid(cfg);
            attach(proc);
        }
    } catch (e) {
        running = false;
        return { ok: false, error: e.message };
    }

    return { ok: true, mode, model: cfg.model };
}

function stopTrainer() {
    running = false;
    if (proc) { try { proc.kill('SIGTERM'); } catch {} proc = null; }
    sseBroadcast('log', { line: '[trainer-server] stopped by user' });
    return { ok: true };
}

// ── Model / token / shard discovery ─────────────────────────────────────────

const MODEL_DIRS = [
    'E:/models/GPT2/small-instruct',
    'E:/models/GPT2/small-instruct/checkpoints',
    'E:/models/GPT2/mini-GPT',
    'E:/models/GPT2/med-GPT',
    'E:/models/GPT2/math_micronaut',
    'E:/models/GPT2/math_micronaut/fiber_00',
    'E:/models/GPT2/math_micronaut/fiber_01',
    'E:/models/GPT2/coder_micronaut/dx11',
    'C:/Users/canna/.kuhul-v1/releases/v3.5.0-WebX/native/bin',
    'C:/Users/canna/.micronaut/models',
];

const TOKEN_DIRS = [
    'E:/models/GPT2/med-GPT',
    'E:/models/GPT2/mini-GPT',
    'E:/models/GPT2/coder_micronaut',
    'E:/models/GPT2/small-instruct/data',
    'E:/models/GPT2',
    'C:/Users/canna/.kuhul-v1/releases/v3.5.0-WebX/native/bin',
];

const SHARD_DIRS = [
    'E:/models/GPT2/small-instruct/shards',
    'C:/Users/canna/.kuhul-v1/releases/v3.5.0-WebX/native/bin',
];

function scanFiles(dirs, ext, extraInfo) {
    const out = [];
    for (const dir of dirs) {
        try {
            fs.readdirSync(dir).forEach(f => {
                if (f.endsWith(ext)) {
                    const p    = path.join(dir, f).replace(/\\/g, '/');
                    const stat = fs.statSync(p);
                    const entry = { label: f, path: p, mb: Math.round(stat.size / 1024 / 1024) };
                    if (extraInfo) Object.assign(entry, extraInfo(f, p, stat));
                    out.push(entry);
                }
            });
        } catch {}
    }
    return out;
}

// ── HTTP ──────────────────────────────────────────────────────────────────────

function json(res, status, obj) {
    const body = JSON.stringify(obj);
    res.writeHead(status, {
        'Content-Type':                'application/json',
        'Access-Control-Allow-Origin': '*',
    });
    res.end(body);
}

function readBody(req) {
    return new Promise((resolve, reject) => {
        let data = '';
        req.on('data', d => data += d);
        req.on('end',  () => {
            try { resolve(data ? JSON.parse(data) : {}); }
            catch (e) { reject(e); }
        });
    });
}

const server = http.createServer(async (req, res) => {
    const url = req.url.split('?')[0];

    if (req.method === 'OPTIONS') {
        res.writeHead(204, {
            'Access-Control-Allow-Origin':  '*',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        });
        return res.end();
    }

    if (url === '/train/stream' && req.method === 'GET')  return sseInit(res);

    if (url === '/train/start'  && req.method === 'POST') {
        const body = await readBody(req).catch(() => ({}));
        const result = await startTrainer(body);
        return json(res, result.ok ? 200 : 409, result);
    }
    if (url === '/train/stop'   && req.method === 'POST') return json(res, 200, stopTrainer());

    if (url === '/train/status' && req.method === 'GET') {
        return json(res, 200, {
            running, mode: curMode, model: curModel,
            step: lastStep, loss: lastLoss,
            elapsed: startTs ? Math.floor((Date.now() - startTs) / 1000) : 0,
            lossLog: lossLog.slice(-300),
        });
    }

    if (url === '/modes'  && req.method === 'GET') return json(res, 200, MODES);
    if (url === '/models' && req.method === 'GET') return json(res, 200, scanFiles(MODEL_DIRS, '.safetensors'));
    if (url === '/tokens' && req.method === 'GET') return json(res, 200, scanFiles(TOKEN_DIRS, '.bin'));
    if (url === '/shards' && req.method === 'GET') return json(res, 200, scanFiles(SHARD_DIRS, '.scxqdds'));

    json(res, 404, { error: 'not found' });
});

server.listen(PORT, '127.0.0.1', () => {
    console.log(`[trainer-server] v2 — port ${PORT}`);
    console.log(`  modes: ${Object.keys(MODES).join(' | ')}`);
    console.log(`  GPU exe: ${TRAINER_EXE}`);
    console.log(`  CPU py:  ${FINETUNE_PY}`);
    console.log(`  python:  ${findPython()}`);
});
