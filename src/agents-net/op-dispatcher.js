// Agents.NET @op dispatch table (Agents.NET.v1.0.0)
//
// Mirrors dotnet-bridge/Dispatcher.java and the HttpWorker POST /run route.
// The @op field in a request payload selects the handler in the .NET worker.
// This module captures the protocol so the JS side can validate ops before
// posting them to the HttpWorker endpoint.

export const DOTNET_OPS = Object.freeze([
  'DOTNET_RUN',
  'DOTNET_MATH_ADD',
  'DOTNET_SIMD_DOT',
  'DOTNET_TENSOR_MATMUL',
  'DOTNET_GPU_INFO',
]);

export const DOTNET_OP_SCHEMAS = Object.freeze({
  DOTNET_MATH_ADD:      { required: ['a', 'b'] },
  DOTNET_SIMD_DOT:      { required: ['vec_a', 'vec_b'] },
  DOTNET_TENSOR_MATMUL: { required: ['a_rows', 'a_cols', 'b_cols', 'data_a', 'data_b'] },
  DOTNET_GPU_INFO:      { required: [] },
  DOTNET_RUN:           { required: ['payload'] },
});

export const DOTNET_WORKER_URL_DEFAULT = 'http://localhost:5010/run';
export const DOTNET_WORKER_HEALTH_PATH = '/health';

export function validateOp(op) {
  if (!op || typeof op['@op'] !== 'string') return { ok: false, error: 'missing @op field' };
  const code = op['@op'];
  if (!DOTNET_OPS.includes(code)) return { ok: false, error: `unknown op: ${code}` };
  const schema = DOTNET_OP_SCHEMAS[code];
  for (const field of schema.required) {
    if (op[field] === undefined) return { ok: false, error: `op ${code} missing required field: ${field}` };
  }
  return { ok: true };
}

// Fetch-based bridge (browser + Node 18+) — replaces dotnet_http_bridge.js http.request
export async function dispatchOp(op, url = DOTNET_WORKER_URL_DEFAULT) {
  const v = validateOp(op);
  if (!v.ok) return { error: v.error };
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(op),
  });
  return res.json();
}
