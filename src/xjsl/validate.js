// XJSL schema validator — pure in-memory (no fs).
// Port of tools/validate-xjsl.mjs from v0.1.1-igpu-trainer-xjsl.
// validateXJSLDoc(doc) returns { ok: boolean, errors: string[] }.

const VALID_SHADER_TYPES  = new Set(['compute']);
const VALID_BUFFER_LAYOUTS = new Set(['read', 'write', 'read_write']);

function checkShader(name, shader, errors) {
  if (shader['@type'] !== 'compute') {
    errors.push(`${name}: @type must be "compute" (got ${JSON.stringify(shader['@type'])})`);
  }

  const wg = shader['@workgroup'];
  if (!Array.isArray(wg) || wg.length !== 3 || wg.some(v => !Number.isInteger(v) || v < 1)) {
    errors.push(`${name}: @workgroup must be [x,y,z] with positive integers`);
  }

  const kernel = shader['@kernel'];
  if (typeof kernel !== 'string' || kernel.trim() === '') {
    errors.push(`${name}: @kernel must be a non-empty string`);
  }

  for (const [bufName, def] of Object.entries(shader['@inputs'] || {})) {
    if (!def['@type']) errors.push(`${name}.@inputs.${bufName}: missing @type`);
  }

  for (const [bufName, def] of Object.entries(shader['@outputs'] || {})) {
    if (!def['@type']) errors.push(`${name}.@outputs.${bufName}: missing @type`);
    if (def['@layout'] && !VALID_BUFFER_LAYOUTS.has(def['@layout'])) {
      errors.push(`${name}.@outputs.${bufName}: @layout must be one of ${[...VALID_BUFFER_LAYOUTS].join('|')}`);
    }
  }

  for (const [uName, uType] of Object.entries(shader['@uniforms'] || {})) {
    if (typeof uType !== 'string') {
      errors.push(`${name}.@uniforms.${uName}: type must be a string`);
    }
  }
}

export function validateXJSLDoc(doc) {
  const errors = [];

  if (!doc || typeof doc !== 'object') {
    return { ok: false, errors: ['XJSL document must be an object'] };
  }

  if (!doc['@shaders'] || typeof doc['@shaders'] !== 'object' || Array.isArray(doc['@shaders'])) {
    errors.push('XJSL document must contain an @shaders object');
    return { ok: false, errors };
  }

  const shaderEntries = Object.entries(doc['@shaders']);
  if (shaderEntries.length === 0) {
    errors.push('@shaders must contain at least one shader definition');
  }

  for (const [name, shader] of shaderEntries) {
    if (!shader || typeof shader !== 'object') {
      errors.push(`${name}: shader definition must be an object`);
      continue;
    }
    checkShader(name, shader, errors);
  }

  return { ok: errors.length === 0, errors };
}

// Validate and return a summary string (for CLI / logging use).
export function validateXJSLDocOrThrow(doc) {
  const result = validateXJSLDoc(doc);
  if (!result.ok) throw new Error(`XJSL validation failed:\n  ${result.errors.join('\n  ')}`);
  return `XJSL validation passed: ${Object.keys(doc['@shaders']).length} shader(s)`;
}

export default validateXJSLDoc;
