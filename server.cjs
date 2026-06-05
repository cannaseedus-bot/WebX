/**
 * server.cjs — KUHUL WebX-3D static file server
 * Serves ./ on port 7430 with COOP/COEP headers required for SharedArrayBuffer.
 * Run via START.bat or: node server.cjs
 */
'use strict';

const http = require('http');
const fs   = require('fs');
const path = require('path');
const url  = require('url');

const PORT = 7430;
const ROOT = __dirname;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript',
  '.mjs':  'application/javascript',
  '.cjs':  'application/javascript',
  '.json': 'application/json',
  '.css':  'text/css',
  '.md':   'text/plain; charset=utf-8',
  '.hlsl': 'text/plain',
  '.wgsl': 'text/plain',
  '.kuhul':'text/plain',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
  '.bin':  'application/octet-stream',
};

function mime(file) {
  return MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

const server = http.createServer((req, res) => {
  const parsed  = url.parse(req.url);
  let   reqPath = decodeURIComponent(parsed.pathname);

  // Route /app → src/index.html
  if (reqPath === '/app' || reqPath === '/app/') reqPath = '/src/index.html';
  // Default index
  if (reqPath === '/' || reqPath === '') reqPath = '/index.html';

  const filePath = path.join(ROOT, reqPath);

  // Security: stay within ROOT
  if (!filePath.startsWith(ROOT)) {
    res.writeHead(403); return res.end('Forbidden');
  }

  // API endpoints
  if (reqPath === '/api/manifest') {
    const f = path.join(ROOT, 'server.manifest.json');
    if (fs.existsSync(f)) {
      res.writeHead(200, { 'Content-Type': 'application/json', ...SECURITY_HEADERS });
      return res.end(fs.readFileSync(f));
    }
  }
  if (reqPath === '/api/cache') {
    const f = path.join(ROOT, 'cache.manifest.json');
    if (fs.existsSync(f)) {
      res.writeHead(200, { 'Content-Type': 'application/json', ...SECURITY_HEADERS });
      return res.end(fs.readFileSync(f));
    }
  }

  if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
    // Try index.html in directory
    const idx = path.join(filePath, 'index.html');
    if (fs.existsSync(idx)) {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', ...SECURITY_HEADERS });
      return res.end(fs.readFileSync(idx));
    }
    res.writeHead(404, SECURITY_HEADERS);
    return res.end(`404 — ${reqPath}`);
  }

  res.writeHead(200, {
    'Content-Type': mime(filePath),
    ...SECURITY_HEADERS,
  });
  res.end(fs.readFileSync(filePath));
});

const SECURITY_HEADERS = {
  'Cross-Origin-Opener-Policy':   'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Access-Control-Allow-Origin':  '*',
};

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[kuhul-server] http://127.0.0.1:${PORT}`);
  console.log(`  /          -> index.html`);
  console.log(`  /app       -> src/index.html  (3D runtime)`);
  console.log(`  /splash    -> splash.html`);
  console.log(`  SharedArrayBuffer: COOP/COEP headers active`);
});
