/**
 * server.cjs — KUHUL WebX-3D static file server  v2
 * Port 7430 with COOP/COEP (SharedArrayBuffer) + cache headers.
 */
'use strict';

const http = require('http');
const fs   = require('fs');
const path = require('path');

const PORT = 7430;
const ROOT = __dirname;

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'application/javascript; charset=utf-8',
  '.mjs':  'application/javascript; charset=utf-8',
  '.cjs':  'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.md':   'text/plain; charset=utf-8',
  '.hlsl': 'text/plain; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.kuhul':'text/plain; charset=utf-8',
  '.svg':  'image/svg+xml',
  '.png':  'image/png',
  '.ico':  'image/x-icon',
  '.bin':  'application/octet-stream',
  '.safetensors': 'application/octet-stream',
  '.npy':  'application/octet-stream',
};

// Cache policy per extension
const CACHE = {
  '.html': 'no-cache',
  '.js':   'public, max-age=60',
  '.mjs':  'public, max-age=60',
  '.json': 'no-cache',
  '.md':   'public, max-age=120',
  '.hlsl': 'public, max-age=3600',
  '.wgsl': 'public, max-age=3600',
  '.svg':  'public, max-age=3600',
  '.png':  'public, max-age=86400',
};

const BASE_HEADERS = {
  'Cross-Origin-Opener-Policy':   'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
  'Cross-Origin-Resource-Policy': 'same-origin',
  'Access-Control-Allow-Origin':  '*',
  'X-Content-Type-Options':       'nosniff',
};

function mime(file) {
  return MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

function cacheHeader(file) {
  return CACHE[path.extname(file).toLowerCase()] || 'no-cache';
}

function headers(filePath) {
  return {
    ...BASE_HEADERS,
    'Content-Type':  mime(filePath),
    'Cache-Control': cacheHeader(filePath),
  };
}

const server = http.createServer((req, res) => {
  // Parse URL safely (no deprecated url.parse)
  let reqPath;
  try {
    reqPath = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  } catch {
    reqPath = '/';
  }

  // Remove trailing slash (except root)
  if (reqPath.length > 1 && reqPath.endsWith('/')) reqPath = reqPath.slice(0, -1);

  // API routes
  if (reqPath === '/api/manifest') return serveJson(res, 'server.manifest.json');
  if (reqPath === '/api/cache')    return serveJson(res, 'cache.manifest.json');

  // SPA routes → index.html
  if (reqPath === '/') reqPath = '/index.html';
  if (reqPath === '/app') reqPath = '/src/index.html';
  if (reqPath === '/splash') reqPath = '/splash.html';

  const filePath = path.join(ROOT, reqPath);

  // Security: must stay within ROOT
  if (!filePath.startsWith(ROOT + path.sep) && filePath !== ROOT) {
    res.writeHead(403, BASE_HEADERS);
    return res.end('403 Forbidden');
  }

  // Serve file
  try {
    const stat = fs.statSync(filePath);
    if (stat.isDirectory()) {
      // Try index.html inside directory
      const idx = path.join(filePath, 'index.html');
      if (fs.existsSync(idx)) {
        res.writeHead(200, headers(idx));
        return res.end(fs.readFileSync(idx));
      }
      res.writeHead(404, BASE_HEADERS);
      return res.end(`404 — ${reqPath}`);
    }
    res.writeHead(200, headers(filePath));
    return res.end(fs.readFileSync(filePath));
  } catch {
    res.writeHead(404, { ...BASE_HEADERS, 'Content-Type': 'text/plain' });
    return res.end(`404 — ${reqPath}`);
  }
});

function serveJson(res, filename) {
  const p = path.join(ROOT, filename);
  try {
    res.writeHead(200, { ...BASE_HEADERS, 'Content-Type': 'application/json' });
    res.end(fs.readFileSync(p));
  } catch {
    res.writeHead(404, BASE_HEADERS);
    res.end('{}');
  }
}

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[kuhul-server] v2  http://127.0.0.1:${PORT}`);
  console.log(`  /          -> index.html`);
  console.log(`  /app       -> src/index.html`);
  console.log(`  COOP/COEP + Cache-Control active`);
});
