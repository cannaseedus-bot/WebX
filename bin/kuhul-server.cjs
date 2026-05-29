#!/usr/bin/env node
'use strict';

/*
 * KUHUL SERVER
 * Deterministic Runtime Host
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const pkg = require('./package.json');

// -----------------------------------------------------------------------------
// Server State
// -----------------------------------------------------------------------------
const state = {
  version: pkg.version,
  pid: process.pid,
  startedAt: new Date().toISOString(),
  cwd: process.cwd()
};

// -----------------------------------------------------------------------------
// Boot Banner
// -----------------------------------------------------------------------------
console.log(`
╔══════════════════════════════════════╗
║        KUHUL SERVER v${pkg.version}           ║
║    Deterministic Runtime Host        ║
╚══════════════════════════════════════╝
`);

console.log('PID:', state.pid);
console.log('CWD:', state.cwd);
console.log('Started:', state.startedAt);

// -----------------------------------------------------------------------------
// Trace Helper
// -----------------------------------------------------------------------------
function writeTrace(event, payload = {}) {
  const trace = {
    event,
    payload,
    timestamp: new Date().toISOString()
  };

  trace.hash = crypto
    .createHash('sha256')
    .update(JSON.stringify(trace))
    .digest('hex');

  fs.appendFileSync(
    path.resolve(state.cwd, 'kuhul-server.log'),
    JSON.stringify(trace) + '\n'
  );
}

// -----------------------------------------------------------------------------
// Example Tick Loop (Stub)
// -----------------------------------------------------------------------------
setInterval(() => {
  writeTrace('tick', { uptime: process.uptime() });
}, 1000);

// -----------------------------------------------------------------------------
// Shutdown Handling
// -----------------------------------------------------------------------------
process.on('SIGINT', () => {
  writeTrace('shutdown', { reason: 'SIGINT' });
  console.log('\nShutting down...');
  process.exit(0);
});
