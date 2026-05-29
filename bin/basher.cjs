#!/usr/bin/env node
'use strict';

/*
 * BASHER — KUHUL Operator Shell
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { program } = require('commander');

const pkg = require('../package.json');

// -----------------------------------------------------------------------------
// Banner
// -----------------------------------------------------------------------------
console.log(`
╔══════════════════════════════════════╗
║           BASHER v${pkg.version}                ║
║     KUHUL Operator Control Shell     ║
╚══════════════════════════════════════╝
`);

// -----------------------------------------------------------------------------
// CLI
// -----------------------------------------------------------------------------
program
  .name('basher')
  .description('KUHUL Operator Shell')
  .version(pkg.version);

// -----------------------------------------------------------------------------
// exec
// -----------------------------------------------------------------------------
program
  .command('exec <cmd>')
  .description('Execute an operator command')
  .option('--record', 'Record execution trace')
  .action((cmd, options) => {
    console.log(`▶ Executing: ${cmd}`);

    if (options.record) {
      const trace = {
        type: 'basher.exec',
        cmd,
        cwd: process.cwd(),
        timestamp: new Date().toISOString()
      };

      trace.hash = crypto
        .createHash('sha256')
        .update(JSON.stringify(trace))
        .digest('hex');

      fs.writeFileSync(
        path.resolve(process.cwd(), 'basher-trace.json'),
        JSON.stringify(trace, null, 2)
      );

      console.log('Trace recorded');
      console.log(`Hash: ${trace.hash}`);
    }

    console.log('✓ Execution complete (stub)');
    process.exit(0);
  });

// -----------------------------------------------------------------------------
// doctor
// -----------------------------------------------------------------------------
program
  .command('doctor')
  .description('Basher diagnostics')
  .action(() => {
    console.log('BASHER Doctor\n');
    console.log('Node:', process.version);
    console.log('Platform:', process.platform);
    console.log('CWD:', process.cwd());
    console.log('Commander: OK');
    process.exit(0);
  });

program.parse(process.argv);

if (!process.argv.slice(2).length) {
  program.outputHelp();
}
