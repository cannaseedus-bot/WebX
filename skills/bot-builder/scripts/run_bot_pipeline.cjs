#!/usr/bin/env node
const http = require('http');

function postJson(port, path, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const request = http.request(
      {
        hostname: '127.0.0.1',
        port,
        path,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
      },
      (response) => {
        let text = '';
        response.on('data', (chunk) => (text += chunk));
        response.on('end', () => resolve({ statusCode: response.statusCode, body: text }));
      },
    );

    request.on('error', reject);
    request.write(body);
    request.end();
  });
}

async function main() {
  const port = Number(process.env.BOT_HELPERS_PORT || 5780);
  const payload = {
    pipeline: 'bot-builder-sample',
    create: true,
    bots: [
      {
        name: 'lint-step',
        category: 'code',
        method: 'lint',
        params: { language: 'javascript' },
      },
      {
        name: 'doc-step',
        category: 'code',
        method: 'generate_docs',
        params: { format: 'json' },
      },
    ],
    input: {
      code: 'const value = 1;\nfunction demo(x) { return x + value; }',
    },
  };

  const result = await postJson(port, '/run', payload);
  process.stdout.write(`${result.body}\n`);
  process.exit(result.statusCode >= 200 && result.statusCode < 300 ? 0 : 1);
}

main().catch((error) => {
  console.error(error.message);
  process.exit(1);
});
