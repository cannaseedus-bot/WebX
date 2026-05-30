import http from 'http';
import { URL } from 'url';

export async function run(op, url = 'http://localhost:5010/run') {
  const body = JSON.stringify(op);
  const u = new URL(url);
  const options = {
    method: 'POST',
    hostname: u.hostname,
    port: u.port,
    path: u.pathname,
    headers: {
      'Content-Type': 'application/json',
      'Content-Length': Buffer.byteLength(body)
    }
  };

  return new Promise((resolve, reject) => {
    const req = http.request(options, (res) => {
      let data = '';
      res.setEncoding('utf8');
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch (e) { reject(e); }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

// CLI
if (import.meta.url === `file://${process.argv[1]}`) {
  const op = process.argv[2] ? JSON.parse(process.argv[2]) : { "@op": "DOTNET_MATH_ADD", "a": 1.5, "b": 2.5 };
  run(op).then(r => console.log(JSON.stringify(r, null, 2))).catch(e => { console.error(e); process.exit(1); });
}
