import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Readable } from 'node:stream';

const root = path.resolve(fileURLToPath(new URL('./dist/', import.meta.url)));
const upstream = process.env.OWNLLM_API_URL || 'http://127.0.0.1:8000';
const routes = new Map([['/api/health', 'GET'], ['/api/domains', 'GET'], ['/api/stream', 'POST'], ['/api/generate', 'POST']]);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.svg': 'image/svg+xml' };
http.createServer(async (req, res) => {
  const pathname = new URL(req.url, 'http://localhost').pathname;
  try {
    if (pathname.startsWith('/api/')) {
      if (routes.get(pathname) !== req.method) { res.writeHead(404).end(); return; }
      const controller = new AbortController();
      res.on('close', () => controller.abort());
      const headers = { 'content-type': 'application/json' };
      if (process.env.OWNLLM_API_KEY) headers['x-api-key'] = process.env.OWNLLM_API_KEY;
      const response = await fetch(new URL(pathname.slice(4), upstream), {
        method: req.method, headers, signal: controller.signal,
        ...(req.method === 'POST' ? { body: req, duplex: 'half' } : {}),
      });
      res.writeHead(response.status, { 'content-type': response.headers.get('content-type') || 'application/json', 'cache-control': 'no-store', ...(response.headers.get('retry-after') ? { 'retry-after': response.headers.get('retry-after') } : {}) });
      if (response.body) Readable.fromWeb(response.body).on('error', () => res.destroy()).pipe(res);
      else res.end();
      return;
    }
    if (!['GET', 'HEAD'].includes(req.method)) { res.writeHead(405).end(); return; }
    const target = path.resolve(root, '.' + decodeURIComponent(pathname));
    if (!target.startsWith(root + path.sep) && target !== root) { res.writeHead(403).end(); return; }
    let file = target;
    let data;
    try { data = await readFile(file); } catch {
      if (path.extname(pathname)) { res.writeHead(404).end(); return; }
      file = path.join(root, 'index.html'); data = await readFile(file);
    }
    res.writeHead(200, { 'content-type': types[path.extname(file)] || 'application/octet-stream', 'x-content-type-options': 'nosniff' });
    res.end(req.method === 'HEAD' ? undefined : data);
  } catch (error) {
    if (res.headersSent) { res.destroy(); return; }
    res.writeHead(502, { 'content-type': 'application/json' }).end(JSON.stringify({ detail: 'Unable to reach the OwnLLM API. Check OWNLLM_API_URL and start the backend.' }));
  }
}).listen(Number(process.env.PORT || 3000), process.env.HOST || '127.0.0.1', () => console.log('CampusMind chat: http://127.0.0.1:' + (process.env.PORT || 3000)));
