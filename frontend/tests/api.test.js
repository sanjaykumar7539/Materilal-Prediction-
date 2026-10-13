import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildRequest, streamReply } from '../src/api.js';

test('sends history as context and leaves domain templating to the backend', () => {
  const request = buildRequest([{ role: 'user', content: 'Earlier question' }, { role: 'assistant', content: 'Earlier answer' }, { role: 'user', content: 'Follow up' }], { domain: 'materials', tokens: 512, temperature: 0.7 });
  assert.equal(request.prompt, 'Follow up');
  assert.equal(request.context, 'User: Earlier question\n\nAssistant: Earlier answer');
  assert.equal(request.domain, 'materials');
  assert.equal(request.apply_template, true);
});
test('rejects oversized context before a request', () => {
  assert.throws(() => buildRequest([{ role: 'user', content: 'x'.repeat(50001) }, { role: 'user', content: 'Next' }], {}), /too long/);
});
test('decodes UTF-8 across stream chunks', async () => {
  const original = globalThis.fetch;
  const bytes = new TextEncoder().encode('Hello 🌍');
  globalThis.fetch = async () => new Response(new ReadableStream({ start(controller) { controller.enqueue(bytes.slice(0, 8)); controller.enqueue(bytes.slice(8)); controller.close(); } }));
  try { let result; await streamReply({}, undefined, text => { result = text; }); assert.equal(result, 'Hello 🌍'); } finally { globalThis.fetch = original; }
});
test('recognizes errors split across backend stream chunks', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode('partial\n[err')); controller.enqueue(new TextEncoder().encode('or] runtime failed\n')); controller.close(); } }));
  try { await assert.rejects(streamReply({}, undefined, () => {}), /runtime failed/); } finally { globalThis.fetch = original; }
});
test('surfaces rate limit retry timing', async () => {
  const original = globalThis.fetch;
  globalThis.fetch = async () => new Response(JSON.stringify({ detail: 'rate limit exceeded' }), { status: 429, headers: { 'Retry-After': '12' } });
  try { await assert.rejects(streamReply({}, undefined, () => {}), /12 seconds/); } finally { globalThis.fetch = original; }
});
