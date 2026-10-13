export async function getJSON(path, signal) {
  const response = await fetch(`/api/${path}`, { signal });
  if (!response.ok) throw new Error(`Backend returned ${response.status}.`);
  return response.json();
}

export function buildRequest(messages, settings) {
  const last = messages.at(-1);
  const history = messages.slice(0, -1).filter(message => message.content && !message.failed);
  const context = history.map(message => `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}`).join('\n\n');
  if (last.content.length > 32000) throw new Error('Your message exceeds 32,000 characters. Shorten it and try again.');
  if (context.length > 50000) throw new Error('This conversation is too long for the model. Start a new chat.');
  const priorUser = history.filter(message => message.role === 'user').at(-1)?.content?.trim();
  const previousMaterial = priorUser && priorUser.length <= 200 ? priorUser : undefined;
  return { previous_material: previousMaterial, prompt: last.content, context: context || undefined, domain: settings.domain || undefined, material: settings.material?.trim() || undefined, parameter: settings.parameter?.trim() || undefined, materials_action: settings.materialsAction || 'profile', max_new_tokens: Number(settings.tokens), temperature: Number(settings.temperature), apply_template: true };
}

export async function streamReply(body, signal, onChunk) {
  const response = await fetch('/api/stream', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), signal });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    const detail = typeof error.detail === 'string' ? error.detail : `Request failed (${response.status}).`;
    throw new Error(response.status === 429 ? `${detail} Try again in ${response.headers.get('retry-after') || '60'} seconds.` : detail);
  }
  if (!response.body) throw new Error('The backend did not return a response stream.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let output = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      output += done ? decoder.decode() : decoder.decode(value, { stream: true });
      const marker = output.indexOf('\n[error] ');
      if (marker !== -1) throw new Error(output.slice(marker + 9).trim());
      onChunk(output);
      if (done) break;
    }
    if (!output.trim()) throw new Error('The model returned an empty response. Try a different prompt.');
  } finally { reader.releaseLock(); }
}
