import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MaterialMessage } from './materialReport.js';
import 'katex/dist/katex.min.css';
import { ArrowUp, ArrowUpRight, Check, Copy, Download, Menu, MessageSquare, Plus, Settings2, Square, Trash2, X, Sparkles, RefreshCw } from 'lucide-react';
import { buildRequest, getJSON, streamReply } from './api.js';
import './styles.css';

const STORAGE = 'campusmind.chats.v1';
const freshChat = () => ({ id: crypto.randomUUID(), title: 'New conversation', messages: [], updated: Date.now() });
function loadChats() {
  try { const value = JSON.parse(localStorage.getItem(STORAGE)); if (Array.isArray(value)) return value.filter(c => typeof c.id === 'string' && typeof c.title === 'string' && Array.isArray(c.messages) && c.messages.every(m => ['user', 'assistant'].includes(m.role) && typeof m.content === 'string')); } catch {}
  return [];
}
function App() {
  const [chats, setChats] = useState(() => { const saved = loadChats(); return saved.length ? saved : [freshChat()]; });
  const [active, setActive] = useState(() => chats[0].id);
  const [draft, setDraft] = useState('');
  const [busy, setBusy] = useState(false);
  const [health, setHealth] = useState(null);
  const [domains, setDomains] = useState([]);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [settings, setSettings] = useState({ domain: '', tokens: 512, temperature: 0.7, materialsAction: 'profile', material: '', parameter: '' });
  const controller = useRef(null);
  const bottom = useRef(null);
  const input = useRef(null);
  const chat = chats.find(c => c.id === active) || chats[0];
  useEffect(() => { try { localStorage.setItem(STORAGE, JSON.stringify(chats)); } catch { setNotice('Browser storage is full or unavailable. New messages may not survive a reload.'); } }, [chats]);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'instant' }); }, [chat.messages]);
  useEffect(() => {
    const abort = new AbortController();
    async function refresh() {
      try { const status = await getJSON('health', abort.signal); setHealth(status); } catch { if (!abort.signal.aborted) setHealth({ ok: false }); }
    }
    refresh();
    getJSON('domains', abort.signal).then(data => setDomains(Array.isArray(data.domains) ? data.domains : [])).catch(() => {});
    const timer = setInterval(refresh, 15000);
    return () => { abort.abort(); controller.current?.abort(); clearInterval(timer); };
  }, []);
  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(''), 5000); return () => clearTimeout(timer); }, [notice]);
  function update(id, fn) { setChats(all => all.map(c => c.id === id ? fn(c) : c)); }
  function newChat() { if (busy) return; const next = freshChat(); setChats(all => [next, ...all]); setActive(next.id); setDraft(''); setError(''); setMobileOpen(false); input.current?.focus(); }
  async function send(event) {
    event.preventDefault();
    if (busy || !draft.trim()) return;
    const text = draft.trim();
    const id = chat.id;
    const messages = [...chat.messages, { role: 'user', content: text }];
    let body;
    try { body = buildRequest(messages, settings); } catch (e) { setError(e.message); return; }
    const abort = new AbortController(); controller.current = abort;
    setBusy(true); setError(''); setDraft('');
    update(id, c => ({ ...c, title: c.messages.length ? c.title : text.slice(0, 48), updated: Date.now(), messages: [...messages, { role: 'assistant', content: '' }] }));
    try {
      await streamReply(body, abort.signal, content => update(id, c => ({ ...c, messages: [...c.messages.slice(0, -1), { role: 'assistant', content }] })));
    } catch (e) {
      update(id, c => ({ ...c, messages: [...c.messages.slice(0, -2), { ...c.messages.at(-2), failed: true }, { ...c.messages.at(-1), failed: true, stopped: e.name === 'AbortError' }] }));
      if (e.name !== 'AbortError') { setError(e.message); setDraft(text); }
    } finally { setBusy(false); controller.current = null; input.current?.focus(); }
  }
  function removeChat() {
    if (busy) return;
    const rest = chats.filter(c => c.id !== active); const next = rest.length ? rest : [freshChat()];
    setChats(next); setActive(next[0].id); setError(''); setDraft('');
  }
  function download() {
    const blob = new Blob([chat.messages.map(m => `## ${m.role === 'user' ? 'You' : 'CampusMind'}\n\n${m.content}`).join('\n\n')], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'campusmind-conversation.md'; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  const materialsMode = (settings.domain || health?.default_domain) === 'materials';
  const ready = health?.ok && (health?.model_loaded || (materialsMode && health?.materials_policy === 'strict'));
  return <div className="app">
    {mobileOpen && <button className="scrim" aria-label="Close navigation" onClick={() => setMobileOpen(false)}/>}
    <aside className={`sidebar ${mobileOpen ? 'is-open' : ''}`}>
      <a className="brand" href="/" aria-label="CampusMind home"><span className="brand-mark"><Sparkles size={20}/></span><span>CampusMind<span className="brand-sub">Your thinking space</span></span></a>
      <button className="new-chat" onClick={newChat} disabled={busy}><Plus size={18}/> New conversation <ArrowUpRight size={14}/></button>
      <div className="section-label">CONVERSATIONS <span>{chats.filter(c => c.messages.length).length}</span></div>
      <nav aria-label="Conversations" className="history">{chats.map(c => <button key={c.id} className={c.id === active ? 'selected' : ''} disabled={busy} onClick={() => { setActive(c.id); setDraft(''); setError(''); setMobileOpen(false); }}><MessageSquare size={16}/><span>{c.title}</span></button>)}</nav>
      <div className="sidebar-bottom"><div className="local-note"><span className="local-dot"/> Saved on this browser<p>Your conversations stay on this device.</p></div><button className="profile" onClick={() => { setSettingsOpen(v => !v); setMobileOpen(false); }}><span className="avatar">Y</span><span>Your workspace<small>OwnLLM</small></span><Settings2 size={17}/></button></div>
    </aside>
    <main>
      <header><div className="header-left"><button className="icon-button mobile-menu" aria-label="Open navigation" onClick={() => setMobileOpen(true)}><Menu size={20}/></button><span className="header-title">{chat.messages.length ? chat.title : 'A fresh perspective'}</span><span className="model-pill">OwnLLM</span></div><div className="header-actions"><span className={`status ${ready ? 'ready' : ''}`}><i/>{!health ? 'Connecting' : ready ? 'Model ready' : health.ok ? 'Model not loaded' : 'API offline'}</span><button className="icon-button" onClick={download} disabled={!chat.messages.length || busy} aria-label="Export conversation"><Download size={18}/></button><button className="icon-button" onClick={() => setSettingsOpen(v => !v)} aria-label="Generation settings" aria-expanded={settingsOpen}><Settings2 size={18}/></button></div></header>
      {settingsOpen && <section className="settings" aria-label="Generation settings"><div className="settings-heading"><strong>Make it your own</strong><button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><X size={18}/></button></div><label>Domain<select aria-label="Domain" disabled={busy} value={settings.domain} onChange={e => setSettings(s => ({ ...s, domain: e.target.value }))}><option value="">Server default{health?.default_domain ? ` (${health.default_domain})` : ''}</option>{domains.map(d => <option key={d} value={d}>{d}</option>)}</select></label><label>Response length <span>{settings.tokens} tokens</span><input disabled={busy} type="range" min="64" max="4096" step="64" value={settings.tokens} onChange={e => setSettings(s => ({ ...s, tokens: e.target.value }))}/></label><label>Temperature <span>{Number(settings.temperature).toFixed(1)}</span><input disabled={busy} type="range" min="0" max="2" step="0.1" value={settings.temperature} onChange={e => setSettings(s => ({ ...s, temperature: e.target.value }))}/></label><p>Lower temperature gives more focused responses. Higher values introduce more variation.</p><button className="text-button" disabled={busy} onClick={removeChat}><Trash2 size={15}/> Delete this conversation</button></section>}
      {settingsOpen && materialsMode && <section className="settings" aria-label="Materials settings"><label>Materials task<select disabled={busy} value={settings.materialsAction} onChange={e => setSettings(s => ({ ...s, materialsAction: e.target.value }))}><option value="profile">Engineering profile</option><option value="verify">Review provided data</option><option value="feedback">Save a correction</option></select></label><label>Material name<input disabled={busy} maxLength={200} value={settings.material} onChange={e => setSettings(s => ({ ...s, material: e.target.value }))} placeholder="Exact material or grade"/></label><label>Specific parameter (optional)<input disabled={busy} maxLength={200} value={settings.parameter} onChange={e => setSettings(s => ({ ...s, parameter: e.target.value }))}/></label><p>{settings.materialsAction === 'feedback' ? 'Your next message is saved locally as an unverified correction. Enter the material name first.' : settings.materialsAction === 'verify' ? 'Paste the text to review. Review is offline and limited; it does not certify exact values.' : 'Profiles use reviewed local records. Missing data returns the verified-data fallback.'}</p></section>}
      <div className="conversation">
        {!chat.messages.length ? <section className="welcome"><div className="welcome-symbol"><Sparkles size={32} strokeWidth={1.5}/></div><h1>A little curiosity.<br/><span>A lot of possibility.</span></h1><p>A quiet space to explore ideas, untangle questions,<br className="desktop-break"/> and find your next clear thought.</p><div className="suggestions">{(materialsMode ? [['Analyze a material', 'Copper'], ['Compare materials', 'Compare copper and aluminium'], ['Explore improvements', 'Improve glass for safety']] : [['Explore an idea', 'Explain a complex concept in simple terms.'], ['Work through a problem', 'Help me break down a challenging problem step by step.'], ['Make something clearer', 'Help me turn a rough idea into a clear, structured plan.']]).map(([title, prompt]) => <button key={title} onClick={() => { setDraft(prompt); input.current?.focus(); }}><span>{title}</span><ArrowUpRight size={18}/></button>)}</div></section> : <div className="messages">{chat.messages.map((m, index) => <article key={index} className={`message ${m.role}`}><div className={`message-avatar ${m.role === 'assistant' ? 'assistant-avatar' : ''}`}>{m.role === 'assistant' ? <Sparkles size={17}/> : 'Y'}</div><div className="message-body"><div className="message-heading">{m.role === 'user' ? 'You' : 'CampusMind'}<span>{m.role === 'assistant' ? 'OwnLLM' : ''}</span></div>{m.content ? <MaterialMessage text={m.content}/> : <p className="thinking">{busy && index === chat.messages.length - 1 ? 'Thinking through it…' : m.stopped ? 'Response stopped.' : 'No response received.'}</p>}{m.role === 'assistant' && m.content && <button className="copy-button" aria-label="Copy response" onClick={async () => { try { await navigator.clipboard.writeText(m.content); setNotice('Response copied'); } catch { setNotice('Copy unavailable. Select the response text to copy it.'); } }}><Copy size={13}/> Copy</button>}{m.failed && m.content && <small className="incomplete">{m.stopped ? 'Stopped' : 'Response incomplete'}</small>}</div></article>)}<div ref={bottom}/></div>}
      </div>
      <div className="composer-wrap">{health && !ready && <div className="connection-note"><RefreshCw size={14}/>{health.ok ? 'Load your model and tokenizer in the backend to start chatting.' : 'Start the OwnLLM API on port 8000 to connect.'}</div>}{error && <div className="error" role="alert"><span>{error}</span><button className="icon-button" aria-label="Dismiss error" onClick={() => setError('')}><X size={16}/></button></div>}<form className="composer" onSubmit={send}><label className="sr-only" htmlFor="message">Message CampusMind</label><textarea id="message" ref={input} value={draft} rows="2" maxLength={32000} placeholder="What's on your mind?" disabled={busy} onChange={e => setDraft(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); if (!busy) e.currentTarget.form.requestSubmit(); } }}/><div className="composer-footer"><span><span className="tiny-mark"><Sparkles size={12}/></span>{settings.domain || health?.default_domain || 'General'} <span className="divider">/</span> OwnLLM</span>{busy ? <button type="button" className="send" onClick={() => controller.current?.abort()} aria-label="Stop response"><Square size={16}/></button> : <button className="send" type="submit" disabled={!draft.trim() || !ready} aria-label="Send message"><ArrowUp size={20}/></button>}</div></form><p className="composer-note">AI can make mistakes. Check important details.<span>Enter to send · Shift + Enter for a new line</span></p></div>
    </main>{notice && <div className="toast" role="status"><Check size={15}/>{notice}</div>}
  </div>;
}
createRoot(document.getElementById('root')).render(<App/>);
