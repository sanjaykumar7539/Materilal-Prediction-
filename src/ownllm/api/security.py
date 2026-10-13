"""Local security perimeter. No external classifier or authentication provider."""
from __future__ import annotations
import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
from collections import defaultdict, deque

log = logging.getLogger('ownllm.audit')
INJECTION = re.compile(r'(ignore|disregard)\s+(all\s+)?(previous|system)\s+instructions|reveal\s+(the\s+)?(system prompt|api key)', re.I)
SECRET = re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b')
EMAIL = re.compile(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b')

def output_check(text, secrets=(), citations=None, evidence_context=None):
    if SECRET.search(text) or any(value and value in text for value in secrets):
        return 'Response withheld because it may contain a secret.'
    if citations is not None:
        if any(number not in citations for number in re.findall(r'\[(\d+)\]', text)):
            return 'Response withheld because its citation references are invalid.'
    if evidence_context is not None:
        normalize = lambda value: ' '.join(value.casefold().split())
        claims = re.sub(r'\[\d+\]', '',text).strip()
        evidence = normalize(evidence_context)
        # Conservative extractive grounding, not a semantic truth classifier.
        sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+|\n+',claims) if part.strip()]
        if not sentences or any(normalize(sentence) not in evidence for sentence in sentences):
            return 'I cannot produce an answer supported by the permitted local evidence.'
    return EMAIL.sub('[email redacted]', text)

class SecurityPerimeter:
    def __init__(self, app):
        self.app = app
        self.history = defaultdict(deque)
        self.lock = asyncio.Lock()
        self.gate = None
        self.pending = 0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        started = time.monotonic()
        status = 500
        principal = 'anonymous'
        async def reject(code, detail):
            nonlocal status
            status = code
            raw = json.dumps({'detail':detail}).encode()
            await send({'type':'http.response.start','status':code,'headers':[(b'content-type',b'application/json'),(b'x-content-type-options',b'nosniff')]})
            await send({'type':'http.response.body','body':raw})
        headers = dict(scope.get('headers', []))
        acquired = queued = False
        try:
            if os.getenv('OWNLLM_REQUIRE_TLS','0') == '1' and scope['scheme'] != 'https':
                return await reject(403,'HTTPS required')
            keys = json.loads(os.getenv('OWNLLM_API_KEYS','{}'))
            if not isinstance(keys,dict) or any(role not in ('reader','writer','admin') for role in keys.values()):
                return await reject(503,'Invalid authentication configuration')
            provided = headers.get(b'x-api-key',b'').decode('utf-8',errors='replace')
            role = None
            for key,value in keys.items():
                if key and hmac.compare_digest(key.encode(),provided.encode()):
                    role = value
                    principal = hashlib.sha256(key.encode()).hexdigest()[:16]
                    break
            peer = (scope.get('client') or ('unknown',0))[0]
            if role is None:
                if keys or peer not in ('127.0.0.1','::1','localhost','testclient'):
                    return await reject(401,'Valid API key required')
                role = 'admin'
                principal = 'local'
            scope.setdefault('state',{})['principal'] = principal
            scope['state']['role'] = role
            if scope['path'] == '/metrics' and role != 'admin':
                return await reject(403,'Administrator permission required')
            if scope['method'] != 'POST' or scope['path'] not in ('/generate','/stream'):
                async def direct_send(message):
                    nonlocal status
                    if message['type']=='http.response.start':
                        status = message['status']
                    await send(message)
                return await self.app(scope,receive,direct_send)
            limit = int(os.getenv('OWNLLM_MAX_REQUEST_BYTES','200000'))
            content = bytearray()
            while True:
                message = await receive()
                if message['type'] == 'http.disconnect':
                    return
                content.extend(message.get('body',b''))
                if len(content)>limit:
                    return await reject(413,'Request too large')
                if not message.get('more_body',False):
                    break
            try:
                body = json.loads(content)
                if not isinstance(body,dict):
                    raise ValueError()
            except (ValueError,UnicodeError):
                return await reject(400,'Invalid JSON request')
            if body.get('materials_action') == 'feedback' and role == 'reader':
                return await reject(403,'Writer permission required for feedback')
            if role != 'admin' and (body.get('system_prompt') or body.get('apply_template') is False):
                return await reject(403,'Administrator permission required for prompt overrides')
            texts = [body.get(field,'') or '' for field in ('prompt','context','system_prompt','material','parameter')]
            if any(not isinstance(text,str) for text in texts):
                return await reject(422,'Text fields must be strings')
            joined = '\n'.join(texts)
            if any(ord(char)<32 and char not in '\n\r\t' for char in joined):
                return await reject(422,'Unsupported control characters')
            if SECRET.search(joined) or any(key and key in joined for key in keys):
                return await reject(400,'Remove secrets from the request')
            if INJECTION.search(joined):
                return await reject(400,'Request conflicts with instruction policy')
            requested = body.get('max_new_tokens',256)
            if not isinstance(requested,int) or requested<1 or requested>4096:
                return await reject(422,'Invalid output token budget')
            # UTF-8 bytes provide a conservative input budget without an external tokenizer.
            cost = len(joined.encode())+requested
            now = time.monotonic()
            async with self.lock:
                queue = self.history[principal]
                while queue and queue[0][0] <= now-60:
                    queue.popleft()
                if sum(item[1] for item in queue)+cost > int(os.getenv('OWNLLM_TOKEN_BUDGET_PER_MINUTE','100000')):
                    return await reject(429,'Token budget exceeded')
                queue.append((now,cost))
                for name in list(self.history):
                    if self.history[name] and self.history[name][-1][0]<=now-60:
                        del self.history[name]
            if self.gate is None:
                self.gate = asyncio.Semaphore(max(1,int(os.getenv('MAX_CONCURRENCY','4'))))
            if self.pending >= int(os.getenv('OWNLLM_MAX_PENDING','16')):
                return await reject(503,'Request queue full')
            self.pending += 1
            queued = True
            try:
                await asyncio.wait_for(self.gate.acquire(),float(os.getenv('OWNLLM_QUEUE_TIMEOUT','5')))
            except asyncio.TimeoutError:
                return await reject(503,'Queue timeout')
            acquired = True
            replayed = False
            async def replay():
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {'type':'http.request','body':bytes(content),'more_body':False}
                await asyncio.Event().wait()
            messages = []
            total = 0
            async def capture(message):
                nonlocal total,status
                if message['type']=='http.response.start':
                    status = message['status']
                total += len(message.get('body',b''))
                if total > int(os.getenv('OWNLLM_MAX_RESPONSE_BYTES','1000000')):
                    raise OverflowError()
                messages.append(message)
            try:
                await asyncio.wait_for(self.app(scope,replay,capture),float(os.getenv('OWNLLM_REQUEST_TIMEOUT','120')))
            except asyncio.TimeoutError:
                return await reject(504,'Request timeout')
            except OverflowError:
                return await reject(502,'Response too large')
            # Buffer streams before release so split-token secrets cannot bypass the guard.
            raw = b''.join(message.get('body',b'') for message in messages)
            start = next(message for message in messages if message['type']=='http.response.start')
            response_headers = dict(start.get('headers',[]))
            if status == 200:
                if b'application/json' in response_headers.get(b'content-type',b''):
                    value = json.loads(raw)
                    if isinstance(value,dict) and isinstance(value.get('text'),str):
                        value['text'] = output_check(value['text'],keys,scope['state'].get('citations'),scope['state'].get('evidence_context'))
                        raw = json.dumps(value).encode()
                else:
                    raw = output_check(raw.decode('utf-8'),keys,scope['state'].get('citations'),scope['state'].get('evidence_context')).encode()
            start['headers'] = [(key,value) for key,value in start.get('headers',[]) if key.lower()!=b'content-length']+[(b'x-content-type-options',b'nosniff')]
            await send(start)
            await send({'type':'http.response.body','body':raw})
        finally:
            if acquired:
                self.gate.release()
            if queued:
                self.pending -= 1
            from ownllm.observability import metrics
            metrics.inc('security_requests')
            if status >= 400:
                metrics.inc('security_rejected_or_failed')
            log.info('request principal=%s path=%s status=%s duration_ms=%.0f',principal,scope['path'],status,(time.monotonic()-started)*1000)
