"""Reproducible document splits and assistant-only instruction training examples."""
from __future__ import annotations
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import random
import re
from pathlib import Path
import torch
from ownllm.data.documents import SUPPORTED, iter_documents

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>"]

def cached_documents(source, cache):
    """Cache extracted text by source-file digest; changed files get a new entry."""
    source, cache = Path(source), Path(cache)
    cache.mkdir(parents=True,exist_ok=True)
    try:
        font_version = version('fonttools')
    except PackageNotFoundError:
        font_version = 'absent'
    parser_version = version('pypdf') + '/' + font_version
    summary = []
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED:
            continue
        with path.open('rb') as handle:
            digest = hashlib.file_digest(handle,'sha256').hexdigest()
        digest = hashlib.sha256((digest + parser_version).encode()).hexdigest()
        entry = cache/(digest+'.json')
        if entry.exists():
            records = json.loads(entry.read_text(encoding='utf-8'))
        else:
            records = [{'suffix':name[len(str(path)):],'text':text} for name,text in iter_documents(path)]
            temporary = entry.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(records,ensure_ascii=False),encoding='utf-8')
            temporary.replace(entry)
        summary.append({'source':str(path),'readable_documents':len(records),'status':'readable' if records else 'no_readable_supported_text'})
        for record in records:
            yield str(path)+record['suffix'],record['text']
    (cache/'report.json').write_text(json.dumps({'parser_version':parser_version,'files':summary},indent=2),encoding='utf-8')

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()

def split_documents(documents, seed=42):
    unique = {}
    duplicates = []
    skipped = []
    for name, text in sorted(documents, key=lambda row: row[0]):
        text = text.replace('\x00', '').strip()
        if len(normalize(text)) < 200:
            skipped.append(name)
            continue
        digest = hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()
        if digest in unique:
            duplicates.append({'source':name, 'duplicate_of':unique[digest]['source']})
        else:
            unique[digest] = {'source':name, 'text':text, 'sha256':digest}
    records = sorted(unique.values(), key=lambda record:record['sha256'])
    random.Random(seed).shuffle(records)
    if len(records) < 3:
        raise ValueError('At least three distinct readable documents are required for train/validation/test splits')
    heldout = max(1, len(records) // 10)
    return {'train':records[2*heldout:], 'validation':records[:heldout], 'test':records[heldout:2*heldout]}, {'duplicates':duplicates, 'short_documents':skipped}

def read_sft(path):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row.get('instruction'), str) or not isinstance(row.get('output'), str) or not row['output'].strip():
            raise ValueError(f'Invalid instruction example at {path}:{number}')
        rows.append(row)
    if not rows:
        raise ValueError(f'Empty instruction dataset: {path}')
    return rows

def instruction_prompt(row):
    question = row['instruction'] + ('\n\n' + row['input'] if row.get('input') else '')
    return f'<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n'

def encode_sft(row, tokenizer, seq_len):
    prefix = tokenizer.encode(instruction_prompt(row)).ids
    answer = tokenizer.encode(row['output']).ids + [tokenizer.token_to_id('<|im_end|>'), tokenizer.token_to_id('<eos>')]
    # Keep a complete answer where possible and reserve at least one prompt token.
    answer = answer[:seq_len]
    prefix = prefix[-max(1, seq_len + 1 - len(answer)):]
    tokens = prefix + answer
    ids = tokens[:-1]
    targets = tokens[1:]
    labels = [-100 if position + 1 < len(prefix) else target for position, target in enumerate(targets)]
    padding = seq_len - len(ids)
    return torch.tensor(ids + [tokenizer.token_to_id('<pad>')] * padding), torch.tensor(labels + [-100] * padding)

def audit_sft(splits):
    audit = {'counts':{name:len(rows) for name, rows in splits.items()}, 'shared_source_ids':{}, 'duplicate_prompts':{}}
    names = list(splits)
    for i, name in enumerate(names):
        for other in names[i+1:]:
            pair = name + '/' + other
            a = {normalize(instruction_prompt(row)) for row in splits[name]}
            b = {normalize(instruction_prompt(row)) for row in splits[other]}
            shared = sorted(a & b)
            audit['duplicate_prompts'][pair] = len(shared)
            audit['shared_source_ids'][pair] = sorted({r.get('source_id') for r in splits[name] if r.get('source_id')} & {r.get('source_id') for r in splits[other] if r.get('source_id')})
            if shared:
                raise ValueError(f'Instruction prompts overlap between {name} and {other}')
    return audit
