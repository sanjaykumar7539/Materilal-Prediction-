"""Pretrain the native model on supplied source/page/text JSONL records."""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ownllm import ModelConfig, OwnLLM
from ownllm.data.memmap import MemmapTokenDataset
from ownllm.data.sharding import shard_encoded_documents
from ownllm.tokenizer_train import train_bpe
from ownllm.training.books import SPECIAL_TOKENS, normalize
from ownllm.training.checkpoint import save_checkpoint
from ownllm.training.engine import Trainer
from train_books import evaluate, write_json
from tokenizers import Tokenizer


def load_base_export(folder, cfg):
    """Load locally saved weights with their exact token-ID mapping."""
    folder = Path(folder)
    base_cfg = ModelConfig.from_dict(json.loads((folder/'model_config.json').read_text()))
    if base_cfg.to_dict() != cfg.to_dict():
        raise ValueError('Base model configuration must match the requested model')
    tokenizer = Tokenizer.from_file(str(folder/'tokenizer.json'))
    if tokenizer.get_vocab_size() != cfg.vocab_size or tokenizer.token_to_id('<eos>') is None:
        raise ValueError('Base tokenizer is incompatible with the model')
    checkpoint = torch.load(folder/'ownllm.pt', map_location='cpu', weights_only=False)
    if checkpoint.get('config') != cfg.to_dict():
        raise ValueError('Base checkpoint configuration does not match its export')
    return tokenizer, checkpoint


def corpus_rows(source):
    """Accept a JSON array or JSONL; never interpret record text as commands."""
    with Path(source).open(encoding='utf-8-sig') as handle:
        first = ''
        while not first and (character := handle.read(1)):
            if not character.isspace():
                first = character
        handle.seek(0)
        if first == '[':
            rows = json.load(handle)
            yield from rows
        else:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def prepare_sources(source, folder, seed=42, exclude_seen_input=None):
    before = source.stat().st_size
    books = defaultdict(list)
    count = 0
    prior = set()
    if exclude_seen_input:
        for row in corpus_rows(exclude_seen_input):
            prior.add(hashlib.sha256(normalize(row['text']).encode()).hexdigest())
    prior_removed = 0
    for number,row in enumerate(corpus_rows(source),1):
            if not isinstance(row,dict) or not isinstance(row.get('source'),str) or not row['source'].strip() or not isinstance(row.get('text'),str):
                raise ValueError(f'Invalid source/text record at line {number}')
            count += 1
            if hashlib.sha256(normalize(row['text']).encode()).hexdigest() in prior:
                prior_removed += 1
                continue
            books[row['source']].append((row.get('page'),row.get('chunk_id'),row['text']))
    if source.stat().st_size != before:
        raise ValueError('Source download changed during reading; wait until it completes')
    names = sorted(books)
    if len(names)<5:
        raise ValueError('At least five source groups are needed for separate book splits')
    random.Random(seed).shuffle(names)
    held = max(1,round(len(names)*.1))
    assignments = {'test':names[:held],'validation':names[held:2*held],'train':names[2*held:]}
    seen = set()
    removed, retained, empty = 0,0,[]
    manifest = {}
    # Held-out duplicates take precedence: a repeated test passage never enters training.
    for split in ('test','validation','train'):
        target = folder/split
        target.mkdir(parents=True)
        manifest[split] = []
        for index,name in enumerate(assignments[split]):
            chunks = []
            for page,chunk_id,text in books[name]:
                normalized = normalize(text)
                digest = hashlib.sha256(normalized.encode()).hexdigest()
                if not normalized or digest in seen:
                    removed += 1
                    continue
                seen.add(digest)
                retained += 1
                chunks.append(text.strip())
            if not chunks:
                empty.append({'source':name,'split':split})
                continue
            content = '\n\n'.join(chunks)
            path = target/f'{index:03d}.txt'
            path.write_text(content,encoding='utf-8')
            manifest[split].append({'source':name,'path':str(path),'chunks':len(chunks),
                                    'text_sha256':hashlib.sha256(content.encode()).hexdigest()})
    if not all(manifest.values()):
        raise ValueError('A split became empty after duplicate removal')
    return {'input_records':count,'source_books':len(books),'retained_chunks':retained,
            'excluded_prior_chunks':prior_removed,
            'excluded_prior_input_sha256':hashlib.sha256(Path(exclude_seen_input).read_bytes()).hexdigest() if exclude_seen_input else None,
            'removed_duplicate_or_empty_chunks':removed,'empty_sources':empty,'splits':manifest,
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'deduplication':'Exact case-folded whitespace-normalized chunk SHA256; test then validation then train precedence',
            'limitations':['Different source filenames may be alternate editions of the same work; near-duplicate books are not guaranteed to be separated.',
                           'A license declaration is not included in the supplied JSONL.',
                           'No independently reviewed engineering measurements are created from these records.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--config',default='configs/campusmind_medium.json')
    parser.add_argument('--out',default='runs/corpus_medium_v1')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seq-len',type=int,default=256)
    parser.add_argument('--batch',type=int,default=1)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--eval-batches',type=int,default=64)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--base-export',help='Continue from this local export, preserving its tokenizer')
    parser.add_argument('--sampling-seed',type=int,default=43)
    parser.add_argument('--new-corpus',action='store_true',help='Continue on a different corpus with newly held-out sources')
    parser.add_argument('--exclude-seen-input',help='Remove exact chunks from the earlier corpus before new source splitting')
    args = parser.parse_args()
    if args.new_corpus and (not args.base_export or not args.exclude_seen_input):
        parser.error('--new-corpus requires --base-export and --exclude-seen-input')
    if min(args.steps,args.seq_len,args.batch,args.threads,args.eval_batches)<1 or args.lr<=0:
        parser.error('Training settings must be positive')
    out = Path(args.out)
    if out.exists():
        raise SystemExit('Choose a new --out to preserve previous runs.')
    out.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    sampling = random.Random(args.sampling_seed)
    started = time.monotonic()
    cfg = ModelConfig.from_dict(json.loads(Path(args.config).read_text()))
    if args.seq_len>cfg.max_seq_len:
        raise ValueError('Training sequence exceeds model context length')
    def record(row):
        with (out/'training.jsonl').open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    corpus = prepare_sources(Path(args.input),out/'prepared',args.seed,args.exclude_seen_input)
    write_json(out/'corpus_report.json',corpus)
    record({'stage':'prepared_sources','records':corpus['input_records'],'retained_chunks':corpus['retained_chunks'],
            'removed':corpus['removed_duplicate_or_empty_chunks'],'books':{k:len(v) for k,v in corpus['splits'].items()}})
    base_checkpoint = None
    if args.base_export:
        tokenizer, base_checkpoint = load_base_export(args.base_export, cfg)
        # Preserve the source groups and exact passages used for held-out evaluation.
        if not args.new_corpus:
            previous = json.loads((Path(args.base_export)/'corpus_report.json').read_text())
            for split in ('train','validation','test'):
                expected = {(row['source'],row['text_sha256']) for row in previous['splits'][split]}
                actual = {(row['source'],row['text_sha256']) for row in corpus['splits'][split]}
                if expected != actual:
                    raise ValueError('Continuation corpus must preserve the original held-out split')
        (out/'tokenizer').mkdir()
        shutil.copy2(Path(args.base_export)/'tokenizer.json',out/'tokenizer/tokenizer.json')
    else:
        tokenizer = train_bpe(sorted((out/'prepared/train').glob('*.txt')),out/'tokenizer',cfg.vocab_size,special_tokens=SPECIAL_TOKENS)
    if tokenizer.get_vocab_size()!=cfg.vocab_size:
        raise ValueError('Corpus could not produce the requested vocabulary size')
    counts = {}
    for split in ('train','validation','test'):
        documents = (tokenizer.encode(path.read_text(encoding='utf-8')).ids+[tokenizer.token_to_id('<eos>')]
                     for path in sorted((out/'prepared'/split).glob('*.txt')))
        metadata = shard_encoded_documents(documents,out/'prepared'/split/'shards')
        counts[split] = sum(row['tokens'] for row in metadata)
    corpus['tokens'] = counts
    write_json(out/'corpus_report.json',corpus)
    datasets = {split:MemmapTokenDataset(sorted((out/'prepared'/split/'shards').glob('*.bin')),args.seq_len)
                for split in ('train','validation','test')}
    if any(not len(value) for value in datasets.values()):
        raise ValueError('Insufficient tokens in a split')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OwnLLM(cfg).to(device)
    if base_checkpoint is not None:
        model.load_state_dict(base_checkpoint['model'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.1)
    trainer = Trainer(model,optimizer,amp='bf16' if device=='cuda' else 'none')
    initial = evaluate(model,datasets['validation'],device,args.batch,args.eval_batches,args.seed)
    best = initial['loss']
    save_checkpoint(out/'best.pt',model,extra={'validation':initial})
    record({'stage':'initial','parameters':model.num_parameters(),'device':device,'tokens':counts,'validation':initial})
    for step in range(1,args.steps+1):
        samples = [datasets['train'][sampling.randrange(len(datasets['train']))] for _ in range(args.batch)]
        ids,labels = [torch.stack([sample[k] for sample in samples]).to(device) for k in (0,1)]
        optimizer.param_groups[0]['lr'] = args.lr*min(1,step/50)*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        loss = trainer.train_batch([(ids,labels)])
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite loss')
        if step==1 or step%50==0:
            record({'stage':'pretrain','step':step,'loss':loss,'elapsed_seconds':round(time.monotonic()-started,1)})
        if step%250==0 or step==args.steps:
            score = evaluate(model,datasets['validation'],device,args.batch,args.eval_batches,args.seed)
            record({'stage':'validation','step':step,**score})
            save_checkpoint(out/'last.pt',model,optimizer,step=step,extra={'validation':score})
            if score['loss']<best:
                best = score['loss']
                save_checkpoint(out/'best.pt',model,step=step,extra={'validation':score})
    selected = torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(selected['model'])
    test = evaluate(model,datasets['test'],device,args.batch,args.eval_batches,args.seed)
    report = {'run':vars(args),'stage':'corpus_pretraining','parameters':model.num_parameters(),'device':device,
              'base_checkpoint_sha256':hashlib.sha256((Path(args.base_export)/'ownllm.pt').read_bytes()).hexdigest() if args.base_export else None,
              'optimizer':'New AdamW optimizer; continuation from weights, not exact optimizer-state resume',
              'vocab_size':cfg.vocab_size,'max_context':cfg.max_seq_len,'corpus_tokens':counts,
              'training_token_presentations':args.steps*args.batch*args.seq_len,
              'initial_validation':initial,'selected_step':selected['step'],'validation':selected['extra']['validation'],
              'test':test,'duration_seconds':time.monotonic()-started,
              'limitations':corpus['limitations']+['Bounded sampled CPU/GPU pretraining, not a complete pass through the corpus.',
                                                  'Training sequences are shorter than the configured maximum context.',
                                                  'No instruction tuning or material-answer accuracy benchmark is performed.',
                                                  'Perplexity measures text prediction, not engineering correctness.']}
    save_checkpoint(out/'final.pt',model,step=selected['step'],extra={'stage':'corpus_pretraining','evaluation':'evaluation.json'})
    write_json(out/'evaluation.json',report)
    export = out/'export'
    export.mkdir()
    shutil.copy2(out/'final.pt',export/'ownllm.pt')
    shutil.copy2(out/'tokenizer/tokenizer.json',export/'tokenizer.json')
    write_json(export/'model_config.json',cfg.to_dict())
    for name in ('evaluation.json','corpus_report.json'):
        shutil.copy2(out/name,export/name)
    summary = f'''# Larger corpus pretraining run

{'Continued training' if args.base_export else 'Trained a fresh model'} with {model.num_parameters():,} parameters and a matching {cfg.vocab_size:,}-token BPE vocabulary. Earlier checkpoints are preserved.

Base export: {args.base_export or 'none (random initialization)'}. Continuation preserves the original tokenizer, with a new AdamW optimizer. {'New corpus source splits; exact prior corpus chunks excluded. Prior paraphrases and alternate editions may remain.' if args.new_corpus else 'Original held-out source split preserved for continuation.'}

- Input: {corpus['input_records']:,} records from {corpus['source_books']} source filenames.
- Retained chunks: {corpus['retained_chunks']:,}; removed exact duplicate or empty chunks: {corpus['removed_duplicate_or_empty_chunks']:,}.
- Excluded exact prior-corpus chunks: {corpus['excluded_prior_chunks']:,}.
- Source-group split counts: { {k:len(v) for k,v in corpus['splits'].items()} }.
- Token counts: {counts}.
- Training: {args.steps} steps, batch {args.batch}, sequence {args.seq_len}; {report['training_token_presentations']:,} token presentations. This is a sampled run, not full-corpus completion.
- Maximum configured context: {cfg.max_seq_len}; longer-context training is not established by this run.
- Initial validation perplexity: {initial['perplexity']:.2f}; selected validation: {report['validation']['perplexity']:.2f}; held-out test: {test['perplexity']:.2f}.
- Test windows evaluated: {test['evaluated_examples']} of {test['available_examples']}; test target tokens: {test['evaluated_tokens']}.
- Duration: {report['duration_seconds']/60:.1f} minutes; device: {device}.

This is an experimental language-pretraining checkpoint, not an instruction-tuned chat model. Perplexity does not certify factual accuracy. Further corpus training and instruction tuning are required before evaluating engineering answers. The existing strict profile renderer is independent of these weights.

Exact normalized chunks are disjoint across splits; alternate editions and paraphrases can remain. The supplied corpus includes no license declaration. Source/page records were read as data, and no document instructions were executed. No public redistribution was performed.
'''
    (out/'REPORT.md').write_text(summary,encoding='utf-8')
    (export/'REPORT.md').write_text(summary,encoding='utf-8')
    sums = [hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in sorted(export.iterdir())]
    (export/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    assert all(hashlib.sha256((export/line.split('  ',1)[1]).read_bytes()).hexdigest()==line.split('  ',1)[0] for line in sums)
    record({'stage':'complete','weights':str(export/'ownllm.pt'),'test':test,'checksums':'verified'})


if __name__ == '__main__':
    main()
