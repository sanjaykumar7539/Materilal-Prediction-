"""Train, instruction-tune, evaluate, and export OwnLLM on a local book collection."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import re
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ownllm import ModelConfig, OwnLLM
from ownllm.data.memmap import MemmapTokenDataset
from ownllm.data.sharding import shard_encoded_documents
from ownllm.inference.generate import token_stream
from ownllm.tokenizer_train import train_bpe
from ownllm.training.books import SPECIAL_TOKENS, audit_sft, cached_documents, encode_sft, instruction_prompt, normalize, read_sft, split_documents
from ownllm.training.checkpoint import save_checkpoint
from ownllm.training.engine import Trainer

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

@torch.inference_mode()
def evaluate(model, dataset, device, batch_size, max_batches, seed):
    if not len(dataset):
        raise ValueError('Evaluation split has no usable token windows')
    model.eval()
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = indices[:max_batches * batch_size]
    total_loss, count = 0.0, 0
    for start in range(0, len(indices), batch_size):
        samples = [dataset[index] for index in indices[start:start+batch_size]]
        ids, labels = (torch.stack([sample[i] for sample in samples]).to(device) for i in range(2))
        valid_positions = (labels != -100).any(dim=0).nonzero()
        if valid_positions.numel():
            length = int(valid_positions[-1].item()) + 1
            ids, labels = ids[:,:length], labels[:,:length]
        _, loss, _ = model(ids, labels)
        tokens = int((labels != -100).sum().item())
        total_loss += loss.item() * tokens
        count += tokens
    if not count:
        raise ValueError('Evaluation split has no target tokens')
    average = total_loss / count
    return {'loss':average, 'perplexity':math.exp(min(average, 20)), 'evaluated_tokens':count, 'evaluated_examples':len(indices), 'available_examples':len(dataset)}

def lexical_metrics(prediction, reference):
    predicted, expected = normalize(prediction).split(), normalize(reference).split()
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    precision = overlap / max(1, len(predicted))
    recall = overlap / max(1, len(expected))
    return {'exact_match':int(normalize(prediction) == normalize(reference)), 'token_f1':2*precision*recall/max(1e-12,precision+recall)}

@torch.inference_mode()
def generate_answer(model, tokenizer, prompt, device, max_new):
    ids = tokenizer.encode(prompt).ids[-model.cfg.max_seq_len:]
    tensor = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []
    for token in token_stream(model, tensor, max_new=max_new, eos=tokenizer.token_to_id('<eos>'), temperature=0):
        value = int(token.item())
        if value == tokenizer.token_to_id('<|im_end|>'):
            break
        generated.append(value)
    return tokenizer.decode(generated)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/drive_books')
    parser.add_argument('--books', help='Reuse an existing book directory without copying it')
    parser.add_argument('--out', default='runs/books_cpu')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--sft-epochs', type=int, default=20)
    parser.add_argument('--vocab', type=int, default=4096)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--eval-batches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.0006)
    parser.add_argument('--config', help='Optional ModelConfig JSON for a larger GPU run')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--reuse-prepared', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.threads, args.eval_batches) < 1 or args.seq_len < 8 or args.sft_epochs < 0:
        parser.error('Steps, batch, threads and evaluation batches must be positive; sequence length >=8')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    sampling = random.Random(args.seed)
    source, out = Path(args.input), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(out/'extraction.log',encoding='utf-8'))
    if (out / 'best.pt').exists() and not args.prepare_only:
        raise SystemExit('This output already has a checkpoint. Choose a new --out to preserve it.')
    prepared = out / 'prepared'
    sft_source = source / 'dataset'
    books_source = Path(args.books) if args.books else source/'books'
    sft = {name:read_sft(sft_source / f'sft_{name}.jsonl') for name in ('train','validation','test')}
    sft_audit = audit_sft(sft)
    if not args.reuse_prepared:
        if prepared.exists():
            raise SystemExit('Prepared data already exists. Use --reuse-prepared or a new --out.')
        def extracted_documents():
            for index, document in enumerate(cached_documents(books_source,books_source.parent/'extraction_cache'),1):
                if index % 10 == 0:
                    print(json.dumps({'stage':'extraction','readable_documents':index,'latest':document[0]}),flush=True)
                yield document
        book_splits, extraction = split_documents(extracted_documents(), args.seed)
        extraction['sources'] = json.loads((books_source.parent/'extraction_cache/report.json').read_text(encoding='utf-8'))
        for name, records in book_splits.items():
            folder = prepared / name
            folder.mkdir(parents=True, exist_ok=True)
            for index, record in enumerate(records):
                (folder / f'{index:05d}.txt').write_text(record['text'], encoding='utf-8')
        training_text = prepared / 'sft_training.txt'
        training_text.write_text('\n'.join(instruction_prompt(row)+row['output']+'<|im_end|><eos>' for row in sft['train']), encoding='utf-8')
        tokenizer = train_bpe([*sorted((prepared/'train').glob('*.txt')),training_text], out/'tokenizer', args.vocab, special_tokens=SPECIAL_TOKENS)
        eos = tokenizer.token_to_id('<eos>')
        shard_counts = {}
        for name in book_splits:
            encoded = (tokenizer.encode(path.read_text(encoding='utf-8')).ids+[eos] for path in sorted((prepared/name).glob('*.txt')))
            metadata = shard_encoded_documents(encoded, prepared/name/'shards')
            shard_counts[name] = sum(row['tokens'] for row in metadata)
        corpus = {'seed':args.seed, 'splits':{name:[{k:v for k,v in row.items() if k != 'text'} for row in records] for name,records in book_splits.items()}, 'tokens':shard_counts, 'extraction':extraction, 'sft':sft_audit, 'deduplication':'SHA256 of case-folded whitespace-normalized document text; near-duplicate editions are not guaranteed to be detected'}
        write_json(out/'corpus_report.json', corpus)
    else:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(out/'tokenizer/tokenizer.json'))
        corpus = json.loads((out/'corpus_report.json').read_text(encoding='utf-8'))
        if corpus['seed'] != args.seed or corpus['sft'] != sft_audit:
            raise SystemExit('Prepared split seed or instruction dataset changed; prepare a new output.')
    if args.prepare_only:
        print(json.dumps({'prepared':str(out), 'tokens':corpus['tokens'], 'documents':{k:len(v) for k,v in corpus['splits'].items()}}), flush=True)
        return
    raw_config = json.loads(Path(args.config).read_text()) if args.config else None
    cfg = ModelConfig.from_dict(raw_config.get('model',raw_config)) if raw_config else ModelConfig(dim=192,n_layers=4,n_heads=6,n_kv_heads=2,ffn_hidden=512,max_seq_len=512)
    cfg.vocab_size = tokenizer.get_vocab_size()
    if args.seq_len > cfg.max_seq_len:
        raise SystemExit('Training sequence length exceeds model context length')
    write_json(out/'model_config.json',cfg.to_dict())
    datasets = {name:MemmapTokenDataset(sorted((prepared/name/'shards').glob('*.bin')),args.seq_len) for name in ('train','validation','test')}
    if any(not len(ds) for ds in datasets.values()):
        raise SystemExit('Not enough tokens in one split. Add documents or reduce --seq-len.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = OwnLLM(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.1)
    trainer = Trainer(model,optimizer,amp='none' if device=='cpu' else 'bf16')
    report = {'run':vars(args), 'device':device, 'parameters':model.num_parameters(), 'vocab_size':cfg.vocab_size, 'corpus_tokens':corpus['tokens'], 'sft_audit':sft_audit, 'limitations':['Compact from-scratch baseline; no claim of reliable engineering answers.', 'Instruction splits share concepts; their scores do not measure unseen-concept generalization.', 'Exact match and token F1 measure text overlap, not factual accuracy.', 'Near-duplicate documents may remain after exact text deduplication.']}
    started = time.monotonic()
    report['initial_validation'] = evaluate(model,datasets['validation'],device,args.batch,args.eval_batches,args.seed)
    best = report['initial_validation']['loss']
    save_checkpoint(out/'pretrain_best.pt',model,step=0,extra={'stage':'pretrain','validation':report['initial_validation']})
    log_file = out/'training.jsonl'
    def record(value):
        with log_file.open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    record({'stage':'initial','parameters':report['parameters'],'device':device,'validation':report['initial_validation']})
    for step in range(1,args.steps+1):
        batch = [datasets['train'][sampling.randrange(len(datasets['train']))] for _ in range(args.batch)]
        ids, targets = (torch.stack([sample[i] for sample in batch]).to(device) for i in range(2))
        warmup = min(1.,step/50)
        decay = .1+.9*.5*(1+math.cos(math.pi*step/args.steps))
        optimizer.param_groups[0]['lr'] = args.lr*warmup*decay
        loss = trainer.train_batch([(ids,targets)])
        if not math.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        if step == 1 or step % 50 == 0:
            record({'stage':'pretrain','step':step,'loss':loss,'elapsed_seconds':round(time.monotonic()-started,1)})
        if step % 250 == 0 or step == args.steps:
            score = evaluate(model,datasets['validation'],device,args.batch,args.eval_batches,args.seed)
            record({'stage':'pretrain_validation','step':step,**score})
            save_checkpoint(out/'last.pt',model,optimizer,step=step,extra={'stage':'pretrain','validation':score,'sampling_rng':sampling.getstate(),'torch_rng':torch.get_rng_state()})
            if score['loss'] < best:
                best = score['loss']
                save_checkpoint(out/'pretrain_best.pt',model,step=step,extra={'stage':'pretrain','validation':score})
    state = torch.load(out/'pretrain_best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model'])
    report['pretrain_validation'] = state['extra']['validation']
    report['selected_pretrain_step'] = state['step']
    encoded_sft = {name:[encode_sft(row,tokenizer,args.seq_len) for row in rows] for name,rows in sft.items()}
    report['truncated_sft_answers'] = {name:sum(len(tokenizer.encode(row['output']).ids)+2>args.seq_len for row in rows) for name,rows in sft.items()}
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr/3,weight_decay=.01)
    trainer = Trainer(model,optimizer,amp='none' if device=='cpu' else 'bf16')
    score = evaluate(model,encoded_sft['validation'],device,args.batch,args.eval_batches,args.seed)
    best_sft = score['loss']
    save_checkpoint(out/'best.pt',model,step=0,extra={'stage':'sft','epoch':0,'validation':score})
    report['initial_sft_validation'] = score
    sft_step = 0
    for epoch in range(1,args.sft_epochs+1):
        order = list(range(len(encoded_sft['train'])))
        sampling.shuffle(order)
        losses = []
        for offset in range(0,len(order),args.batch):
            batch = [encoded_sft['train'][index] for index in order[offset:offset+args.batch]]
            ids, targets = (torch.stack([sample[i] for sample in batch]).to(device) for i in range(2))
            length = int((targets != -100).any(dim=0).nonzero()[-1].item()) + 1
            ids, targets = ids[:,:length], targets[:,:length]
            loss = trainer.train_batch([(ids,targets)])
            if not math.isfinite(loss):
                raise RuntimeError('Non-finite instruction training loss')
            losses.append(loss)
            sft_step += 1
        score = evaluate(model,encoded_sft['validation'],device,args.batch,args.eval_batches,args.seed)
        record({'stage':'sft','epoch':epoch,'train_loss':sum(losses)/len(losses),'validation':score,'elapsed_seconds':round(time.monotonic()-started,1)})
        save_checkpoint(out/'last.pt',model,optimizer,step=sft_step,extra={'stage':'sft','epoch':epoch,'validation':score})
        if score['loss'] < best_sft:
            best_sft = score['loss']
            save_checkpoint(out/'best.pt',model,step=sft_step,extra={'stage':'sft','epoch':epoch,'validation':score})
    state = torch.load(out/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()
    report['selected_sft_epoch'] = state['extra']['epoch']
    report['sft_validation'] = state['extra']['validation']
    # Test partitions are used only once after validation-based checkpoint selection.
    report['book_test'] = evaluate(model,datasets['test'],device,args.batch,args.eval_batches,args.seed)
    report['sft_test'] = evaluate(model,encoded_sft['test'],device,args.batch,args.eval_batches,args.seed)
    predictions = []
    def evaluate_questions(rows, label, numerical=False):
        scores = []
        for index,row in enumerate(rows):
            prediction = generate_answer(model,tokenizer,instruction_prompt(row),device,args.max_new_tokens)
            metric = lexical_metrics(prediction,row['output'])
            if numerical:
                pattern = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
                expected = [float(value) for value in re.findall(pattern,row['output'])]
                observed = [float(value) for value in re.findall(pattern,prediction)]
                metric['numeric_value_match'] = int(bool(expected) and any(all(math.isclose(a,b,rel_tol=1e-3,abs_tol=1e-6) for a,b in zip(expected,observed[start:start+len(expected)])) for start in range(max(0,len(observed)-len(expected)+1))))
            predictions.append({'split':label,'id':row.get('id',str(index)),'question':row['instruction'],'reference':row['output'],'prediction':prediction,**metric})
            scores.append(metric)
        keys = ('exact_match','token_f1','numeric_value_match') if numerical else ('exact_match','token_f1')
        return {key:sum(score[key] for score in scores)/max(1,len(scores)) for key in keys} | {'questions':len(scores)}
    report['sft_test_generation'] = evaluate_questions(sft['test'],'sft_test')
    csv_path = sft_source/'evaluation.csv'
    if csv_path.exists():
        with csv_path.open(encoding='utf-8-sig',newline='') as handle:
            questions = [dict(id=row['id'],instruction=row['question'],output=row['reference_answer']) for row in csv.DictReader(handle)]
        report['supplied_evaluation_generation'] = evaluate_questions(questions,'evaluation_csv')
    numerical_path = sft_source/'numerical_problems.jsonl'
    if numerical_path.exists():
        rows = [json.loads(line) for line in numerical_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        numerical = [dict(id=row['id'],instruction=row['question'],output=row['answer']) for row in rows]
        report['numerical_generation'] = evaluate_questions(numerical,'numerical_problems',numerical=True)
        report['limitations'].append('Numeric value matching does not verify units, reasoning, or physical validity.')
    write_json(out/'predictions.json',predictions)
    save_checkpoint(out/'final.pt',model,step=state['step'],extra={'stage':'complete','selected_sft_epoch':report['selected_sft_epoch'],'report':'evaluation.json'})
    report['duration_seconds'] = time.monotonic()-started
    report['weights'] = str(out/'final.pt')
    report['tokenizer'] = str(out/'tokenizer/tokenizer.json')
    write_json(out/'evaluation.json',report)
    export = out/'export'
    export.mkdir(exist_ok=True)
    shutil.copy2(out/'final.pt',export/'ownllm.pt')
    shutil.copy2(out/'tokenizer/tokenizer.json',export/'tokenizer.json')
    write_json(export/'model_config.json',cfg.to_dict())
    write_json(export/'evaluation.json',report)
    with (export/'SHA256SUMS').open('w',encoding='utf-8') as handle:
        for path in sorted(export.iterdir()):
            if path.name != 'SHA256SUMS':
                with path.open('rb') as binary:
                    handle.write(hashlib.file_digest(binary,'sha256').hexdigest()+'  '+path.name+'\n')
    record({'stage':'complete','weights':report['weights'],'book_test':report['book_test'],'sft_test':report['sft_test'],'duration_seconds':report['duration_seconds']})

if __name__ == '__main__':
    main()
