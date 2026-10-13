"""Fine-tune an existing local checkpoint on the user's split QA/RAG ZIP."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import zipfile

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ownllm import ModelConfig, OwnLLM
from ownllm.training.books import audit_sft, encode_sft, instruction_prompt, normalize
from ownllm.training.checkpoint import save_checkpoint
from ownllm.training.engine import Trainer
from ownllm.training.factual import critical_token_weights
from train_books import evaluate, generate_answer, lexical_metrics, write_json


def prepare(archive_path, out):
    """Read only six expected JSONL members; give held-out questions precedence."""
    raw, members = {}, []
    with zipfile.ZipFile(archive_path) as archive:
        for group in ('reviewed_qa','rag_knowledge'):
            for split in ('train','validation','test'):
                name = f'split_dataset/{group}/{split}.jsonl'
                if archive.getinfo(name).file_size > 64*1024*1024:
                    raise ValueError('Dataset member exceeds the 64 MiB limit')
                content = archive.read(name)
                rows = [json.loads(line) for line in content.decode('utf-8-sig').splitlines() if line.strip()]
                if not all(isinstance(row,dict) for row in rows):
                    raise ValueError(f'Non-object record in {name}')
                raw[group,split] = rows
                members.append({'name':name,'rows':len(rows),'sha256':hashlib.sha256(content).hexdigest()})
    splits = {name:[] for name in ('train','validation','test')}
    seen, removed = {}, []
    for split in ('test','validation','train'):
        for index,row in enumerate(raw['reviewed_qa',split]):
            question,answer = row.get('question'),row.get('answer')
            if not isinstance(question,str) or not question.strip() or not isinstance(answer,str) or not answer.strip():
                raise ValueError(f'Invalid QA record: {split}:{index+1}')
            key = normalize(question)
            if key in seen:
                previous = seen[key]
                removed.append({'split':split,'row':index+1,'kept_split':previous['split'],
                                'question':question,'different_answer':normalize(answer)!=previous['answer']})
                continue
            seen[key] = {'split':split,'answer':normalize(answer)}
            splits[split].append(dict(id=f'qa-{split}-{index+1}',instruction=question.strip(),input='',
                                     output=answer.strip(),module='Supplied QA',concept='materials_qa',
                                     source_id='supplied_qa',supplied_verified=row.get('verified')))
        for row in raw['rag_knowledge',split]:
            if not isinstance(row.get('concept'),str) or not isinstance(row.get('id'),str):
                raise ValueError('RAG record needs concept and id')
            for field in ('definition','units_formula','test','standard','factors','graph','comparison','application'):
                value = row.get(field)
                if not isinstance(value,str) or not value.strip():
                    continue
                splits[split].append(dict(id=f'rag-{row["id"]}-{field}',
                    instruction=f'Describe {field.replace("_"," ")} for {row["concept"]}.',input='',
                    output=value.strip(),module=row.get('module','Supplied RAG'),concept=row['concept'],
                    source_id=row['id'],review_status=row.get('review_status')))
    report = {'archive_sha256':hashlib.sha256(Path(archive_path).read_bytes()).hexdigest(),
              'members':members,'removed_questions':removed,'audit':audit_sft(splits),
              'precedence':'test, then validation, then train; one record per normalized QA question',
              'limitations':['No license declaration is included in the supplied ZIP.',
                             'Supplied verified flags are preserved as metadata, not independent factual verification.',
                             'Splits can share concepts, paraphrases and answer facts despite exact-question separation.',
                             'RAG records are educational summaries marked as requiring technical review.',
                             'Records are not automatically admitted to the reviewed engineering-profile collection.']}
    prepared = out/'prepared'
    prepared.mkdir()
    for split,rows in splits.items():
        (prepared/f'sft_{split}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
        write_json(prepared/f'rag_{split}.json',raw['rag_knowledge',split])
    write_json(out/'dataset_report.json',report)
    return splits,report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',required=True)
    parser.add_argument('--checkpoint',default='runs/open_materials_v1/export/pretrain.pt')
    parser.add_argument('--tokenizer',default='runs/open_materials_v1/export/tokenizer.json')
    parser.add_argument('--out',default='runs/split_dataset_v1')
    parser.add_argument('--epochs',type=int,default=4)
    parser.add_argument('--seq-len',type=int,default=256)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.0002)
    parser.add_argument('--max-new-tokens',type=int,default=192)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--provenance',help='Local JSON source/license audit for a prepared dataset')
    parser.add_argument('--critical-field-weight',type=float,default=1,
                        help='Training loss multiplier for entity, value and unit tokens')
    parser.add_argument('--factual-selection',action='store_true',
                        help='Select by generated entity/value/unit accuracy on validation, then loss')
    args = parser.parse_args()
    if min(args.epochs,args.seq_len,args.batch,args.threads,args.max_new_tokens)<1 or args.lr<=0 or not math.isfinite(args.critical_field_weight) or args.critical_field_weight<1:
        parser.error('Training settings must be positive')
    out = Path(args.out)
    if out.exists():
        raise SystemExit('Choose a new output directory to preserve previous runs.')
    out.mkdir(parents=True)
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    splits,dataset_report = prepare(args.archive,out)
    dataset_license = 'not declared; no public redistribution performed'
    evaluation_scope = 'Test cases can share concepts and answer facts with training.'
    if args.provenance:
        source_audit = json.loads(Path(args.provenance).read_text(encoding='utf-8'))
        dataset_license = source_audit['license']
        evaluation_scope = source_audit['evaluation_scope']
        if not isinstance(dataset_license,str) or not isinstance(evaluation_scope,str):
            raise ValueError('Provenance license and evaluation_scope must be strings')
        dataset_report['source_audit'] = source_audit
        dataset_report['limitations'] = source_audit['limitations'] + [evaluation_scope,
            'Records are not automatically admitted to the reviewed engineering-profile collection.']
        write_json(out/'dataset_report.json',dataset_report)
    tokenizer = Tokenizer.from_file(args.tokenizer)
    state = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg = ModelConfig.from_dict(state['config'])
    if cfg.vocab_size != tokenizer.get_vocab_size() or args.seq_len>cfg.max_seq_len:
        raise ValueError('Tokenizer vocabulary or sequence length does not match the checkpoint')
    model = OwnLLM(cfg)
    model.load_state_dict(state['model'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    trainer = Trainer(model,optimizer,amp='none')
    encoded = {name:[encode_sft(row,tokenizer,args.seq_len) for row in rows] for name,rows in splits.items()}
    field_weights = [critical_token_weights(row,tokenizer,args.seq_len,args.critical_field_weight)
                     for row in splits['train']] if args.critical_field_weight>1 else None
    lengths = {name:[len(tokenizer.encode(instruction_prompt(row)).ids)+len(tokenizer.encode(row['output']).ids)+2 for row in rows] for name,rows in splits.items()}
    report = {'run':vars(args),'device':'cpu','parameters':model.num_parameters(),
              'dataset':dataset_report,'length_audit':{name:{'max_tokens':max(values),'over_sequence_limit':sum(v>args.seq_len+1 for v in values)} for name,values in lengths.items()},
              'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'tokenizer_sha256':hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()}
    def record(value):
        with (out/'training.jsonl').open('a',encoding='utf-8') as handle:
            handle.write(json.dumps(value)+'\n')
        print(json.dumps(value),flush=True)
    score = evaluate(model,encoded['validation'],'cpu',args.batch,len(encoded['validation']),args.seed)
    def factual_validation():
        from check_property_predictions import parse, same_value
        passed = 0
        for row in splits['validation']:
            prediction = generate_answer(model,tokenizer,instruction_prompt(row),'cpu',args.max_new_tokens)
            expected, actual = parse(row['output']),parse(prediction)
            name = row['output'].split('General Overview\n',1)[1].split(': ',1)[0]
            overview = prediction.split('General Overview\n',1)[-1].split('\n',1)[0]
            passed += bool(actual and expected and same_value(expected['value'],actual['value'])
                           and expected['unit']==actual['unit'] and overview.startswith(name+': '))
        return passed
    factual_score = factual_validation() if args.factual_selection else 0
    best = (factual_score,-score['loss'])
    report['initial_validation_factual_correct'] = factual_score if args.factual_selection else None
    report['initial_validation'] = score
    save_checkpoint(out/'best.pt',model,extra={'epoch':0,'validation':score})
    record({'stage':'prepared','counts':dataset_report['audit']['counts'],
            'removed':len(dataset_report['removed_questions']),'length_audit':report['length_audit'],'validation':score})
    step = 0
    for epoch in range(1,args.epochs+1):
        order = list(range(len(encoded['train'])))
        rng.shuffle(order)
        losses = []
        for offset in range(0,len(order),args.batch):
            samples = [encoded['train'][index] for index in order[offset:offset+args.batch]]
            ids,labels = [torch.stack([sample[k] for sample in samples]) for k in (0,1)]
            length = int((labels!=-100).any(dim=0).nonzero()[-1])+1
            batch = (ids[:,:length],labels[:,:length])
            if field_weights is not None:
                weights = torch.stack([field_weights[index] for index in order[offset:offset+args.batch]])
                batch = (*batch,weights[:,:length])
            loss = trainer.train_batch([batch])
            if not math.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            losses.append(loss)
            step += 1
            if step % 100 == 0:
                record({'stage':'sft_progress','epoch':epoch,'step':step,'loss':loss,'elapsed_seconds':round(time.monotonic()-started,1)})
        score = evaluate(model,encoded['validation'],'cpu',args.batch,len(encoded['validation']),args.seed)
        factual_score = factual_validation() if args.factual_selection else 0
        score['generated_entity_value_unit_correct'] = factual_score if args.factual_selection else None
        record({'stage':'validation','epoch':epoch,'train_loss':sum(losses)/len(losses),**score,'elapsed_seconds':round(time.monotonic()-started,1)})
        save_checkpoint(out/'last.pt',model,optimizer,step=step,extra={'epoch':epoch,'validation':score})
        candidate = (factual_score,-score['loss'])
        if candidate>best:
            best = candidate
            save_checkpoint(out/'best.pt',model,step=step,extra={'epoch':epoch,'validation':score})
    selected = torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(selected['model'])
    report['selected_epoch'] = selected['extra']['epoch']
    report['validation'] = selected['extra']['validation']
    report['test'] = evaluate(model,encoded['test'],'cpu',args.batch,len(encoded['test']),args.seed)
    save_checkpoint(out/'final.pt',model,step=selected['step'],extra={'stage':'complete','selected_epoch':report['selected_epoch']})
    predictions = []
    for index,row in enumerate(splits['test']):
        prediction = generate_answer(model,tokenizer,instruction_prompt(row),'cpu',args.max_new_tokens)
        predictions.append(dict(id=row['id'],module=row['module'],question=row['instruction'],reference=row['output'],prediction=prediction,**lexical_metrics(prediction,row['output'])))
        if (index+1)%50 == 0:
            record({'stage':'test_generation','completed':index+1,'total':len(splits['test'])})
    report['generation'] = {key:sum(p[key] for p in predictions)/len(predictions) for key in ('exact_match','token_f1')}
    report['generation']['questions'] = len(predictions)
    report['duration_seconds'] = time.monotonic()-started
    report['limitations'] = dataset_report['limitations']+['Exact match and token F1 measure reference overlap, not scientific correctness.',
                                                         'Compact CPU model; no engineering reliability claim.']
    write_json(out/'predictions.json',predictions)
    write_json(out/'evaluation.json',report)
    export = out/'export'
    export.mkdir()
    shutil.copy2(out/'final.pt',export/'ownllm.pt')
    shutil.copy2(args.tokenizer,export/'tokenizer.json')
    write_json(export/'model_config.json',cfg.to_dict())
    shutil.copy2(out/'evaluation.json',export/'evaluation.json')
    shutil.copy2(out/'dataset_report.json',export/'dataset_report.json')
    write_json(export/'PROVENANCE.json',{'base_checkpoint':args.checkpoint,'base_sha256':report['checkpoint_sha256'],
                                       'user_archive':args.archive,'archive_sha256':dataset_report['archive_sha256'],
                                       'user_dataset_license':dataset_license})
    if args.provenance:
        shutil.copy2(args.provenance,export/'SOURCE_PROVENANCE.json')
        attribution = Path(args.provenance).parent/'ATTRIBUTION.md'
        if attribution.exists():
            shutil.copy2(attribution,export/'ATTRIBUTION.md')
    base = Path(args.checkpoint).parent
    for name in ('ATTRIBUTION.md','LICENSE_MANIFEST.json'):
        if (base/name).exists():
            shutil.copy2(base/name,export/('BASE_'+name))
    summary = f'''# Supplied dataset fine-tuning run

Fine-tuned the local book-pretrained checkpoint on the supplied split QA/RAG dataset. Existing checkpoints are preserved.

- Split counts after cleaning and RAG conversion: {dataset_report['audit']['counts']}.
- Removed duplicate/overlapping QA records: {len(dataset_report['removed_questions'])}.
- Epochs: {args.epochs}; selected by validation: {report['selected_epoch']}.
- Test loss/perplexity: {report['test']['loss']:.4f} / {report['test']['perplexity']:.3f}.
- Generated reference exact match: {report['generation']['exact_match']:.2%}; token F1: {report['generation']['token_f1']:.3f}, across {len(predictions)} questions.
- Runtime: {report['duration_seconds']/60:.1f} minutes on CPU.

Dataset license: {dataset_license}. Base textbook attribution is preserved separately and does not assign its license to the supplied dataset. Verified flags are dataset metadata; technical accuracy has not been independently established. {evaluation_scope} Lexical metrics do not certify engineering correctness.

The strict material-profile template and reviewed-data fallback remain enforced separately by the API. This run does not promote supplied QA/RAG records to verified engineering values. See predictions.json and evaluation.json for actual outputs and measured results.
'''
    (out/'REPORT.md').write_text(summary,encoding='utf-8')
    (export/'REPORT.md').write_text(summary,encoding='utf-8')
    sums = [hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in sorted(export.iterdir())]
    (export/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    for line in sums:
        digest,name = line.split('  ',1)
        assert hashlib.sha256((export/name).read_bytes()).hexdigest()==digest
    record({'stage':'complete','weights':str(export/'ownllm.pt'),'generation':report['generation'],'test':report['test'],'checksums':'verified'})


if __name__ == '__main__':
    main()
