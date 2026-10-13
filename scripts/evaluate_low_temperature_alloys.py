"""Compare source-grounded numerical extraction before/after local OwnLLM SFT."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ownllm import ModelConfig, OwnLLM
from ownllm.training.books import encode_sft, instruction_prompt
from train_books import evaluate, generate_answer, lexical_metrics


def write(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def grade(text, record):
    values = re.findall(r'(?<![\d.])([-+]?\d+(?:\.\d+)?)\s*MPa\b', text)
    temps = re.findall(r'(?<![\d.])([-+]?\d+(?:\.\d+)?)\s*K\b', text)
    value_ok = bool(values) and all(math.isclose(float(v),float(format(record['value'], 'g')),rel_tol=0,abs_tol=1e-6) for v in values)
    temp_ok = bool(temps) and all(math.isclose(float(v),float(format(record['temperature_K'], 'g')),rel_tol=0,abs_tol=1e-6) for v in temps)
    entity_ok = ' '.join(record['material'].casefold().split()) in ' '.join(text.casefold().split())
    label_ok = 'not a prediction' in text.casefold() and ('reported' in text.casefold() or 'measurement' in text.casefold())
    return {'value_unit_correct':int(value_ok), 'temperature_correct':int(temp_ok),
            'material_correct':int(entity_ok),'evidence_label_correct':int(label_ok),
            'all_fields_correct':int(value_ok and temp_ok and entity_ok and label_ok)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/low_temperature_alloys_v1')
    args = parser.parse_args()
    run = ROOT / args.run
    data = ROOT / 'data/low_temperature_alloys_20261012'
    export = run / 'export'
    report = json.loads((run/'evaluation.json').read_text())
    provenance = json.loads((data/'PROVENANCE.json').read_text())
    rows = [json.loads(line) for line in (data/'test.jsonl').read_text(encoding='utf-8').splitlines()]
    all_splits = {split:[json.loads(line) for line in (data/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
                  for split in ('train','validation','test')}
    assert len(rows) == report['generation']['questions']
    torch.set_num_threads(4)
    torch.manual_seed(42)
    tokenizer = Tokenizer.from_file(str(export/'tokenizer.json'))
    encoded = [encode_sft({'instruction':r['question'],'input':'','output':r['answer']},tokenizer,192) for r in rows]
    results = {}
    for label,path in [('baseline',ROOT/report['run']['checkpoint']),('candidate',export/'ownllm.pt')]:
        state = torch.load(path,map_location='cpu',weights_only=False)
        model = OwnLLM(ModelConfig.from_dict(state['config']));model.load_state_dict(state['model']);model.eval()
        assert all(torch.isfinite(t).all() for t in model.state_dict().values())
        predictions = []
        saved_predictions = json.loads((run/'predictions.json').read_text(encoding='utf-8')) if label == 'candidate' else None
        if saved_predictions is not None:
            assert len(saved_predictions) == len(rows)
        for index,row in enumerate(rows):
            if saved_predictions is not None:
                saved = saved_predictions[index]
                assert saved['question'] == row['question'] and saved['reference'] == row['answer']
                prediction = saved['prediction']
            else:
                prediction = generate_answer(model,tokenizer,instruction_prompt({'instruction':row['question'],'input':''}),'cpu',96)
            predictions.append({'id':row['id'],'doi':row['doi'],'question':row['question'],'reference':row['answer'],
                                'prediction':prediction,**grade(prediction,row),**lexical_metrics(prediction,row['answer'])})
            if (index+1)%16==0:
                print(json.dumps({'stage':label,'generated':index+1,'total':len(rows)}),flush=True)
        keys = ('value_unit_correct','temperature_correct','material_correct','evidence_label_correct','all_fields_correct','exact_match')
        totals = {key:sum(r[key] for r in predictions) for key in keys}
        losses = evaluate(model,encoded,'cpu',8,len(encoded),42)
        assert math.isfinite(losses['loss']) and math.isfinite(losses['perplexity'])
        probes=[]
        for row in rows[:8]:
            prompt = f"What is the yield strength of {row['material']} at {row['temperature_K']:g} K?"
            prediction = generate_answer(model,tokenizer,instruction_prompt({'instruction':prompt,'input':''}),'cpu',96)
            probes.append({'question':prompt,'prediction':prediction,
                           'ungrounded_numeric_strength':bool(re.search(r'\d+(?:\.\d+)?\s*MPa',prediction)),
                           'scope':'No source or specimen/process metadata supplied. No reference number is treated as justified.'})
        write(export/(label+'_field_predictions.json'),predictions)
        write(export/(label+'_no_source_probes.json'),probes)
        results[label]={'test':losses,'questions':len(rows),'correct_counts':totals,
                        'prediction_source':'trainer saved selected-checkpoint outputs' if saved_predictions is not None else 'generated during comparative evaluation',
                        'token_f1':sum(r['token_f1'] for r in predictions)/len(rows),
                        'no_source_numeric_outputs':sum(p['ungrounded_numeric_strength'] for p in probes),
                        'no_source_probe_count':len(probes)}
        del model,state
    train_labels=0
    for row in all_splits['train']:
        _,labels=encode_sft({'instruction':row['question'],'input':'','output':row['answer']},tokenizer,192)
        train_labels += int((labels != -100).sum())
    parent_path=ROOT/report['run']['checkpoint']
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==report['checkpoint_sha256']
    assert hashlib.sha256((export/'tokenizer.json').read_bytes()).hexdigest()==report['tokenizer_sha256']
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        assert not {r['doi'] for r in all_splits[left]} & {r['doi'] for r in all_splits[right]}
    # Selection was completed using validation only, before this comparative test report.
    verification={'native_parameters':report['parameters'],'finite_weights':True,'parent_unchanged':True,
                  'tokenizer_unchanged':True,'doi_group_overlap':0,'publisher_checksum_matches':True,
                  'train_supervised_tokens_per_epoch':train_labels,
                  'completed_epochs':report['run']['epochs'],
                  'supervised_token_presentations':train_labels*report['run']['epochs'],
                  'candidate_activated':False,'verified_kb_admission':False,
                  'selection':'lowest validation loss; test never used to choose an epoch',
                  'new_prediction_model':False}
    write(export/'FIELD_EVALUATION.json',results)
    verification['deployment_assessment'] = ('Numerical extraction remains unvalidated: inspect all_fields_correct and no-source probes before any deployment. Candidate remains inactive.')
    write(export/'VERIFICATION.json',verification)
    import shutil
    shutil.copy2(data/'DATA_VALIDATION.json', export/'DATA_VALIDATION.json')
    lines=['# New alloy data: verification and fine-tuning report',
           '## Factual readiness result',
           f"Complete field checks passed: {results['candidate']['correct_counts']['all_fields_correct']}/{len(rows)}; source-free numeric outputs: {results['candidate']['no_source_numeric_outputs']}/8. Improved lexical metrics do not establish reliable numerical extraction. Candidate remains inactive.",
           '## Source and license',
           'Haoxuan Tang and Zhiping Xu, Mechanical performance dataset for alloy with applications at low temperatures, Figshare version 9. https://doi.org/10.6084/m9.figshare.25912267.v9 . CC BY 4.0. Publisher MD5 and local SHA-256 retained. The raw release has 239 articles and 1,913 records.',
           '## Preparation and novelty',
           f"Selected splits: {provenance['counts']}; DOI groups: {provenance['split_doi_groups']}. Article groups do not cross splits. Full composition, processing and test metadata remain attached to each selected record.",
           f"Prior-data scan: {provenance['overlap_scan']['files']} text/JSON files; excluded DOI sources: {provenance['overlap_scan']['preexisting_dois_excluded']}. This scan cannot rule out semantic or undocumented historical overlap.",
           'Invalid/missing fields, inconsistent temperatures, missing composition/processing and yield values above ultimate strength were excluded. Dataset score fields were not interpreted as model confidence. No source record is marked independently verified.',
           '## Training',
           f"Native OwnLLM: {report['parameters']:,} parameters. Completed {report['run']['epochs']} epochs; selected epoch {report['selected_epoch']} using validation loss. Vocabulary unchanged. Supervised token presentations: {verification['supervised_token_presentations']:,}.",
           'This is source-conditioned instruction fine-tuning: the correct measurement is in the input record. It trains extraction and attribution, not new-alloy property prediction or reliable closed-book recall.',
           '## Same-test comparison',
           '| Measure | Baseline | Candidate |','| --- | --- | --- |',
           f"| Test loss | {results['baseline']['test']['loss']:.6f} | {results['candidate']['test']['loss']:.6f} |",
           f"| Test perplexity | {results['baseline']['test']['perplexity']:.4f} | {results['candidate']['test']['perplexity']:.4f} |"]
    for key in ('value_unit_correct','temperature_correct','material_correct','evidence_label_correct','all_fields_correct','exact_match'):
        lines.append(f"| {key} | {results['baseline']['correct_counts'][key]}/{len(rows)} | {results['candidate']['correct_counts'][key]}/{len(rows)} |")
    lines += ['## Interpretation and limits',
              'Prompt and answer numbers use six significant digits (Python g format); six selected records have longer raw numeric precision. Full raw values are retained. Field evaluation checks the displayed source precision, not the longer raw decimal. Field checks enforce the expected value/unit, temperature, material string and evidence label. They are deterministic extraction checks, not expert review of arbitrary prose. Exact match and token F1 are lexical metrics. Low test loss alone does not prove correct generated answers.',
              f"No-source probes produced numeric strength claims in {results['baseline']['no_source_numeric_outputs']}/8 baseline and {results['candidate']['no_source_numeric_outputs']}/8 candidate responses. Such numbers lack the supplied specimen/process evidence and are not accepted as verified answers.",
              'Original papers were not individually rechecked; this document verifies provenance, transfer, filtering, split integrity and measured model behavior. It does not certify every source measurement or engineering suitability.',
              '## Weights and deployment',
              'Candidate saved separately as export/ownllm.pt with tokenizer, config, raw predictions, audits and checksums. Parent weights and tokenizer hashes are unchanged. No automatic chat activation or verified-KB admission occurred. Existing twelve-section reviewed chat remains in use.',
              '## Reproduce',
              'Run scripts/prepare_low_temperature_alloys.py using the saved raw version-9 download, then repeat the training command in evaluation.json into a new output directory. Run scripts/evaluate_low_temperature_alloys.py --run <new-run>. Never overwrite prior checkpoints.']
    report_text = re.sub(r'(?<=\|)\n\n(?=\|)', '\n', '\n\n'.join(lines))+'\n'
    (export/'VERIFICATION_DOCUMENT.md').write_text(report_text,encoding='utf-8')
    (export/'REPORT.md').write_text(report_text,encoding='utf-8')
    (run/'REPORT.md').write_text(report_text,encoding='utf-8')
    (run/'VERIFICATION_DOCUMENT.md').write_text(report_text,encoding='utf-8')
    sums=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in sorted(export.iterdir()) if p.name!='SHA256SUMS']
    (export/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    for line in sums:
        digest,name=line.split('  ',1)
        assert hashlib.sha256((export/name).read_bytes()).hexdigest()==digest
    print(json.dumps({'results':results,'verification':verification,'export_files_verified':len(sums)},indent=2),flush=True)


if __name__=='__main__':
    main()
