"""Evaluate complete held-out book windows and report task-specific generation checks."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ownllm import ModelConfig, OwnLLM
from ownllm.data.memmap import MemmapTokenDataset
from train_books import evaluate


def task_checks(prediction, reference, concept):
    if concept.startswith('profile_format:'):
        fields = ('Knowledge', 'Typical Value / Range', 'Unit / Scale', 'Formula / Equation')
        expected = [line for line in reference.splitlines() if any(field in line for field in fields)]
        headings = (('### 1. General Overview', '### 2. Engineering Profile')
                    if '### 1. General Overview' in reference else
                    ('GENERAL OVERVIEW', 'ENGINEERING PROFILE (Filtered Parameters)'))
        return {'required_headings':int(all(h in prediction for h in headings)),
                'all_supplied_fields_copied':int(all(line in prediction for line in expected)),
                'no_na_placeholder':int(not re.search(r'\bN/A\b',prediction,re.I))}
    if concept == 'unknown_fallback':
        return {'exact_fallback':int(prediction.strip() == reference.strip())}
    if concept == 'constraint_selection':
        match = re.search(r'Choose ([ABC])\b',reference)
        if match:
            choices = re.findall(r'\bChoose\s+([ABC])\b',prediction,re.I)
            passed = bool(choices) and all(c.upper() == match[1] for c in choices)
        else:
            passed = bool(re.search(r'\bno feasible candidate\b',prediction,re.I)) and not re.search(r'\bChoose\s+[ABC]\b',prediction,re.I)
        return {'selection_matches_reference':int(passed)}
    if concept.startswith('formula:'):
        expected = re.search(r'Answer:\s*([-+\d.eE]+)\s*(MPa|GPa|J|mm)',reference)
        # Require an explicitly marked final answer and its unit, not a number echoed from the question.
        supplied = re.findall(r'Answer:\s*([-+\d.eE]+)\s*(MPa|GPa|J|mm)\b',prediction,re.I)
        valid = False
        if expected and supplied:
            try:
                value,unit = supplied[-1]
                valid = unit.casefold() == expected[2].casefold() and math.isclose(float(value),float(expected[1]),rel_tol=1e-3,abs_tol=1e-6)
            except ValueError:
                pass
        return {'explicit_final_value_and_unit':int(valid)}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='runs/open_materials_v1')
    parser.add_argument('--sources',default='data/open_materials')
    parser.add_argument('--curriculum',default='data/open_materials_final')
    args = parser.parse_args()
    run, sources, curriculum = Path(args.run),Path(args.sources),Path(args.curriculum)
    report = json.loads((run/'evaluation.json').read_text(encoding='utf-8'))
    corpus = json.loads((run/'corpus_report.json').read_text(encoding='utf-8'))
    torch.set_num_threads(report['run']['threads'])
    dataset = MemmapTokenDataset(sorted((run/'prepared/test/shards').glob('*.bin')),report['run']['seq_len'])
    for label, filename in [('pretrain_book_test_full','pretrain_best.pt'),('book_test_full','final.pt')]:
        state = torch.load(run/filename,map_location='cpu',weights_only=False)
        model = OwnLLM(ModelConfig.from_dict(state['config']))
        model.load_state_dict(state['model'])
        report[label] = evaluate(model,dataset,'cpu',report['run']['batch'],len(dataset),report['run']['seed'])
        print(json.dumps({label:report[label]}),flush=True)
    test = {r['id']:r for r in (json.loads(line) for line in
            (curriculum/'dataset/sft_test.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())}
    predictions = json.loads((run/'predictions.json').read_text(encoding='utf-8'))
    groups = {}
    for prediction in predictions:
        if prediction['split'] != 'sft_test':
            continue
        row = test[prediction['id']]
        kind = row['concept'].split(':')[0]
        values = task_checks(prediction['prediction'],row['output'],row['concept'])
        bucket = groups.setdefault(kind,{'questions':0,'passed':{}})
        bucket['questions'] += 1
        for name,value in values.items():
            bucket['passed'][name] = bucket['passed'].get(name,0)+value
    report['task_specific_checks'] = groups
    report['limitations'].extend([
        'Task-specific checks use patterns and do not certify scientific validity or reasoning.',
        'Book partitions are disjoint page blocks of the same two volumes, not independent source books.',
        'Profile-format exercises use placeholders and do not supply real measurements for 66 properties.',
        'Materials API profiles are enforced by a separate reviewed-record renderer, not by these weights.',
        'The reviewed engineering collection is empty until actual sourced records are reviewed and added.'
    ])
    (run/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    export = run/'export'
    shutil.copy2(run/'pretrain_best.pt',export/'pretrain.pt')
    shutil.copy2(run/'evaluation.json',export/'evaluation.json')
    shutil.copy2(sources/'LICENSE_MANIFEST.json',export/'LICENSE_MANIFEST.json')
    shutil.copy2(sources/'ATTRIBUTION.md',export/'ATTRIBUTION.md')
    shutil.copy2(curriculum/'curriculum_report.json',export/'curriculum_report.json')
    text = f'''# Openly licensed materials training run

Trained from random initialization on extracted text from two specific OpenStax editions declaring CC BY 4.0, followed by synthetic instruction practice. See ATTRIBUTION.md and LICENSE_MANIFEST.json for source editions, license evidence and hashes. Earlier Drive records and checkpoints were excluded from this run.

## Data and model

- Parameters: {report['parameters']:,}; device: {report['device']}.
- Book page-block splits: {len(corpus['splits']['train'])} train, {len(corpus['splits']['validation'])} validation, {len(corpus['splits']['test'])} test.
- Book token counts: {corpus['tokens']}.
- Instruction splits: {report['sft_audit']['counts']}.
- Pretraining: {report['run']['steps']} steps. Instruction tuning: {report['run']['sft_epochs']} epochs; validation selected epoch {report['selected_sft_epoch']}.
- Training and initial evaluation time: {report['duration_seconds']/60:.1f} minutes.

## Evaluation

- Full held-out book perplexity before instruction tuning: {report['pretrain_book_test_full']['perplexity']:.3f}.
- Full held-out book perplexity after instruction tuning: {report['book_test_full']['perplexity']:.3f}.
- Instruction test perplexity: {report['sft_test']['perplexity']:.3f}.
- Generated test exact match: {report['sft_test_generation']['exact_match']:.3%}; token F1: {report['sft_test_generation']['token_f1']:.3f}.
- Numerical benchmark: {report.get('numerical_generation')}.

Task-specific checks (pass counts, not certified accuracy):

```json
{json.dumps(groups,indent=2)}
```

## Limits and serving

These compact weights are experimental. The formatting exercises cover all 66 property group names but contain placeholders rather than real material values. Formula tasks cover four basic calculation types. Test cases share concepts and book volumes with training. Correct tokens or matching numbers do not establish engineering competence.

Strict materials responses use the reviewed local record renderer. Its collection starts empty, so missing verified data produces the exact fallback. Feedback remains unverified and does not update weights. Offline verification checks written scalar arithmetic and limited structural fields, rather than performing a complete scientific review.

`ownllm.pt` is the instruction checkpoint; `pretrain.pt` preserves the book-language checkpoint. Both require this export's tokenizer. `evaluation.json`, predictions in the run directory, and SHA256SUMS provide reviewable evidence. Training code licensing is separate from source-data licensing; source attribution is preserved here.
'''
    (run/'REPORT.md').write_text(text,encoding='utf-8')
    (export/'REPORT.md').write_text(text,encoding='utf-8')
    sums = []
    for path in sorted(export.iterdir()):
        if path.name != 'SHA256SUMS':
            sums.append(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name)
    (export/'SHA256SUMS').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    for line in sums:
        digest,name = line.split('  ',1)
        assert hashlib.sha256((export/name).read_bytes()).hexdigest() == digest
    print(json.dumps({'task_specific_checks':groups,'checksums':'verified','report':str(run/'REPORT.md')}),flush=True)


if __name__ == '__main__':
    main()
