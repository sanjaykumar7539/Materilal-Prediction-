"""Prepare licensed source-conditioned alloy QA with article-grouped holdouts."""
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zipfile

from tokenizers import Tokenizer
from ownllm.training.books import instruction_prompt

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'data/low_temperature_alloys_20261012'


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def main():
    raw = DEST / 'raw/AlloyData-2024LT.json'
    metadata = json.loads((DEST / 'raw/metadata.json').read_text())
    assert metadata['license']['name'] == 'CC BY 4.0'
    assert hashlib.md5(raw.read_bytes()).hexdigest() == metadata['files'][0]['computed_md5']
    dataset = json.loads(raw.read_text(encoding='utf-8'))['L-T']
    assert dataset['unit']['yield_strength'] == 'MPa' and dataset['unit']['temperature_default'] == 'K'
    known_dois = {a['metadata'].get('doi', '').strip().lower() for a in dataset['articles']}
    existing_dois, scanned, scanned_bytes = set(), 0, 0
    pattern = re.compile(rb'10\.\d{4,9}/[^\s"<>\\]+', re.I)
    for path in (ROOT / 'data').rglob('*'):
        if not path.is_file() or DEST in path.parents or path.suffix.lower() not in {'.json', '.jsonl', '.txt', '.md'}:
            continue
        scanned += 1
        with path.open('rb') as handle:
            tail = b''
            while chunk := handle.read(2 * 1024 * 1024):
                scanned_bytes += len(chunk)
                block = tail + chunk
                for match in pattern.findall(block):
                    doi = match.decode('ascii', errors='ignore').lower().rstrip('.,;)]}')
                    if doi in known_dois:
                        existing_dois.add(doi)
                tail = block[-512:]
    records, rejected, fingerprints = [], collections.Counter(), set()
    for article_index, article in enumerate(dataset['articles']):
        doi = article['metadata'].get('doi', '').strip().lower()
        for record_index, record in enumerate(article.get('scidata', {}).get('datasets', [])):
            if not doi.startswith('10.') or doi in existing_dois:
                rejected['missing_or_preexisting_source_doi'] += 1
                continue
            material = record.get('materials', {})
            prop = record.get('mech_prop', {})
            name, value, temp = material.get('mat_name'), prop.get('yield_strength'), prop.get('temperature')
            if not isinstance(name, str) or not name.strip() or not finite(value) or value <= 0 or not finite(temp) or temp <= 0:
                rejected['missing_or_invalid_identity_yield_temperature'] += 1
                continue
            ultimate = prop.get('ultimate_strength')
            if finite(ultimate) and ultimate < value:
                rejected['yield_exceeds_ultimate_strength'] += 1
                continue
            composition = material.get('composition')
            if not isinstance(composition, list) or not composition or not material.get('ratio_type'):
                rejected['missing_composition_or_basis'] += 1
                continue
            if not record.get('processing', {}).get('proc_para') or record.get('testing', {}).get('test_type') != 'tensile':
                rejected['missing_processing_or_non_tensile_test'] += 1
                continue
            tested = record.get('testing', {}).get('test_tem')
            if not isinstance(tested, list) or temp not in tested:
                rejected['inconsistent_test_temperature'] += 1
                continue
            fingerprint = hashlib.sha256(json.dumps([material, prop, record['processing'], record['testing']], sort_keys=True).encode()).hexdigest()
            if fingerprint in fingerprints:
                rejected['duplicate_measurement'] += 1
                continue
            fingerprints.add(fingerprint)
            rid = f'LT9-A{article_index+1}-R{record_index+1}'
            question = (f'Read the supplied alloy record and report only its yield strength and test temperature. '
                        f'Do not predict an unmeasured alloy.\nRecord: {rid}; material: {name}; '
                        f'yield strength: {value:g} MPa; test temperature: {temp:g} K. '
                        f'Values apply only to this dataset specimen and its recorded processing.')
            answer = (f'{name}: reported yield strength {value:g} MPa at {temp:g} K. '
                      'This is a dataset-reported measurement for the specified specimen, not a prediction or a universal grade value.')
            records.append({'id':rid,'doi':doi,'question':question,'answer':answer,'verified':False,
                            'value':value,'unit':'MPa','temperature_K':temp,'material':name,
                            'article_metadata':article['metadata'],'source_record':record,
                            'evidence':'Dataset-reported experimental measurement; original paper not independently rechecked.'})
    tokenizer = Tokenizer.from_file(str(ROOT / 'runs/chemistry_properties_v4/export/tokenizer.json'))
    eligible = []
    for record in records:
        row = {'instruction':record['question'],'input':'','output':record['answer']}
        length = len(tokenizer.encode(instruction_prompt(row)).ids) + len(tokenizer.encode(row['output']).ids) + 2
        if length <= 193:
            record['tokens'] = length
            eligible.append(record)
        else:
            rejected['exceeds_192_token_training_window'] += 1
    rng = random.Random(20261012)
    groups = sorted({r['doi'] for r in eligible}); rng.shuffle(groups)
    n = max(1, round(len(groups) * .12))
    allocation = {doi: ('test' if i < n else 'validation' if i < 2*n else 'train') for i,doi in enumerate(groups)}
    splits = {}
    for split, limit in [('train',512),('validation',64),('test',64)]:
        candidates = [r for r in eligible if allocation[r['doi']] == split]; rng.shuffle(candidates)
        per_article = collections.Counter(); selected = []
        for record in candidates:
            if per_article[record['doi']] >= 8 or len(selected) >= limit:
                continue
            selected.append(record); per_article[record['doi']] += 1
        splits[split] = selected
        assert selected, split
    for left, right in [('train','test'),('train','validation'),('validation','test')]:
        assert not {r['doi'] for r in splits[left]} & {r['doi'] for r in splits[right]}
        assert not {r['question'] for r in splits[left]} & {r['question'] for r in splits[right]}
    for split, rows in splits.items():
        (DEST / (split+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    with zipfile.ZipFile(DEST / 'sft.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for split, rows in splits.items():
            archive.writestr(f'split_dataset/reviewed_qa/{split}.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
            archive.writestr(f'split_dataset/rag_knowledge/{split}.jsonl', '')
    provenance = {
        'license':'CC BY 4.0; Haoxuan Tang and Zhiping Xu, Figshare article 25912267 version 9.',
        'source_url':'https://doi.org/10.6084/m9.figshare.25912267.v9',
        'license_url':metadata['license']['url'],'retrieved':'2026-10-12',
        'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),
        'raw_md5_matches_publisher':True,
        'evaluation_scope':'Source-conditioned extraction on unseen article DOI groups; answers are supplied in the prompt. This is not closed-book knowledge, new-alloy prediction, or an independent scientific-accuracy benchmark.',
        'limitations':['Publisher dataset was machine-extracted and manually corrected; original papers were not individually rechecked.',
                      'Source records are not marked independently verified or admitted into the verified chat knowledge base.',
                      'Full composition, processing and testing metadata are retained in the dataset; compact training prompts identify the specimen but do not reproduce all metadata.',
                      'Prior-corpus novelty scan covers local data JSON/JSONL/TXT/MD DOI mentions; undetected paraphrases, PDFs, archives and undocumented historical exposures may overlap.',
                      'Only a bounded, article-diverse subset is fine-tuned; materials and related compositions can occur across DOI groups.'],
        'raw_articles':len(dataset['articles']),'raw_records':sum(len(a.get('scidata',{}).get('datasets',[])) for a in dataset['articles']),
        'eligible_records':len(eligible),'eligible_doi_groups':len(groups),'rejections':dict(rejected),
        'counts':{s:len(v) for s,v in splits.items()},'split_doi_groups':{s:len({r['doi'] for r in v}) for s,v in splits.items()},
        'overlap_scan':{'files':scanned,'bytes':scanned_bytes,'preexisting_dois_excluded':sorted(existing_dois)},
        'max_tokens':{s:max(r['tokens'] for r in v) for s,v in splits.items()},
        'seed':20261012,'per_article_cap':8,
    }
    (DEST / 'PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    (DEST / 'ATTRIBUTION.md').write_text('# Attribution\n\nHaoxuan Tang and Zhiping Xu. Mechanical performance dataset for alloy with applications at low temperatures, version 9. https://doi.org/10.6084/m9.figshare.25912267.v9\n\nCC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . Download metadata and publisher checksum retained in raw/. Changes: quality filters, DOI-grouped sampling and source-conditioned question/answer conversion. Original per-record article citations and full metadata are retained. No publisher endorsement is implied.\n',encoding='utf-8')
    print(json.dumps(provenance,indent=2))


if __name__ == '__main__':
    main()
