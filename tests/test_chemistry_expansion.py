import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/chemistry_expansion'
KB = ROOT/'domains/materials/verified_records.json'


def test_reviewed_descriptors_and_book_links_do_not_promote_unverified_values(tmp_path):
    policy = MaterialsPolicy(KB,tmp_path/'feedback.sqlite3')
    hydrogen = policy.profile('element hydrogen')
    assert hydrogen.count('**Unit / Scale:**') == 2
    assert 'Related reading:' not in hydrogen and 'https://' not in hydrogen
    assert policy.profile('element hydrogen',parameter='Boiling Point') == fallback('Hydrogen (element)')
    water = policy.profile('compound water',parameter='Calculated Molecular Mass')
    assert '18.015' in water and 'Da' in water and 'calculated' in water
    assert policy.profile('compound water',parameter='Yield Strength') == fallback('Water (compound)')
    collection = json.loads(KB.read_text(encoding='utf-8'))
    props = [p for r in collection['materials'] for p in r['properties']]
    assert sum(not p['verified'] for p in props) == 878
    audit = json.loads((KB.parent/'CHEMISTRY_EXPANSION_REVIEW.json').read_text(encoding='utf-8'))
    assert audit['knowledge_base_sha256'] == hashlib.sha256(KB.read_bytes()).hexdigest()
    assert audit['verified_properties'] == 959


def test_sodium_symbol_is_not_an_unavailable_placeholder(tmp_path):
    from ownllm.knowledge.materials_policy import _meaningful
    assert _meaningful('Na')
    assert not _meaningful('NA') and not _meaningful('N/A')
    response = MaterialsPolicy(KB,tmp_path/'feedback.sqlite3').profile('element sodium')
    assert '**Chemical Symbol:**' in response
    assert '**Typical Value / Range:** Na (Conditions:' in response


@pytest.mark.skipif(not (DATA/'property_sft.zip').exists(),reason='Optional downloaded chemistry data absent')
def test_property_training_is_disjoint_and_preserves_uncertainties():
    with zipfile.ZipFile(DATA/'property_sft.zip') as archive:
        rows = {s:[json.loads(l) for l in archive.read(f'split_dataset/reviewed_qa/{s}.jsonl').decode('utf-8').splitlines()] for s in ['train','validation','test']}
    prompts = {s:{r['question'] for r in values} for s,values in rows.items()}
    assert prompts['train'].isdisjoint(prompts['validation'])
    assert prompts['train'].isdisjoint(prompts['test'])
    assert prompts['test'].isdisjoint(prompts['validation'])
    assert len({r['entity'] for r in rows['train'] if r['kind']=='element'}) == 118
    assert len({r['entity'] for r in rows['train'] if r['kind']=='compound'}) == 200
    for values in rows.values():
        for row in values:
            assert 'Local source record:' in row['question']
            assert 'Conditions:' in row['answer']
            if row['data_kind']=='unconfirmed':
                assert 'Do not treat as verified experimental data' in row['answer']
    report = json.loads((DATA/'PROVENANCE.json').read_text(encoding='utf-8'))
    assert report['unconfirmed_properties'] == 72
    assert len(report['books']) == 2
    assert all(b['license']=='CC BY 4.0' for b in report['books'])
