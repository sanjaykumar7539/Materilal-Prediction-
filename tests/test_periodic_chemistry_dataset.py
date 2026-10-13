import json
from pathlib import Path
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/periodic_chemistry'
pytestmark = pytest.mark.skipif(not (DATA/'chemistry_sft.zip').exists(),
    reason='Optional downloaded chemistry training data is not present')


def test_chemistry_training_has_complete_element_coverage_and_disjoint_prompts():
    with zipfile.ZipFile(DATA / 'chemistry_sft.zip') as archive:
        splits = {split: [json.loads(line) for line in archive.read(
            f'split_dataset/reviewed_qa/{split}.jsonl').decode('utf-8').splitlines()]
            for split in ['train', 'validation', 'test']}
    elements = {r['entity'] for r in splits['train'] if r['kind'] == 'element'}
    assert len(elements) == 118
    assert {'Hydrogen', 'Helium', 'Uranium', 'Oganesson'} <= elements
    assert len({r['entity'] for r in splits['train'] if r['kind'] == 'compound'}) == 20
    questions = {split: {r['question'].strip().casefold() for r in rows} for split, rows in splits.items()}
    assert questions['train'].isdisjoint(questions['test'])
    assert questions['train'].isdisjoint(questions['validation'])
    assert questions['validation'].isdisjoint(questions['test'])
    for rows in splits.values():
        for row in rows:
            assert '### 1. General Overview' in row['answer']
            assert '### 2. Engineering Profile' in row['answer']
            assert '**Knowledge:**' in row['answer'] and '**Unit / Scale:**' in row['answer']
            assert '**Typical Value / Range:**' in row['answer']
            assert 'N/A' not in row['answer']
            assert row.get('verified') is not True


def test_compound_formulas_and_license_provenance_are_retained():
    provenance = json.loads((DATA / 'PROVENANCE.json').read_text(encoding='utf-8'))
    records = {r['name']:r for r in provenance['compound_records']}
    assert records['water']['formula'] == 'H2O'
    assert records['sodium chloride']['formula'] == 'NaCl'
    assert records['sodium chloride']['source_formula'] == 'Cl.Na'
    assert records['calcium carbonate']['formula'] == 'CaCO3'
    assert records['ammonia']['formula'] == 'NH3'
    assert 'not unseen-entity' in provenance['evaluation_scope']
    assert provenance['elements'] == 118 and provenance['compounds'] == 20
    assert 'CC BY-SA 3.0' in provenance['license'] and 'CC BY 4.0' in provenance['license']
    for record in records.values():
        assert record['chebi_id'].startswith('CHEBI:') and record['source'].startswith('http://purl.obolibrary.org/')
