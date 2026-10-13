import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from ownllm.api import service
from ownllm.api.limits import DistributedRateLimiter
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / 'domains/materials/verified_records.json'


def test_reference_profiles_preserve_conditions_and_units(tmp_path, monkeypatch):
    policy = MaterialsPolicy(RECORDS, tmp_path / 'feedback.sqlite3')
    monkeypatch.setattr(service, '_materials_policy', policy)
    # Keep this request-heavy integration check independent of other tests'
    # shared per-minute request history, while retaining real rate limiting.
    monkeypatch.setattr(service, '_limiter', DistributedRateLimiter(limit=30, window=60))
    client = TestClient(service.app)
    for material, prop, value, unit, condition in [
        ('generic aluminium', "Young's Modulus", '70', 'GPa', 'temperature not specified'),
        ('generic copper', 'Thermal Conductivity', '390', 'W/(m·K)', 'near 0 °C'),
        ('generic water', 'Specific Heat Capacity', '4186', 'J/(kg·K)', '15.0 °C'),
        ('generic ice', 'Specific Heat Capacity', '2090', 'J/(kg·K)', '−50 °C to 0 °C'),
        ('generic wood', 'Thermal Conductivity', '0.08–0.16', 'W/(m·K)', 'near 0 °C'),
        ('generic glycerol', 'Specific Heat Capacity', '2410', 'J/(kg·K)', '25 °C'),
        ('generic polystyrene foam', 'Thermal Conductivity', '0.010', 'W/(m·K)', 'not solid polystyrene'),
    ]:
        request = {'domain': 'materials', 'prompt': material, 'material': material, 'parameter': prop}
        response = client.post('/generate', json=request)
        assert response.status_code == 200
        answer = response.json()['text']
        assert value in answer and unit in answer and condition in answer
        assert '### 1. General Overview' in answer and '### 2. Engineering Profile' in answer
        assert answer.count('**Unit / Scale:**') == 1
        assert 'Source:' not in answer and 'Download for free at' not in answer
        assert client.post('/stream', json=request).text == answer


def test_generic_data_cannot_substitute_for_unknown_grades_or_properties(tmp_path):
    policy = MaterialsPolicy(RECORDS, tmp_path / 'feedback.sqlite3')
    for request in ['6061-T6 aluminium', 'C11000 copper', 'solid polystyrene', '304 copper alloy']:
        # Explicit material selection is how the API disambiguates a name.
        assert policy.profile(request, material=request) == fallback(request)
    assert policy.profile('generic copper', parameter='Yield Strength') == fallback('Copper (generic reference)')
    assert policy.profile('generic water', parameter="Young's Modulus") == fallback('Water (generic reference)')
    assert 'Thermal Conductivity' not in policy.profile('generic iron')
    assert 'temperature not specified' in policy.profile('generic nickel')
    assert 'Yield Strength' in policy.profile('304', material='304')


def test_collection_and_reference_provenance_are_complete(tmp_path):
    raw = json.loads(RECORDS.read_text(encoding='utf-8'))
    audit = json.loads((RECORDS.parent / 'REFERENCE_SOURCE_REVIEW.json').read_text(encoding='utf-8'))
    assert audit['knowledge_base_sha256'] == hashlib.sha256(RECORDS.read_bytes()).hexdigest()
    assert audit['total_records'] == len(raw['materials']) == 349
    assert audit['total_properties'] == sum(len(r['properties']) for r in raw['materials']) == 1837
    policy = MaterialsPolicy(RECORDS, tmp_path / 'feedback.sqlite3')
    assert len(policy.records()) == len(raw['materials'])
    references = [r for r in raw['materials'] if r['name'].endswith('(generic reference)')]
    assert len(references) == 17
    for record in references:
        assert record['license'] == 'CC BY 4.0'
        assert record['review_date'] == '2026-10-08'
        answer = policy.profile(record['name'], material=record['name'])
        assert answer.count('**Unit / Scale:**') == len(record['properties'])
        assert 'N/A' not in answer
        for prop in record['properties']:
            assert prop['verified'] is True and prop['applicable'] is True
            assert 'PDF page' in prop['source'] and 'Table ' in prop['source']
            assert 'not a specification limit' in prop['conditions']
