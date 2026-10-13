import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient
from ownllm.api import service
from ownllm.api.limits import DistributedRateLimiter
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback

ROOT = Path(__file__).resolve().parents[1]
KB = ROOT / 'domains/materials/verified_records.json'


def test_complete_reviewed_element_identities_and_compounds():
    raw = json.loads(KB.read_text(encoding='utf-8'))
    elements = [r for r in raw['materials'] if r['name'].endswith('(element)')]
    compounds = [r for r in raw['materials'] if r['name'].endswith('(compound)')]
    assert len(elements) == 118 and len(compounds) == 200
    numbers = {int(r['properties'][0]['value']) for r in elements}
    assert numbers == set(range(1,119))
    for element in elements:
        reviewed = [p for p in element['properties'] if p['verified']]
        assert len(reviewed) == 2
        assert all(p['applicable'] for p in reviewed)
        assert all('pubchem.ncbi.nlm.nih.gov' in p['source'] for p in reviewed)
    audit = json.loads((KB.parent/'CHEMISTRY_SOURCE_REVIEW.json').read_text(encoding='utf-8'))
    assert audit['knowledge_base_sha256'] == hashlib.sha256(KB.read_bytes()).hexdigest()
    assert audit['total_records'] == 349 and audit['total_properties'] == 1837


def test_identity_profiles_work_offline_and_do_not_invent_engineering_values(tmp_path,monkeypatch):
    policy = MaterialsPolicy(KB,tmp_path/'feedback.sqlite3')
    monkeypatch.setattr(service,'_materials_policy',policy)
    monkeypatch.setattr(service,'_runtime',None)
    monkeypatch.setattr(service,'_limiter',DistributedRateLimiter(limit=30,window=60))
    client = TestClient(service.app)
    for query, value, source in [('element hydrogen','1','pubchem'),
        ('element aluminium','13','pubchem'),('element oganesson','118','pubchem'),
        ('compound sodium chloride','NaCl','ChEBI'),('compound water','H2O','ChEBI')]:
        body = {'domain':'materials','prompt':query,'material':query}
        response = client.post('/generate',json=body)
        assert response.status_code == 200
        answer = response.json()['text']
        assert value in answer and 'Source:' not in answer
        assert '## 01. Material Overview' in answer and '## 02. Complete Material Properties' in answer
        assert 'N/A' not in answer
        assert client.post('/stream',json=body).text == answer
    assert policy.profile('element copper',parameter='Yield Strength') == fallback('Copper (element)')
    assert policy.profile('compound water',parameter='Melting Point') == fallback('Water (compound)')
    assert 'Thermal Conductivity' in policy.profile('generic copper')
    assert policy.profile('6061-T6 aluminium',material='6061-T6 aluminium') == fallback('6061-T6 aluminium')
