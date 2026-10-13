from pathlib import Path
from fastapi.testclient import TestClient
from ownllm.api import service
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback

def test_reviewed_stainless_answers_are_grade_specific_and_conditioned(tmp_path,monkeypatch):
    path = Path(__file__).resolve().parents[1]/'domains/materials/verified_records.json'
    engine = MaterialsPolicy(path,tmp_path/'feedback.sqlite3')
    monkeypatch.setattr(service,'_materials_policy',engine)
    from ownllm.api.limits import DistributedRateLimiter
    monkeypatch.setattr(service, '_limiter', DistributedRateLimiter(limit=30,window=60))
    client = TestClient(service.app)
    for grade,value in [('304','540–750'),('304L','520–700'),('304LN','550–750'),('305','500–650'),('321','520–720'),('347','520–720'),('301LN','650–850'),('301','600–950')]:
        response = client.post('/generate',json={'domain':'materials','prompt':grade})
        assert response.status_code == 200
        text = response.json()['text']
        assert value in text and '20 °C' in text and 'cold-rolled' in text
        assert '## 01. Material Overview' in text and '## 02. Complete Material Properties' in text
        assert 'Source:' not in text and 'https://' not in text
        assert sum(line.startswith('| ') and '(Conditions:' in line for line in text.splitlines()) == 7
        assert client.post('/stream',json={'domain':'materials','prompt':grade}).text == text
    assert engine.profile('Unlisted example') == fallback('Unlisted example')
    assert engine.profile('304',parameter='Curie Temperature') == fallback('304 stainless steel')

def test_complete_grade_set_omits_missing_values_and_preserves_standards(tmp_path):
    path=Path(__file__).resolve().parents[1]/'domains/materials/verified_records.json'
    engine=MaterialsPolicy(path,tmp_path/'feedback.sqlite3')
    for grade in ('201','201LN'):
        text=engine.profile(grade,material=grade)
        assert text.count('**Unit / Scale:**') == 5
        assert '**Specific Heat Capacity:**' not in text
        assert '**Thermal Expansion Coefficient:**' not in text
    assert 'ASTM A240' in engine.profile('439M',material='439M')
    assert 'EN 10028-7' in engine.profile('4622',material='4622')
    assert '220' in engine.profile('441',material='441',parameter="Young's Modulus")
    assert '1.4306' in engine.profile('EN 1.4306',material='EN 1.4306')
    assert engine.profile('UNS S30403',material='UNS S30403') == fallback('UNS S30403')
