from pathlib import Path

from fastapi.testclient import TestClient

from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback

ROOT = Path(__file__).resolve().parents[1]


def test_plain_material_and_specific_property_questions(tmp_path):
    policy = MaterialsPolicy(ROOT/'domains/materials/verified_records.json',tmp_path/'feedback.sqlite3')
    expected = policy.profile('compound water')
    assert policy.profile('water') == expected
    assert policy.profile('Tell me about water?') == expected
    formula = policy.profile('What is the chemical formula of water?')
    assert '**Chemical Formula:**' in formula and 'H2O' in formula
    assert '**Calculated Molecular Mass:**' not in formula
    mass = policy.profile('molecular mass of water')
    assert '18.015' in mass and '**Unit / Scale:** Da' in mass
    assert policy.profile('boiling point of hydrogen') == fallback('Hydrogen (element)')
    assert policy.profile('yield strength of water') == fallback('Water (compound)')
    assert policy.profile('Water alloy grade 999') == fallback('Water alloy grade 999')
    assert policy.profile('chemical formula of Unknown chemical') == fallback('chemical formula of Unknown chemical')


def test_natural_questions_use_verified_http_paths(tmp_path,monkeypatch):
    import ownllm.api.service as service
    from ownllm.api.limits import DistributedRateLimiter
    monkeypatch.setattr(service,'_materials_policy',MaterialsPolicy(
        ROOT/'domains/materials/verified_records.json',tmp_path/'feedback.sqlite3'))
    monkeypatch.setattr(service,'_runtime',None)
    monkeypatch.setattr(service,'_limiter',DistributedRateLimiter(limit=30,window=60))
    client = TestClient(service.app)
    for endpoint in ['/generate','/stream']:
        response = client.post(endpoint,json={'prompt':'chemical formula of water','domain':'materials'})
        assert response.status_code==200
        text = response.json()['text'] if endpoint=='/generate' else response.text
        assert 'H2O' in text and '**Chemical Formula:**' in text


def test_water_explanation_is_content_not_a_source_directory(tmp_path):
    policy = MaterialsPolicy(ROOT/'domains/materials/verified_records.json',tmp_path/'feedback.sqlite3')
    text = policy.profile('What is water?')
    assert 'hydrogen bonds' in text and 'biological processes' in text
    assert '**Chemical Formula:**' in text and 'H2O' in text
    assert 'Source:' not in text and 'Related reading:' not in text
    assert 'http://' not in text and 'https://' not in text
    assert '### 1. General Overview' in text and '### 2. Engineering Profile' in text


def test_material_question_variants_and_generic_properties(tmp_path):
    policy = MaterialsPolicy(ROOT/'domains/materials/verified_records.json',tmp_path/'feedback.sqlite3')
    records = policy.records()
    policy.records = lambda: records
    for question, property_name in [
        ('What are the properties of copper?', 'Thermal Conductivity'),
        ('Tell me about stainless steel 304', 'Yield Strength'),
        ('What is the thermal conductivity of copper?', 'Thermal Conductivity'),
        ('Glass', 'Thermal Conductivity'),
        ('Wood', 'Specific Heat Capacity'),
        ('Explain iron', 'Atomic Number'),
        ('Properties of aluminium', "Young's Modulus"),
        ('Young modulus of aluminium', "Young's Modulus"),
    ]:
        result = policy.profile(question)
        assert '### 2. Engineering Profile' in result, question
        assert '**' + property_name + ':**' in result, question
    for question in ['Copper alloy ZZ999','Water alloy grade 999','UNS S30403','EN 1.4372']:
        assert policy.profile(question) == fallback(question)
    assert policy.profile('thermal conductivity of element copper') == fallback('Copper (element)')
