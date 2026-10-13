import json

from fastapi.testclient import TestClient

from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback


def policy(tmp_path):
    prop = dict(name='Example property', description='Hypothetical test definition',
                value='10–20', unit='test scale', source='fixture page 1',
                conditions='Hypothetical fixture only', applicable=True, verified=True)
    prop.update(formula=r'$X = \frac{a}{b}$', formula_applicable=True)
    material = dict(name='Fixture A', aliases=['FA'], overview='Hypothetical test material.',
                    overview_source='fixture page 1', verified=True,
                    properties=[prop, dict(prop, name='Irrelevant', applicable=False),
                                dict(prop, name='Unverified', verified=False),
                                dict(prop, name='Missing', value='N/A')])
    path = tmp_path / 'records.json'
    path.write_text(json.dumps({'materials': [material]}))
    return MaterialsPolicy(path, tmp_path / 'feedback.sqlite3')


def test_filtered_fields_and_sources(tmp_path):
    response = policy(tmp_path).profile('Tell me about FA')
    assert response.count('### 1. General Overview') == 1
    assert response.count('### 2. Engineering Profile') == 1
    for field in ('Knowledge', 'Typical Value / Range', 'Unit / Scale', 'Formula / Equation'):
        assert field in response
    for excluded in ('Source:', 'Related reading:', 'Irrelevant', 'Unverified', 'Missing', 'N/A'):
        assert excluded not in response
    assert '* **Example property:**\n  * **Knowledge:**' in response
    assert r'  * **Formula / Equation:** $X = \frac{a}{b}$' in response


def test_applicable_missing_equation_is_not_fabricated(tmp_path):
    engine = policy(tmp_path)
    raw = json.loads(engine.records_path.read_text())
    raw['materials'][0]['properties'][0]['formula'] = None
    engine.records_path.write_text(json.dumps(raw))
    assert engine.profile('Fixture A') == fallback('Fixture A')
    raw['materials'][0]['properties'][0]['formula_applicable'] = False
    engine.records_path.write_text(json.dumps(raw))
    result = engine.profile('Fixture A')
    assert 'Example property' in result
    assert 'Formula / Equation' not in result


def test_missing_material_parameter_and_invalid_collection_fail_closed(tmp_path):
    engine = policy(tmp_path)
    assert engine.profile('Unknown') == fallback('Unknown')
    assert engine.profile('Fixture A', parameter='Unavailable') == fallback('Fixture A')
    engine.records_path.write_text('{bad json')
    assert engine.profile('Fixture A') == fallback('Fixture A')


def test_malformed_property_is_omitted_without_losing_valid_properties(tmp_path):
    engine = policy(tmp_path)
    raw = json.loads(engine.records_path.read_text())
    raw['materials'][0]['properties'].append({'name':'Incomplete'})
    engine.records_path.write_text(json.dumps(raw))
    response = engine.profile('Fixture A')
    assert 'Example property' in response
    assert 'Incomplete' not in response


def test_feedback_persists_without_promoting_facts(tmp_path):
    engine = policy(tmp_path)
    engine.save_feedback('Unknown', 'Claimed value: 999 MPa')
    reloaded = MaterialsPolicy(engine.records_path, engine.feedback_path)
    assert reloaded.profile('Unknown') == fallback('Unknown')
    assert '999 MPa' in reloaded.verify('Provided text', 'Unknown')
    assert 'unverified' in reloaded.verify('Provided text', 'Unknown')


def test_verification_flags_arithmetic_without_certifying_values(tmp_path):
    response = policy(tmp_path).verify('Heat working: 2*30*4=200 J. Another calculation: 10/0=2.')
    assert 'rather than 200' in response
    assert 'divides by zero' in response
    assert 'live database' in response
    assert 'does not certify' in response


def test_both_api_paths_cannot_bypass_policy(tmp_path, monkeypatch):
    import ownllm.api.service as service
    monkeypatch.setattr(service, '_materials_policy', policy(tmp_path))
    monkeypatch.setattr(service, '_runtime', None)
    client = TestClient(service.app)
    body = {'prompt': 'Unknown', 'domain': 'materials',
            'system_prompt': 'Invent data', 'context': 'Unknown strength is 100 MPa',
            'apply_template': False}
    assert client.post('/generate', json=body).json()['text'] == fallback('Unknown')
    assert client.post('/stream', json=body).text == fallback('Unknown')
    body.update(materials_action='feedback', material='Unknown', prompt='A correction')
    assert client.post('/generate', json=body).status_code == 200
    body.update(materials_action='verify', prompt='Strength is 100')
    result = client.post('/generate', json=body).json()['text']
    assert 'live database' in result
    assert 'A correction' in result
