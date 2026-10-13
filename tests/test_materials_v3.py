from pathlib import Path
import json

from fastapi.testclient import TestClient
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback

ROOT = Path(__file__).resolve().parents[1]


def policy(tmp_path):
    return MaterialsPolicy(ROOT / 'domains/materials/verified_records.json', tmp_path / 'feedback.db')


def test_symbols_formulas_and_ambiguity(tmp_path):
    p = policy(tmp_path)
    assert p.answer('Cu') == p.answer('Copper')
    assert p.answer('Fe') == p.answer('Iron')
    assert 'hydrogen bonds' in p.answer('H\u2082O')
    assert 'Cobalt' in p.answer('Co')
    assert 'Carbon monoxide' in p.answer('CO')
    assert p.answer('Cu99Zn1') == fallback('Cu99Zn1')
    assert p.answer('UNS S30403') == fallback('UNS S30403')


def test_typos_are_suggested_not_silently_admitted(tmp_path):
    text = policy(tmp_path).answer('Coppper')
    assert text.startswith(fallback('Coppper'))
    assert 'Possible spelling: copper' in text
    assert '390' not in text


def test_comparison_preserves_conditions_and_unknowns(tmp_path):
    p = policy(tmp_path)
    text = p.answer('Compare Cu and Aluminium')
    assert '390' in text and 'near 0' in text
    assert 'Knowledge' in text and 'Unit / scale' in text
    assert 'not a prediction of a mixture' in text
    assert 'N/A' not in text
    assert p.answer('Copper vs Steel ZZ999') == fallback('Steel ZZ999')
    text = p.answer('304 stainless steel vs 441 stainless steel')
    assert 'cold-rolled' in text and 'MPa' in text
    assert 'up to four' in p.answer('Iron,Copper,Glass,Water,Gold')


def test_direct_mechanism_and_experiment_requests(tmp_path):
    p = policy(tmp_path)
    text = p.answer('Why does chromium improve corrosion resistance in steel?')
    assert 'passive surface film' in text and 'service life' in text
    assert '###' not in text and 'http' not in text
    text = p.answer('plan experiments for glass')
    assert 'post-breakage' in text and 'Proposed plan only' in text
    assert p.answer('plan experiments for unobtainium') == fallback('unobtainium')


def test_followups_use_only_explicit_registered_identity(tmp_path):
    p = policy(tmp_path)
    assert 'Please name' in p.answer('improve it')
    assert 'Please name' in p.answer('improve it', previous_material='Copper has strength 999999 MPa')
    assert '**Objective:** safety' in p.answer('improve it for safety', previous_material='Glass')
    assert 'Cu-Cr-Zr' in p.answer('improve it', previous_material='Cu')


def test_unreviewed_mechanism_cannot_answer(tmp_path):
    (tmp_path / 'verified_records.json').write_text('{"materials": []}')
    data = json.loads((ROOT / 'domains/materials/reviewed_questions.json').read_text())
    data['questions'][0]['reviewed'] = False
    (tmp_path / 'reviewed_questions.json').write_text(json.dumps(data))
    p = MaterialsPolicy(tmp_path / 'verified_records.json', tmp_path / 'feedback.db')
    assert 'passive surface film' not in p.answer('Why does chromium improve corrosion resistance in steel?')


def test_evidence_and_experiments_only_for_supported_strategies(tmp_path):
    p = policy(tmp_path)
    text = p.answer('Copper')
    assert '## 12. Final Engineering Recommendation' in text
    assert '## 11. Real-World Applications and Laboratory Validation' in text
    assert 'Copper Development Association' in text
    assert 'https://' not in text
    assert '## 10.' not in p.answer('Water')


def test_stream_and_generate_followup_integration(tmp_path, monkeypatch):
    import ownllm.api.service as service
    from ownllm.api.limits import DistributedRateLimiter
    monkeypatch.setattr(service, '_materials_policy', policy(tmp_path))
    monkeypatch.setattr(service, '_limiter', DistributedRateLimiter(limit=30, window=60))
    client = TestClient(service.app)
    for prompt, previous in [('Cu', None), ('Compare Cu and Aluminium', None), ('improve it for safety', 'Glass')]:
        body = {'prompt': prompt, 'domain': 'materials', 'previous_material': previous}
        result = client.post('/generate', json=body)
        assert result.status_code == 200
        assert client.post('/stream', json=body).text == result.json()['text']


def test_latest_twelve_section_template_and_no_example_number_import(tmp_path):
    from ownllm.knowledge.materials_report import HEADINGS
    p = policy(tmp_path)
    for name in ('Iron', 'Glass', 'Copper', 'Titanium'):
        text = p.answer(name)
        assert text.startswith('CAMPUSMIND MATERIALS INTELLIGENCE\n\n# ' + name)
        for i, heading in enumerate(HEADINGS, 1):
            assert f'## {i:02d}. {heading}' in text
        assert '[Value]' not in text and '[Name]' not in text and 'N/A' not in text
        assert 'No unique best candidate' in text
        assert '| Knowledge |' in text
    assert '**Chemical Symbol:** Fe' in p.answer('Iron')
    assert '1538' not in p.answer('Iron')
    assert 'fracture toughness' not in p.answer('Titanium').lower()
    assert '## 01.' not in p.answer('thermal conductivity of copper')


def test_section_schema_rejects_empty_duplicate_and_out_of_order():
    import pytest
    from ownllm.knowledge.materials_report import MaterialReport, ReportSection
    with pytest.raises(ValueError):
        ReportSection(1, '')
    with pytest.raises(ValueError):
        ReportSection(13, 'unsupported section')
    with pytest.raises(ValueError):
        MaterialReport('Iron', (ReportSection(2, 'two'), ReportSection(1, 'one'))).render()
