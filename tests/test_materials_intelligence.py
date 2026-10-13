import json
from pathlib import Path

from fastapi.testclient import TestClient
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback
from ownllm.knowledge.materials_intelligence import HEADINGS

ROOT = Path(__file__).resolve().parents[1]


def engine(tmp_path):
    return MaterialsPolicy(ROOT/'domains/materials/verified_records.json',tmp_path/'feedback.sqlite3')


def test_six_material_reports_have_real_baselines_and_qualified_candidates(tmp_path):
    policy = engine(tmp_path)
    expected = {'Iron':'zinc', 'Copper':'Cu-Cr-Zr', 'Glass':'Laminated',
                'Aluminium':'6061', 'Steel':'quenching', 'Titanium':'Ti-6Al-4V'}
    for name, candidate in expected.items():
        text = policy.answer(name)
        for i,heading in enumerate(HEADINGS,1):
            assert f'## {i:02d}. {heading}' in text
        assert candidate.lower() in text.lower()
        assert 'No validated engineering-property prediction model is installed or executed' in text
        assert 'https://' not in text and 'Source:' not in text
    assert '390' in policy.answer('Copper') and 'near 0' in policy.answer('Copper')
    assert 'Representative grade: 304 stainless steel' in policy.answer('Steel')
    assert 'Representative grade: 441 stainless steel' in policy.answer('Steel')
    assert '540' in policy.answer('Steel') and '20 ' in policy.answer('Steel')
    assert 'Yield Strength' not in policy.answer('Titanium')


def test_focused_questions_unknowns_and_objectives(tmp_path):
    policy = engine(tmp_path)
    question = 'thermal conductivity of copper'
    assert policy.answer(question) == policy.profile(question)
    assert policy.answer('Water alloy ZZ999') == fallback('Water alloy ZZ999')
    assert policy.answer('improve Water alloy ZZ999') == fallback('Water alloy ZZ999')
    assert policy.answer('UNS S30403') == fallback('UNS S30403')
    text = policy.answer('Improve glass for safety')
    ranking = text.split('## 09. Engineering Optimization')[1]
    assert ranking.index('Laminated glass') < ranking.index('Thermally toughened')
    assert '**Objective:** safety' in text
    assert 'no matching reviewed candidate evidence' in policy.answer('Improve copper for biocompatibility')
    assert policy.answer('Copper',parameter='Thermal Conductivity') == policy.profile('Copper',parameter='Thermal Conductivity')


def test_known_material_without_strategies_has_honest_limits(tmp_path):
    text = engine(tmp_path).answer('Water')
    assert 'hydrogen bonds' in text
    assert 'No reviewed candidate combination' in text
    assert text.count('No reviewed candidate combination') == 1
    assert '## 04.' not in text
    assert 'H2O' in text


def test_unreviewed_catalog_cannot_supply_recommendations(tmp_path):
    source = ROOT/'domains/materials'
    (tmp_path/'verified_records.json').write_bytes((source/'verified_records.json').read_bytes())
    data=json.loads((source/'intelligence_catalog.json').read_text())
    data['materials']['Copper']['reviewed']=False
    (tmp_path/'intelligence_catalog.json').write_text(json.dumps(data))
    text=MaterialsPolicy(tmp_path/'verified_records.json',tmp_path/'feedback.sqlite3').answer('Copper')
    assert 'Cu-Cr-Zr' not in text and 'No reviewed candidate combination' in text


def test_generate_and_stream_use_identical_analysis_without_native_guessing(tmp_path,monkeypatch):
    import ownllm.api.service as service
    from ownllm.api.limits import DistributedRateLimiter
    class NoModel:
        async def generate_text(self,*args,**kwargs):
            raise AssertionError('A native guess must not replace reviewed intelligence')
    policy=engine(tmp_path)
    monkeypatch.setattr(service,'_materials_policy',policy)
    monkeypatch.setattr(service,'_runtime',NoModel())
    monkeypatch.setattr(service,'_limiter',DistributedRateLimiter(limit=30,window=60))
    client=TestClient(service.app)
    body={'prompt':'Copper','domain':'materials'}
    response=client.post('/generate',json=body)
    assert response.status_code==200
    assert response.json()['text']==policy.answer('Copper')
    assert client.post('/stream',json=body).text==response.json()['text']


def test_integrated_rules_preserve_taxonomy_and_filtering():
    import re
    prompt=(ROOT/'domains/materials/SYSTEM_PROMPT.md').read_text(encoding='utf-8')
    taxonomy=prompt.split('## Parameter Knowledge Base',1)[1]
    names=re.findall(r'^\d+\. ([^:\n]+):',taxonomy,re.M)
    assert len(names)==66 and len(set(names))==66
    assert names[0]=='Tensile Strength' and names[-1]=='Nanoscale Behavior'
    assert 'NO FILLERS' in prompt and 'Knowledge' in prompt and 'Unit / Scale' in prompt
    assert '### 8. Confidence and validation' in prompt
    assert 'unverified local feedback' in prompt


def test_representative_grades_keep_conditions_and_never_replace_unknown_grade(tmp_path):
    policy=engine(tmp_path)
    text=policy.answer('Steel')
    assert 'not universal family properties' in text
    assert 'cold-rolled' in text and '| MPa |' in text
    assert 'N/A' not in text
    assert text.count('No validated engineering-property prediction model')==1
    assert 'Evidence type' in text
    assert policy.answer('Steel ZZ999')==fallback('Steel ZZ999')
    assert policy.answer('yield strength of Steel ZZ999')==fallback('yield strength of Steel ZZ999')
