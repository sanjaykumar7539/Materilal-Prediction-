"""Audit reviewed chat output and conservatively gate a local candidate checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ownllm.knowledge.materials_policy import MaterialsPolicy, fallback


def plain(text):
    return re.sub(r'\\([\\`*_{}\[\]<>#|])',r'\1',text).strip()


def audit_reviewed_chat(kb, feedback):
    policy = MaterialsPolicy(kb,feedback)
    raw = json.loads(kb.read_text(encoding='utf-8'))['materials']
    passed, properties, excluded, failures = 0,0,0,[]
    for record in raw:
        if not record['verified']:
            continue
        text = policy.profile(record['name'],material=record['name'])
        expected = [p for p in record['properties'] if p['verified'] and p['applicable']]
        blocks = re.findall(r'^\* \*\*([^\n]+):\*\*\n(.*?)(?=\n\n\* \*\*|\Z)',text,re.M|re.S)
        actual = {plain(name):body for name,body in blocks}
        valid = text.count('### 1. General Overview')==1 and text.count('### 2. Engineering Profile')==1
        valid &= len(blocks)==len(expected) and set(actual)=={p['name'] for p in expected}
        for prop in expected:
            body = actual.get(prop['name'],'')
            field = lambda name: re.search(r'\*\*'+re.escape(name)+r':\*\* ([^\n]+)',body)
            knowledge,value,unit = [field(name) for name in ['Knowledge','Typical Value / Range','Unit / Scale']]
            valid &= bool(knowledge and plain(knowledge[1])==prop['description'])
            valid &= bool(unit and plain(unit[1])==prop['unit'])
            expected_value = f"{prop['value']} (Conditions: {prop['conditions']})"
            valid &= bool(value and plain(value[1])==expected_value)
            properties += 1
        excluded += sum(not p['verified'] for p in record['properties'])
        passed += bool(valid)
        if not valid:
            failures.append(record['name'])
    unknowns = ['Unlisted test material','Unknown alloy grade 999','Hypothetical chemical XQ']
    fallback_passed = sum(policy.profile(name,material=name)==fallback(name) for name in unknowns)
    # Isolate an in-process HTTP client; do not alter the running server or production limiter.
    from fastapi.testclient import TestClient
    import ownllm.api.service as service
    from ownllm.api.limits import DistributedRateLimiter
    class NoModel:
        async def generate_text(self,*args,**kwargs):
            raise AssertionError('Reviewed material answer must not invoke the model')
    api_passed = 0
    with patch.object(service,'_materials_policy',policy), patch.object(service,'_runtime',NoModel()), \
         patch.object(service,'_limiter',DistributedRateLimiter(limit=30,window=60)):
        client = TestClient(service.app)
        for name in ['Hydrogen (element)','Water (compound)','Unknown alloy grade 999']:
            expected = policy.profile(name,material=name)
            for endpoint in ['/generate','/stream']:
                response = client.post(endpoint,json={'prompt':'Report sourced properties','material':name,
                                                       'domain':'materials','apply_template':False})
                actual = response.json().get('text') if endpoint=='/generate' and response.status_code==200 else response.text
                api_passed += response.status_code==200 and actual==expected
    return {'records_checked':passed+len(failures),'records_correct':passed,
            'source_checked_fields_checked':properties,'unreviewed_fields_excluded':excluded,
            'unknown_fallback_correct':fallback_passed,'unknown_fallback_cases':len(unknowns),
            'http_cases_correct':api_passed,'http_cases':6,'failures':failures,
            'passed':not failures and fallback_passed==len(unknowns) and api_passed==6,
            'scope':'Exact reproduction of the local reviewed collection, not a new independent scientific review.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--baseline',default='runs/chemistry_properties_v1')
    parser.add_argument('--knowledge',default='domains/materials/verified_records.json')
    args = parser.parse_args()
    run,baseline,kb = Path(args.run),Path(args.baseline),Path(args.knowledge)
    grade = json.loads((run/'property_evaluation.json').read_text())
    original = json.loads((baseline/'property_evaluation.json').read_text())
    with tempfile.TemporaryDirectory(dir=ROOT/'runs') as directory:
        chat = audit_reviewed_chat(kb,Path(directory)/'feedback.sqlite3')
    eligible = grade['questions']>=64 and grade['complete_source_profile_correct_accuracy']>=.95 and chat['passed']
    report = {'baseline':str(baseline),'candidate':str(run),
              'baseline_value_unit_correct':original['value_and_unit_correct'],
              'candidate_value_unit_correct':grade['value_and_unit_correct'],
              'baseline_complete_profiles':original['complete_source_profile_correct'],
              'candidate_complete_profiles':grade['complete_source_profile_correct'],
              'questions':grade['questions'],'reviewed_chat':chat,
              'native_checkpoint_eligible_for_source_grounded_trial':eligible,
              'threshold':'At least 64 cases and 95% complete profiles, plus all reviewed-chat checks.',
              'activation_performed':False,
              'knowledge_sha256':hashlib.sha256(kb.read_bytes()).hexdigest(),
              'predictions_sha256':hashlib.sha256((run/'predictions.json').read_bytes()).hexdigest(),
              'limitations':['The source-grounded test supplies values; this is not closed-book recall.',
                             'This is a repeated existing benchmark, not a fresh blind generalization test.',
                             'Passing this gate would permit a supervised trial, not engineering certification.',
                             'Verified chat accuracy is measured independently of native generated answers.']}
    for folder in [run,run/'export']:
        (folder/'readiness.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
