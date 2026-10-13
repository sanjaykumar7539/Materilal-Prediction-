import importlib.util
import json
from pathlib import Path

from ownllm.knowledge.materials_policy import MaterialsPolicy


def test_reviewed_chat_audit_detects_a_corrupted_numeric_answer(tmp_path,monkeypatch):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('readiness',root/'scripts/evaluate_factual_readiness.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kb = tmp_path/'records.json'
    prop = {'name':'Reviewed property','description':'Fixture definition','value':'10','unit':'MPa',
            'source':'Fixture source','conditions':'Fixture conditions','verified':True,'applicable':True}
    kb.write_text(json.dumps({'materials':[{'name':'Fixture','overview':'Fixture overview',
                  'overview_source':'Fixture source','verified':True,'properties':[prop,
                  {**prop,'name':'Unreviewed property','value':'999','verified':False}]}]}))
    result = module.audit_reviewed_chat(kb,tmp_path/'feedback.sqlite3')
    assert result['passed'] and result['records_correct']==1
    assert result['source_checked_fields_checked']==1 and result['unreviewed_fields_excluded']==1
    original = MaterialsPolicy.profile
    def corrupt(self,*args,**kwargs):
        return original(self,*args,**kwargs).replace('Range:** 10 (','Range:** 20 (')
    monkeypatch.setattr(MaterialsPolicy,'profile',corrupt)
    result = module.audit_reviewed_chat(kb,tmp_path/'feedback.sqlite3')
    assert not result['passed'] and result['failures']==['Fixture']
