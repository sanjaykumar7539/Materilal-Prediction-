import importlib.util
import json
from pathlib import Path
import zipfile


def test_conversion_keeps_held_out_answers_and_removes_question_leakage(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('split_training',scripts/'train_split_dataset.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path/'input.zip'
    rows = {
        'train':[{'question':' A shared QUESTION ','answer':'Training answer','verified':True},
                 {'question':'Train only','answer':'Training fact','verified':True},
                 {'question':'train   only','answer':'Different duplicate answer','verified':True}],
        'validation':[{'question':'Validation only','answer':'Validation fact','verified':True}],
        'test':[{'question':'a shared question','answer':'Held-out answer','verified':True}],
    }
    with zipfile.ZipFile(archive,'w') as handle:
        for split in rows:
            handle.writestr(f'split_dataset/reviewed_qa/{split}.jsonl',
                            '\n'.join(json.dumps(row) for row in rows[split]))
            handle.writestr(f'split_dataset/rag_knowledge/{split}.jsonl',
                            json.dumps({'id':split,'concept':split+' concept','definition':split+' definition'}))
    out = tmp_path/'run'
    out.mkdir()
    splits,report = module.prepare(archive,out)
    assert len(report['removed_questions']) == 2
    assert all(item['different_answer'] for item in report['removed_questions'])
    assert all(count == 0 for count in report['audit']['duplicate_prompts'].values())
    assert splits['test'][0]['output'] == 'Held-out answer'
    assert splits['train'][0]['output'] == 'Training fact'
    assert splits['train'][0]['supplied_verified'] is True
    assert 'verified' not in splits['train'][0]
    assert all(row['source_id'] not in ('test','validation') for row in splits['train'])
