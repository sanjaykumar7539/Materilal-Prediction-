import importlib.util
import json
from pathlib import Path


def test_source_split_removes_shared_chunks_and_preserves_book_groups(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('corpus_training',scripts/'train_corpus_jsonl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for source in range(5):
        rows.extend([{'source':f'book-{source}','page':1,'chunk_id':f'{source}-a','text':f'Unique source {source} content.'},
                     {'source':f'book-{source}','page':2,'chunk_id':f'{source}-b','text':'Shared passage.'}])
    rows.append(dict(rows[0],page=3,chunk_id='extra'))
    path = tmp_path/'corpus.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows),encoding='utf-8')
    report = module.prepare_sources(path,tmp_path/'prepared')
    assert report['input_records'] == 11
    assert report['retained_chunks'] == 6
    assert report['removed_duplicate_or_empty_chunks'] == 5
    assert {name:len(items) for name,items in report['splits'].items()} == {'test':1,'validation':1,'train':3}
    sources = [{item['source'] for item in report['splits'][split]} for split in ('train','validation','test')]
    assert not (sources[0]&sources[1] or sources[0]&sources[2] or sources[1]&sources[2])
    documents = {name:' '.join(Path(item['path']).read_text() for item in items)
                 for name,items in report['splits'].items()}
    assert 'Shared passage.' in documents['test']
    assert 'Shared passage.' not in documents['train']
    assert 'Shared passage.' not in documents['validation']


def test_json_array_excludes_previously_seen_chunks_before_splitting(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('corpus_array_training',scripts/'train_corpus_jsonl.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'source':f'book-{i}','text':f'New passage {i}.'} for i in range(5)]
    rows.append({'source':'previous-book','text':'  ALREADY  seen. '})
    source = tmp_path/'new.json'
    source.write_text(json.dumps(rows),encoding='utf-8')
    prior = tmp_path/'old.jsonl'
    prior.write_text(json.dumps({'source':'old','text':'Already seen.'})+'\n',encoding='utf-8')
    report = module.prepare_sources(source,tmp_path/'prepared',exclude_seen_input=prior)
    assert report['input_records'] == 6
    assert report['excluded_prior_chunks'] == 1
    assert report['retained_chunks'] == 5
    assert all('already' not in Path(item['path']).read_text().lower()
               for split in report['splits'].values() for item in split)
