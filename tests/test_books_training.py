import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ownllm.tokenizer_train import train_bpe
from ownllm.training.books import SPECIAL_TOKENS, audit_sft, cached_documents, encode_sft, split_documents


def test_document_splits_are_reproducible_and_duplicates_stay_out():
    docs = [(f'book{i}',f'Distinct document {i}. ' * 30) for i in range(10)]
    docs += [('copy','DISTINCT   DOCUMENT 0. ' * 30)]
    splits, report = split_documents(docs)
    assert len(report['duplicates']) == 1
    assert [len(splits[name]) for name in ('train','validation','test')] == [8,1,1]
    assert splits == split_documents(reversed(docs))[0]
    hashes = [row['sha256'] for rows in splits.values() for row in rows]
    assert len(hashes) == len(set(hashes))


def test_duplicate_instruction_prompt_is_rejected():
    row = {'instruction':'Explain friction','output':'Friction resists motion.'}
    with pytest.raises(ValueError,match='overlap'):
        audit_sft({'train':[row],'test':[row]})


def test_extraction_cache_invalidates_changed_source(tmp_path):
    source = tmp_path/'books'
    source.mkdir()
    paper = source/'paper.txt'
    paper.write_text('First version')
    cache = tmp_path/'cache'
    assert list(cached_documents(source,cache))[0][1] == 'First version'
    paper.write_text('Updated version')
    assert list(cached_documents(source,cache))[0][1] == 'Updated version'


def test_tokenizer_roundtrip_and_assistant_mask(tmp_path):
    source = tmp_path/'source.txt'
    source.write_text('Explain friction. Friction resists motion. Café 🌍' * 20,encoding='utf-8')
    tokenizer = train_bpe([source],tmp_path/'tokenizer',512,special_tokens=SPECIAL_TOKENS)
    assert tokenizer.decode(tokenizer.encode('Café 🌍').ids) == 'Café 🌍'
    ids, labels = encode_sft({'instruction':'Explain friction','output':'Friction resists motion.'},tokenizer,64)
    assert len(ids) == len(labels) == 64
    assert labels[0].item() == -100
    assert (labels != -100).any()
    supervised = labels[labels != -100].tolist()
    assert tokenizer.decode(supervised) == 'Friction resists motion.'


def test_training_pipeline_saves_loadable_weights_and_test_report(tmp_path):
    source = tmp_path/'input'
    books, dataset = source/'books', source/'dataset'
    books.mkdir(parents=True)
    dataset.mkdir()
    for index in range(5):
        (books/f'{index}.txt').write_text(f'Document {index} describes a different material property. '*100)
    for split in ('train','validation','test'):
        (dataset/f'sft_{split}.jsonl').write_text(json.dumps({'instruction':f'Explain {split} material','output':'A material property.'})+'\n')
    config = tmp_path/'config.json'
    config.write_text(json.dumps({'dim':32,'n_layers':1,'n_heads':4,'n_kv_heads':2,'ffn_hidden':64,'max_seq_len':64}))
    out = tmp_path/'run'
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,str(root/'scripts/train_books.py'),'--input',str(source),'--out',str(out),'--config',str(config),'--steps','1','--sft-epochs','1','--vocab','512','--seq-len','32','--batch','1','--threads','1','--eval-batches','1','--max-new-tokens','4'],cwd=root,text=True,capture_output=True)
    assert result.returncode == 0,result.stderr
    state = torch.load(out/'final.pt',weights_only=False)
    assert state['config']['dim'] == 32
    assert state['extra']['stage'] == 'complete'
    report = json.loads((out/'evaluation.json').read_text())
    assert report['book_test']['evaluated_tokens'] > 0
    assert report['sft_test_generation']['questions'] == 1
    assert (out/'export/SHA256SUMS').exists()
