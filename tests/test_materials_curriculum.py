"""Audit generated answers independently of the curriculum builder."""
import json
from pathlib import Path
import re
import subprocess
import sys

from ownllm.training.books import audit_sft


def test_curriculum_has_varied_valid_decisions_and_correct_calculations(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path/'curriculum'
    result = subprocess.run([sys.executable,str(root/'scripts/prepare_materials_curriculum.py'),
                             '--out',str(out),'--licensed-only'],cwd=root,capture_output=True,text=True)
    assert result.returncode == 0,result.stderr
    splits = {split:[json.loads(line) for line in
                     (out/'dataset'/f'sft_{split}.jsonl').read_text(encoding='utf-8').splitlines()]
              for split in ('train','validation','test')}
    assert all(value == 0 for value in audit_sft(splits)['duplicate_prompts'].values())
    rows = [row for group in splits.values() for row in group]
    outcomes = set()
    profile_ids = set()
    for row in rows:
        concept = row['concept']
        if concept == 'constraint_selection':
            threshold = int(re.search(r'≥(\d+) MPa',row['instruction'])[1])
            candidates = [(name,int(yield_value),int(density)) for name,yield_value,density in
                          re.findall(r'([ABC]): yield (\d+) MPa, density (\d+) kg/m³',row['input'])]
            feasible = [candidate for candidate in candidates if candidate[1] >= threshold]
            expected = min(feasible,key=lambda candidate:candidate[2])[0] if feasible else 'none'
            outcomes.add(expected)
            assert (f'Choose {expected}.' in row['output']) if feasible else ('No feasible candidate.' in row['output'])
        if concept.startswith('formula:'):
            numbers = [float(n) for n in re.findall(r'\d+(?:\.\d+)?',row['instruction'])]
            answer = float(re.search(r'Answer: ([-+\d.eE]+)',row['output'])[1])
            family = int(concept.split(':')[1])
            if family == 0:
                expected = numbers[0]/numbers[1]
            elif family == 1:
                expected = numbers[0]/numbers[1]/1000
            elif family == 2:
                expected = numbers[0]*numbers[1]*numbers[2]
            else:
                values = re.search(r'coefficient (\d+)×10\^-6 K\^-1, length (\d+) m, and temperature rise (\d+) K',row['instruction'])
                coefficient,length,change = map(float,values.groups())
                expected = coefficient*length*change/1000
            assert abs(answer-expected) < 1e-8
        if concept.startswith('profile_format:'):
            profile_ids.add(int(concept.split(':')[1]))
    assert outcomes == {'A','B','C','none'}
    assert profile_ids == set(range(1,67))
