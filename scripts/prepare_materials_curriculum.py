"""Reproducible synthetic practice plus the original curated book instructions.

Synthetic profile records teach extraction/formatting, not real material constants.
Never import these hypothetical records into the verified materials collection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ownllm.training.books import audit_sft

MECHANICS = 'https://openstax.org/books/university-physics-volume-1/pages/12-3-stress-strain-and-elastic-modulus'
HEAT = 'https://openstax.org/books/university-physics-volume-2/pages/1-key-equations'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/materials_66_v2')
    parser.add_argument('--base', default='data/drive_books/dataset')
    parser.add_argument('--licensed-only', action='store_true', help='Exclude Drive SFT training records with unestablished licenses')
    args = parser.parse_args()
    out, base = Path(args.out), Path(args.base)
    if out.exists():
        raise SystemExit('Choose a new output directory; existing data is preserved.')
    dataset = out/'dataset'
    dataset.mkdir(parents=True)
    taxonomy = (ROOT/'domains/materials/SYSTEM_PROMPT.md').read_text(encoding='utf-8').split('## Parameter Knowledge Base',1)[1]
    names = re.findall(r'^\d+\. ([^:\n]+):', taxonomy, re.M)
    assert len(names) == 66
    splits = {name:[] for name in ('train','validation','test')}
    generated = []

    def add(identifier, instruction, context, answer, concept, source):
        generated.append(dict(id=identifier, instruction=instruction, input=context,
                              output=answer, module='Materials curriculum', concept=concept,
                              source_id=source))

    for index, name in enumerate(names, 1):
        for variant in range(8):
            material = f'Hypothetical sample P{index:02d}-{variant}'
            # Deliberately non-physical placeholders: this task tests copying supplied fields.
            description = f'Supplied description {index}-{variant}'
            value = f'Supplied range {variant+1} to {variant+2}'
            unit = f'Supplied scale {index}'
            context = (f'Formatting exercise only; no real engineering values. Material={material}. '
                       f'Overview=Hypothetical formatting sample. Property={name}; '
                       f'Knowledge={description}; Value={value}; Unit={unit}; '
                       'applicable=true. Another property is irrelevant; omit it.')
            answer = (f'### 1. General Overview\n\nHypothetical formatting sample.\n\n'
                      f'### 2. Engineering Profile\n\n* **{name}:**\n'
                      f'  * **Knowledge:** {description}\n'
                      f'  * **Typical Value / Range:** {value}\n'
                      f'  * **Unit / Scale:** {unit}')
            add(f'profile-{index}-{variant}', 'Format only the supplied applicable property.',
                context, answer, f'profile_format:{index}', f'hypothetical-profile-{index}-{variant}')

    rng = random.Random(20261007)
    numeric = []
    for index in range(400):
        a, b, c = rng.randint(2,90), rng.randint(2,30), rng.randint(2,20)
        family = index % 4
        if family == 0:
            force = a*b
            question = f'Uniform axial load {force} N acts on original area {b} mm². Calculate engineering stress.'
            formula, answer, result, source = 'σ=F/A', f'{a} MPa', a, MECHANICS
            working = f'{force}/{b}={a} N/mm²; 1 N/mm²=1 MPa.'
        elif family == 1:
            question = f'In the linear elastic region, axial stress is {a*b} MPa and strain is {b/1000:g}. Calculate Young\'s modulus.'
            formula, answer, result, source = 'E=σ/ε', f'{a} GPa', a, MECHANICS
            working = f'{a*b}/({b}/1000)={a*1000} MPa={a} GPa.'
        elif family == 2:
            question = f'Assume constant specific heat {a} J/(kg·K), mass {b} kg, and temperature rise {c} K with no phase change. Calculate heat transferred.'
            formula, answer, result, source = 'Q=mcΔT', f'{a*b*c} J', a*b*c, HEAT
            working = f'{b}×{a}×{c}={a*b*c} J.'
        else:
            question = f'Assume constant linear expansion coefficient {a}×10^-6 K^-1, length {b} m, and temperature rise {c} K. Calculate length increase.'
            result = a*b*c/1000
            formula, answer, source = 'ΔL=αLΔT', f'{result:g} mm', HEAT
            working = f'{a}×10^-6×{b}×{c} m={result:g} mm.'
        # Number alone is not considered success; retain units and applicability in the label.
        add(f'formula-{index}', question, '', f'{formula}. {working} Answer: {answer}.',
            f'formula:{family}', source)
        numeric.append(dict(id=f'formula-{index}',module='Materials calculations',
                            question=question,formula=formula,working=working,answer=answer))

    for index in range(100):
        threshold = rng.randint(100,500)
        low, high = threshold-10, threshold+20
        density = rng.randint(2000,6000)
        candidates = [(high,density+1000), (low,density-500), (high+10,density)]
        rng.shuffle(candidates)
        if index % 5 == 0:
            candidates = [(low-offset*10,d) for offset,(_,d) in enumerate(candidates)]
        feasible = [(label,y,d) for label,(y,d) in zip('ABC',candidates) if y >= threshold]
        question = f'Case {index}: require yield strength ≥{threshold} MPa. Choose the lowest-density feasible hypothetical candidate, or report no feasible candidate.'
        context = ' '.join(f'{label}: yield {y} MPa, density {d} kg/m³.' for label,(y,d) in zip('ABC',candidates))+' Only these two criteria apply.'
        if feasible:
            label,y,d = min(feasible,key=lambda item:item[2])
            rejected = ', '.join(label for label,(y,_) in zip('ABC',candidates) if y < threshold)
            answer = f'Choose {label}. {rejected} fails the {threshold} MPa requirement. {label} has the lowest density among feasible candidates ({d} kg/m³). This decision uses only the supplied hypothetical criteria.'
        else:
            answer = f'No feasible candidate. A, B and C all fail the {threshold} MPa yield requirement. Lower density does not override this hard constraint.'
        add(f'decision-{index}',question,context,answer,'constraint_selection',f'hypothetical-decision-{index}')
    for index in range(100):
        name = f'Unknown compound U{index:03d}'
        answer = f'I currently do not have verified engineering data for {name} in my offline knowledge base. Please verify the material name or provide additional context.'
        add(f'unknown-{index}',f'Request engineering data for {name}.','Offline knowledge base has no record for this name.',answer,'unknown_fallback',f'unknown-{index}')

    # Split within task families, with distinct sample identities and question strings.
    grouped = {}
    seen_prompts = set()
    duplicates = 0
    for row in generated:
        key = ' '.join((row['instruction']+'\n'+row['input']).split()).casefold()
        if key in seen_prompts:
            duplicates += 1
            continue
        seen_prompts.add(key)
        grouped.setdefault(row['concept'],[]).append(row)
    for rows in grouped.values():
        rng.shuffle(rows)
        validation = max(1,len(rows)//10)
        test = max(1,len(rows)//10)
        splits['validation'].extend(rows[:validation])
        splits['test'].extend(rows[validation:validation+test])
        splits['train'].extend(rows[validation+test:])
    for name in splits:
        if not args.licensed_only:
            splits[name].extend(json.loads(line) for line in
                               (base/f'sft_{name}.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())
        (dataset/f'sft_{name}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in splits[name]),encoding='utf-8')
    test_ids = {r['id'] for r in splits['test']}
    # Use only test numeric examples here; never score the generated training questions.
    numerical = [r for r in numeric if r['id'] in test_ids]
    if not args.licensed_only:
        numerical.extend(json.loads(line) for line in (base/'numerical_problems.jsonl').read_text(encoding='utf-8').splitlines() if line.strip())
    (dataset/'numerical_problems.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in numerical),encoding='utf-8')
    if not args.licensed_only:
        shutil.copy2(base/'evaluation.csv',dataset/'evaluation.csv')
    report = dict(audit=audit_sft(splits), synthetic_examples=len(generated)-duplicates,
                  duplicate_questions_removed=duplicates, licensed_only=args.licensed_only,
                  profile_format_property_groups=len(names), formula_sources=[MECHANICS,HEAT],
                  limitations=['Profile tasks contain explicit placeholders, not verified material properties.',
                               'Held-out examples share task families; this does not test unseen-concept generalization.',
                               'Formula families share reference sources across splits.',
                               'Book corpus is selected separately by --books; no claims of complete knowledge for 66 properties.'])
    (out/'curriculum_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (out/'SHA256SUMS').open('w') as handle:
        for path in sorted(dataset.iterdir()):
            handle.write(hashlib.sha256(path.read_bytes()).hexdigest()+'  dataset/'+path.name+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
