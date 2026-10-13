"""Score explicit identity fields in saved generations; never certify physics."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def fields(text):
    text = text.replace('*', '')
    result = {}
    names = ['Atomic Number', 'Chemical Symbol', 'Chemical Formula']
    for name in names:
        match = re.search(r'(?:^|\n)\s*' + re.escape(name) + r':\s*\n(.*?)(?=\n\s*(?:Atomic Number|Chemical Symbol|Chemical Formula):|\Z)', text, re.S)
        if match:
            value = re.search(r'Typical Value / Range:\s*([^\n]+)', match.group(1))
            if value:
                result[name] = value.group(1).strip().rstrip('.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    run = Path(args.run)
    rows = json.loads((run/'predictions.json').read_text(encoding='utf-8'))
    counts = {}
    reviewed = []
    for row in rows:
        expected, predicted = fields(row['reference']), fields(row['prediction'])
        if not expected:
            raise ValueError('Reference identity fields could not be parsed')
        correct = {name: predicted.get(name) == value for name,value in expected.items()}
        def overview(text):
            match = re.search(r'General Overview\s*\n(.*?)(?=\n#+\s*2\.|\Z)',text,re.S)
            return match.group(1).strip() if match else ''
        expected_overview, predicted_overview = overview(row['reference']), overview(row['prediction'])
        entity = expected_overview.split(' is ',1)[0]
        name_correct = bool(entity and re.search(r'(?<!\w)'+re.escape(entity)+r'(?!\w)',predicted_overview,re.I))
        identifier = re.search(r'CHEBI:\d+',expected_overview)
        identifier_correct = identifier is None or identifier.group(0) in predicted_overview
        for name, matches in correct.items():
            count = counts.setdefault(name,{'correct':0,'total':0})
            count['total'] += 1
            count['correct'] += int(matches)
        reviewed.append(dict(id=row['id'], expected=expected, predicted=predicted,
            field_matches=correct, overview_name_correct=name_correct, source_identifier_correct=identifier_correct,
            all_identity_fields_correct=all(correct.values()) and name_correct and identifier_correct))
    result = dict(questions=len(rows), all_identity_fields_correct=sum(r['all_identity_fields_correct'] for r in reviewed),
        by_field=counts, cases=reviewed,
        limitations=['Strict matching of explicitly labelled generated fields plus the overview entity name and ChEBI identifier when present; missing or malformed fields count as incorrect.',
            'Questions concern facts/entities seen in training; this does not assess unseen compounds, reasoning, physical properties or scientific reliability.'])
    result['identity_profile_accuracy'] = result['all_identity_fields_correct']/len(rows)
    (run/'identity_evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    export = run/'export'
    (export/'identity_evaluation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    summary = '\nIdentity-field evaluation: '+str(result['all_identity_fields_correct'])+'/'+str(len(rows))+' profiles have all identity fields correct ('+format(result['identity_profile_accuracy'],'.2%')+'). See identity_evaluation.json for field-level results. These are known-entity, held-out-prompt tests.\n'
    for dest in [run/'REPORT.md',export/'REPORT.md']:
        dest.write_text(dest.read_text(encoding='utf-8')+summary,encoding='utf-8')
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in sorted(export.iterdir()) if p.name != 'SHA256SUMS']
    (export/'SHA256SUMS').write_text('\n'.join(hashes)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['questions','all_identity_fields_correct','identity_profile_accuracy','by_field']}))


if __name__ == '__main__':
    main()
