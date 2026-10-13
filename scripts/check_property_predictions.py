"""Score explicit source-grounded value, unit and condition reproduction."""
import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re


def parse(text):
    flat = text.replace('*','')
    value = re.search(r'Typical Value / Range:\s*([^\n]+)',flat)
    unit = re.search(r'Unit / Scale:\s*([^\n]+)',flat)
    if not value or not unit:
        return None
    scalar, separator, conditions = value.group(1).partition(' (Conditions: ')
    return {'value':scalar.strip(),'unit':unit.group(1).strip(),
        'conditions':conditions.rstrip(')').strip() if separator else ''}


def same_value(expected, predicted):
    if expected == predicted:
        return True
    numeric = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
    if re.fullmatch(numeric,expected) and re.fullmatch(numeric,predicted):
        try:
            return Decimal(expected)==Decimal(predicted)
        except InvalidOperation:
            pass
    return False


def content_fields(text):
    property_name = re.search(r'^\* \*\*([^\n]+):\*\*$',text,re.M)
    knowledge = re.search(r'\*\*Knowledge:\*\* ([^\n]+)',text)
    return (property_name[1] if property_name else None,
            knowledge[1] if knowledge else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    run = Path(parser.parse_args().run)
    cases = []
    for row in json.loads((run/'predictions.json').read_text(encoding='utf-8')):
        expected, predicted = parse(row['reference']),parse(row['prediction'])
        if expected is None:
            raise ValueError('Reference fields missing')
        value_ok = predicted is not None and same_value(expected['value'],predicted['value'])
        unit_ok = predicted is not None and expected['unit']==predicted['unit']
        condition_ok = predicted is not None and expected['conditions']==predicted['conditions']
        overview_reference = row['reference'].split('General Overview\n',1)[1].split('\n',1)[0]
        name, _, status = overview_reference.partition(': ')
        overview_prediction = row['prediction'].split('General Overview\n',1)[-1].split('\n',1)[0]
        name_ok = overview_prediction.casefold().startswith(name.casefold()+': ')
        kind = status.split(' source record',1)[0]
        status_ok = kind in overview_prediction
        format_ok = (row['prediction'].count('### 1. General Overview')==1
                     and row['prediction'].count('### 2. Engineering Profile')==1
                     and len(re.findall(r'^\* \*\*[^\n]+:\*\*$',row['prediction'],re.M))==1)
        expected_content,actual_content = content_fields(row['reference']),content_fields(row['prediction'])
        property_ok = expected_content[0] is not None and actual_content[0]==expected_content[0]
        knowledge_ok = expected_content[1] is not None and actual_content[1]==expected_content[1]
        cases.append(dict(id=row['id'],property=expected_content[0],expected=expected,predicted=predicted,value_correct=value_ok,
            unit_correct=unit_ok,conditions_correct=condition_ok,entity_correct=name_ok,
            data_kind_correct=status_ok,format_correct=format_ok,
            property_correct=property_ok,knowledge_correct=knowledge_ok,
            value_and_unit_correct=value_ok and unit_ok,
            complete_source_profile_correct=value_ok and unit_ok and condition_ok and name_ok and status_ok and format_ok and property_ok and knowledge_ok))
    result = {'questions':len(cases),'cases':cases,
        'limitations':['Values were supplied in the question. This measures source-grounded reproduction, not closed-book recall or scientific reasoning.',
            'Numeric equivalence permits 0 versus 0.0. Other strings, units and conditions use strict reference matching.',
            'Data-kind/source conditions must be preserved; unreviewed elemental values are not certified by model evaluation.']}
    for key in ['value_correct','unit_correct','conditions_correct','entity_correct','data_kind_correct','format_correct','property_correct','knowledge_correct','value_and_unit_correct','complete_source_profile_correct']:
        result[key] = sum(bool(case[key]) for case in cases)
        result[key+'_accuracy'] = result[key]/len(cases)
    result['by_property'] = {}
    for case in cases:
        group = result['by_property'].setdefault(case['property'],{'questions':0,'value_and_unit_correct':0,'complete_profiles':0})
        group['questions'] += 1
        group['value_and_unit_correct'] += bool(case['value_and_unit_correct'])
        group['complete_profiles'] += bool(case['complete_source_profile_correct'])
    for folder in [run,run/'export']:
        (folder/'property_evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        with (folder/'REPORT.md').open('a',encoding='utf-8') as handle:
            handle.write(f'\nSource-grounded generated values and units correct: {result["value_and_unit_correct"]}/{len(cases)}. Complete profiles including entity, data kind, format and conditions: {result["complete_source_profile_correct"]}/{len(cases)}. This is not closed-book recall. See property_evaluation.json.\n')
    export = run/'export'
    sums=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name for p in sorted(export.iterdir()) if p.name!='SHA256SUMS']
    (export/'SHA256SUMS').write_text('\n'.join(sums)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','limitations','by_property']}))


if __name__=='__main__':
    main()
