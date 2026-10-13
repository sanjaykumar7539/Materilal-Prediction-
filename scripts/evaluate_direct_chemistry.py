"""Small direct-question diagnostic: no numerical source facts supplied to the model."""
import argparse
import json
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ownllm import OwnLLM, ModelConfig
from ownllm.knowledge.materials_policy import fallback
from ownllm.training.books import instruction_prompt
from train_books import generate_answer
from check_property_predictions import parse, same_value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    args=parser.parse_args()
    run=Path(args.run)
    torch.set_num_threads(4)
    state=torch.load(run/'export/ownllm.pt',map_location='cpu',weights_only=False)
    model=OwnLLM(ModelConfig.from_dict(state['config']))
    model.load_state_dict(state['model']); model.eval()
    tokenizer=Tokenizer.from_file(str(run/'export/tokenizer.json'))
    records=json.loads((ROOT/'domains/materials/verified_records.json').read_text(encoding='utf-8'))['materials']
    cases=[]
    for name,propname in [('Water (compound)','Chemical Formula'),
                          ('Sodium Chloride (compound)','Chemical Formula'),
                          ('Carbon Dioxide (compound)','Chemical Formula'),
                          ('Water (compound)','Calculated Molecular Mass'),
                          ('Sodium (element)','Atomic Number'),
                          ('Iron (element)','Atomic Number'),
                          ('Gold (element)','Chemical Symbol'),
                          ('Carbon (element)','Chemical Symbol')]:
        record=next(r for r in records if r['name'].casefold()==name.casefold())
        prop=next(p for p in record['properties'] if p['name']==propname and p['verified'] and p['applicable'])
        bare=record['name'].rsplit(' (',1)[0]
        question=f'Report {propname} for {bare}.'
        prediction=generate_answer(model,tokenizer,instruction_prompt({'instruction':question}),'cpu',224)
        fields=parse(prediction)
        correct=bool(fields and same_value(prop['value'],fields['value']) and prop['unit']==fields['unit'])
        cases.append({'question':question,'kind':'known','expected_value':prop['value'],
                      'expected_unit':prop['unit'],'prediction':prediction,'correct':correct})
    for name in ['Zyrometallium-X9','Invented compound QZ-17','Unlisted alloy ZZ999']:
        assert not any(name.casefold() in [r['name'].casefold(),*[a.casefold() for a in r['aliases']]] for r in records)
        question=f'Report the engineering profile for {name}.'
        prediction=generate_answer(model,tokenizer,instruction_prompt({'instruction':question}),'cpu',224)
        cases.append({'question':question,'kind':'unknown','expected':fallback(name),
                      'prediction':prediction,'correct':prediction.strip()==fallback(name)})
    report={'cases':cases,'known_questions':8,'known_value_unit_correct':sum(c['correct'] for c in cases if c['kind']=='known'),
            'unknown_questions':3,'unknown_exact_fallback_correct':sum(c['correct'] for c in cases if c['kind']=='unknown'),
            'limitations':['Small diagnostic, not a statistically representative benchmark.',
                           'Facts can have appeared in earlier training. Only source facts in the prompt are withheld.',
                           'Value/unit checks do not certify complete-profile accuracy. Unknown fallback is exact.',
                           'This tests native weights; verified chat responses are evaluated separately.']}
    for folder in [run,run/'export']:
        (folder/'direct_questions.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','limitations']}))


if __name__=='__main__':
    main()
