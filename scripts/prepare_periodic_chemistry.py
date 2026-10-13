"""Download licensed identity data and build a bounded local chemistry SFT ZIP."""
import hashlib
import json
from pathlib import Path
import random
import re
from collections import Counter
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/periodic_chemistry'
ELEMENT_URL = 'https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/PeriodicTableJSON.json'
LICENSE_URL = 'https://raw.githubusercontent.com/Bowserinator/Periodic-Table-JSON/master/LICENSE.md'
COMPOUNDS = ['water', 'ethanol', 'methanol', 'carbon dioxide', 'carbon monoxide',
    'ammonia', 'methane', 'sodium chloride', 'calcium carbonate', 'hydrogen peroxide',
    'silicon dioxide', 'aluminium oxide', 'acetone', 'benzene', 'acetic acid',
    'sulfuric acid', 'nitric acid', 'sodium hydroxide', 'urea', 'glycerol']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    originals = OUT / 'originals'
    originals.mkdir(exist_ok=True)
    session = requests.Session()
    def download(url, filename, params=None):
        path = originals / filename
        if path.exists():
            return path.read_bytes()
        response = session.get(url, params=params, timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)
        return response.content
    license_text = download(LICENSE_URL, 'ELEMENT_LICENSE.md').decode()
    if 'Attribution-ShareAlike 3.0' not in license_text:
        raise ValueError('Unexpected element-data license')
    elements = json.loads(download(ELEMENT_URL, 'elements.json'))['elements']
    elements = sorted([e for e in elements if isinstance(e.get('number'), int) and 1 <= e['number'] <= 118], key=lambda e: e['number'])
    if [e['number'] for e in elements] != list(range(1,119)) or len({e['symbol'] for e in elements}) != 118:
        raise ValueError('Expected exactly 118 distinct elements')
    compounds = []
    failures = []
    for index, name in enumerate(COMPOUNDS):
        search = json.loads(download('https://www.ebi.ac.uk/ols4/api/search', f'search_{index}.json',
            {'q': name, 'ontology': 'chebi', 'exact': 'true', 'rows': 30}))
        choices = [d for d in search['response']['docs'] if d.get('label','').casefold() == name.casefold()]
        if len(choices) != 1:
            failures.append({'name': name, 'reason': 'No unique exact ChEBI label'})
            continue
        chosen = choices[0]
        term = json.loads(download('https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms', f'term_{index}.json', {'iri': chosen['iri']}))
        terms = term.get('_embedded',{}).get('terms',[])
        if len(terms) != 1:
            failures.append({'name':name, 'reason':'No unique ontology term'})
            continue
        item = terms[0]
        annotation = item.get('annotation',{})
        formulas = [v for k,values in annotation.items() if k.lower().split('/')[-1] in ('formula','chemical_formula','generalized_empirical_formula') for v in (values if isinstance(values,list) else [values]) if isinstance(v,str)]
        if len(set(formulas)) != 1:
            failures.append({'name':name, 'reason':'No unique source formula', 'annotation_keys':list(annotation)})
            continue
        conventional = {'ammonia':'NH3','sodium chloride':'NaCl','calcium carbonate':'CaCO3',
            'silicon dioxide':'SiO2','sulfuric acid':'H2SO4','sodium hydroxide':'NaOH'}
        display = conventional.get(name, formulas[0])
        def atoms(formula):
            counts = Counter()
            for symbol, count in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
                counts[symbol] += int(count or 1)
            return counts
        if atoms(display) != atoms(formulas[0]):
            raise ValueError(f'Formula normalization changed stoichiometry for {name}')
        compounds.append(dict(name=name, formula=display, source_formula=formulas[0], source=chosen['iri'], chebi_id=chosen['obo_id'], description=item.get('description',[])))
        print(json.dumps({'compound':name,'formula':formulas[0],'id':chosen['obo_id']}),flush=True)
    if not compounds:
        raise ValueError('No supported compound formulas acquired; inspect raw source responses')
    rows = []
    for element in elements:
        name, symbol, number = element['name'], element['symbol'], element['number']
        output = f'### 1. General Overview\n{name} is the chemical element with symbol {symbol} and atomic number {number}.\n\n### 2. Engineering Profile\n* **Atomic Number:**\n  * **Knowledge:** Number of protons in the nucleus.\n  * **Typical Value / Range:** {number}\n  * **Unit / Scale:** dimensionless integer\n* **Chemical Symbol:**\n  * **Knowledge:** Element notation in the periodic table.\n  * **Typical Value / Range:** {symbol}\n  * **Unit / Scale:** chemical symbol'
        prompts = [f'Give the periodic-table profile of {name}.', f'Identify the element {name}.', f'What is the atomic number and symbol of {name}?', f'Provide the element record for {symbol}.', f'Explain element number {number}.', f'Show the chemical identity of {name}.']
        for prompt in prompts:
            rows.append(dict(question=prompt, answer=output, entity=name, kind='element', source=element.get('source',ELEMENT_URL)))
    for compound in compounds:
        name, formula = compound['name'], compound['formula']
        output = f'### 1. General Overview\n{name.capitalize()} is recorded in ChEBI as {compound["chebi_id"]}, with chemical formula {formula}.\n\n### 2. Engineering Profile\n* **Chemical Formula:**\n  * **Knowledge:** Elemental composition notation for the specified chemical entity; ionic materials use a formula unit.\n  * **Typical Value / Range:** {formula}\n  * **Unit / Scale:** chemical formula'
        for prompt in [f'Give the chemical profile of {name}.', f'Identify the compound {name}.', f'What is the chemical formula of {name}?', f'Provide the ChEBI identity of {name}.']:
            rows.append(dict(question=prompt, answer=output, entity=name, kind='compound', source=compound['source']))
    # Every acquired entity occurs in training. Validation/test use distinct
    # prompts for known facts; these are NOT unseen-entity evaluations.
    train = rows[:]
    validation, test = [], []
    rng = random.Random(42)
    held = rng.sample(elements, 60)
    lookup = {row['entity']: row['answer'] for row in rows}
    for i, element in enumerate(held):
        target = validation if i < 20 else test
        target.append(dict(question=f'List the periodic identity fields for {element["name"]}.', answer=lookup[element['name']], entity=element['name'], kind='element'))
    for i, compound in enumerate(compounds):
        target = validation if i % 3 == 0 else test
        target.append(dict(question=f'List the chemical identity fields for {compound["name"]}.',answer=lookup[compound['name']],entity=compound['name'],kind='compound'))
    archive = OUT / 'chemistry_sft.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for split,values in [('train',train),('validation',validation),('test',test)]:
            z.writestr(f'split_dataset/reviewed_qa/{split}.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in values))
            z.writestr(f'split_dataset/rag_knowledge/{split}.jsonl','')
    report = dict(elements=118, compounds=len(compounds), compound_records=compounds, excluded_compounds=failures,
        split_counts={'train':len(train),'validation':len(validation),'test':len(test)},
        evaluation_scope='Distinct held-out prompts about identities/facts already represented in training; not unseen-entity generalization or engineering-property validation.',
        license='Mixed source licenses: elements CC BY-SA 3.0; ChEBI CC BY 4.0.',
        license_urls=[LICENSE_URL,'https://www.ebi.ac.uk/chebi/about/'],
        limitations=['Element collection is community-maintained; physical properties, predicted metrics and discovery claims excluded.',
            'Compound formulas are ChEBI source records; no measured engineering properties asserted.',
            'No records automatically promoted to verified engineering knowledge.',
            'This bounded compound set is not all known compounds.'],
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(originals.iterdir())})
    (OUT/'PROVENANCE.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    (OUT/'ATTRIBUTION.md').write_text('# Chemistry dataset attribution\n\nElement identities adapted from Bowserinator/Periodic-Table-JSON and its cited contributors: '+ELEMENT_URL+' . Derived element QA data: CC BY-SA 3.0, https://creativecommons.org/licenses/by-sa/3.0/ . Original license retained in originals/ELEMENT_LICENSE.md.\n\nCompound formulas adapted from ChEBI, EMBL-EBI: https://www.ebi.ac.uk/chebi/about/ . CC BY 4.0, https://creativecommons.org/licenses/by/4.0/ . Stable entity identifiers retained in PROVENANCE.json.\n\nModification: selected identity fields rendered as QA profiles; engineered properties and unsupported values excluded. Source licenses apply to their respective derived data.\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['elements','compounds','excluded_compounds','split_counts']}),flush=True)


if __name__ == '__main__':
    main()
