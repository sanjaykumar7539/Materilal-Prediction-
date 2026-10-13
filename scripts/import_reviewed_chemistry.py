"""Cross-check element identities and import reviewed identity-only records."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/periodic_chemistry'
KB = ROOT / 'domains/materials/verified_records.json'
URL = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/JSON'
DATE = '2026-10-08'


def main():
    path = DATA / 'originals/pubchem_periodic_table.json'
    if not path.exists():
        response = requests.get(URL,timeout=45)
        response.raise_for_status()
        path.write_bytes(response.content)
    table = json.loads(path.read_text(encoding='utf-8'))['Table']
    columns = table['Columns']['Column']
    primary = {int(r['Cell'][0]):dict(zip(columns,r['Cell'])) for r in table['Row']}
    if set(primary) != set(range(1,119)):
        raise ValueError('Expected all 118 PubChem element identities')
    original = json.loads((DATA/'originals/elements.json').read_text(encoding='utf-8'))['elements']
    original = {e['number']:e for e in original if 1 <= e['number'] <= 118}
    spellings = {'aluminium':'aluminum','caesium':'cesium'}
    def normalize(name):
        return spellings.get(name.casefold(),name.casefold())
    records = []
    def prop(name, description, value, unit, source, condition):
        return dict(name=name,description=description,value=value,unit=unit,source=source,
            conditions=condition,verified=True,applicable=True,formula_applicable=False)
    for number,row in sorted(primary.items()):
        base = original[number]
        if base['symbol'] != row['Symbol'] or normalize(base['name']) != normalize(row['Name']):
            raise ValueError(f'Element identity mismatch at {number}')
        name = row['Name']
        source = f'{URL} — AtomicNumber {number} row, Name/Symbol/AtomicNumber fields; reviewed {DATE}'
        aliases = [f'element {name.lower()}',f'element {base["name"].lower()}',f'element {row["Symbol"]}']
        records.append(dict(name=f'{name} (element)',aliases=list(dict.fromkeys(aliases)),
            overview=f'{name} is the chemical element with symbol {row["Symbol"]} and atomic number {number}. This record establishes elemental identity only; it does not establish an alloy grade or measured engineering properties.',
            overview_source=source,verified=True,review_date=DATE,
            verification_scope='Name, symbol and atomic number checked against PubChem; no physical-property certification.',
            properties=[prop('Atomic Number','Number of protons in the nucleus.',str(number),'dimensionless integer',source,'Element identity; isotope-independent; temperature and processing do not change atomic number.'),
                prop('Chemical Symbol','Standard element notation in the periodic table.',row['Symbol'],'chemical symbol',source,'Element nomenclature; case-sensitive symbol, not a material grade.')]))
    provenance = json.loads((DATA/'PROVENANCE.json').read_text(encoding='utf-8'))
    terms = {}
    for term_path in (DATA/'originals').glob('term_*.json'):
        for term in json.loads(term_path.read_text(encoding='utf-8')).get('_embedded',{}).get('terms',[]):
            terms[term['obo_id']] = term
    for record in provenance['compound_records']:
        name, formula, identifier = record['name'],record['formula'],record['chebi_id']
        term = terms[identifier]
        if term['is_obsolete'] or '3_STAR' not in term.get('in_subset',[]) or term['label'].casefold() != name.casefold():
            raise ValueError(f'Compound curation/identity check failed: {identifier}')
        if term['annotation']['generalized_empirical_formula'] != [record['source_formula']]:
            raise ValueError(f'Compound source formula changed: {identifier}')
        def atoms(value):
            counts = Counter()
            for symbol, count in re.findall(r'([A-Z][a-z]?)(\d*)',value):
                counts[symbol] += int(count or 1)
            return counts
        if atoms(formula) != atoms(record['source_formula']):
            raise ValueError(f'Compound formula stoichiometry mismatch: {identifier}')
        source = f'{record["source"]} — ChEBI curated formula record; reviewed {DATE}; CC BY 4.0; EMBL-EBI, https://www.ebi.ac.uk/chebi/about/'
        records.append(dict(name=f'{name.capitalize()} (compound)',aliases=[f'compound {name}',f'compound {identifier}'],
            overview=f'{name.capitalize()} is identified by ChEBI as {identifier}, with chemical formula {formula}. This is an identity record; no phase, hydration variant or measured engineering-property values beyond the specified entity are asserted.',
            overview_source=source,verified=True,review_date=DATE,license='CC BY 4.0',
            license_url='https://creativecommons.org/licenses/by/4.0/',
            verification_scope='Curated source formula transcription; conventional ordering preserves elemental counts.',
            properties=[prop('Chemical Formula','Elemental composition notation for the specified chemical entity; ionic materials use a formula unit.',formula,'chemical formula',source,
                f'Specific entity {identifier}; original ChEBI notation {record["source_formula"]}; not a solution concentration, reaction equation or measured physical property.')]))
    collection = json.loads(KB.read_text(encoding='utf-8'))
    names = {r['name'] for r in records}
    collection['materials'] = [r for r in collection['materials'] if r['name'] not in names]+records
    KB.write_text(json.dumps(collection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    digest = hashlib.sha256(KB.read_bytes()).hexdigest()
    totals = dict(total_records=len(collection['materials']),total_properties=sum(len(r['properties']) for r in collection['materials']))
    audit = dict(review_date=DATE,primary_element_url=URL,primary_element_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        elements_cross_checked=118,compounds=20,properties_added=sum(len(r['properties']) for r in records),
        knowledge_base_sha256=digest,**totals,
        limitations=['Identity fields only; no physical values from the PubChem export imported.',
            'No blanket open-license claim for PubChem contributed physical-property data.',
            'Aliases require element/compound prefixes to avoid silently substituting generic identities for alloy grades.',
            'ChEBI formula data is CC BY 4.0; source IDs and original notation retained.'])
    (KB.parent/'CHEMISTRY_SOURCE_REVIEW.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    for filename in ['SOURCE_REVIEW.json','REFERENCE_SOURCE_REVIEW.json']:
        p = KB.parent/filename
        prior = json.loads(p.read_text(encoding='utf-8'))
        prior['knowledge_base_sha256'] = digest
        if 'total_records' in prior:
            prior.update(totals)
        p.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'elements_cross_checked':118,'compounds_added':20,**totals}))


if __name__ == '__main__':
    main()
