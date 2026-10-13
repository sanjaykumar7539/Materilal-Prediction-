"""Admit curated chemical descriptors and stage unreviewed elemental values."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/chemistry_expansion'
KB = ROOT/'domains/materials/verified_records.json'


def main():
    source = json.loads((DATA/'property_records.json').read_text(encoding='utf-8'))
    provenance = json.loads((DATA/'PROVENANCE.json').read_text(encoding='utf-8'))
    coverage = json.loads((DATA/'element_book_index.json').read_text(encoding='utf-8'))
    collection = json.loads(KB.read_text(encoding='utf-8'))
    backup = KB.parent/'verified_records_before_chemistry_expansion.json'
    if not backup.exists():
        shutil.copy2(KB,backup)
    terms = {}
    for folder, pattern in [(DATA/'originals','chebi_terms_*.json'),(ROOT/'data/periodic_chemistry/originals','term_*.json')]:
        for path in folder.glob(pattern):
            for term in json.loads(path.read_text(encoding='utf-8')).get('_embedded',{}).get('terms',[]):
                terms[term['obo_id']] = term
    elements = {int(r['properties'][0]['value']):r for r in collection['materials'] if r['name'].endswith('(element)')}
    compounds = {r['name'].removesuffix(' (compound)').casefold():r for r in collection['materials'] if r['name'].endswith('(compound)')}
    for record in source:
        if record['kind']=='element':
            target = elements[record['atomic_number']]
            names = {p['name'] for p in record['properties']}
            target['properties'] = [p for p in target['properties'] if p['name'] not in names] + record['properties']
            if any(p['verified'] for p in record['properties']):
                raise ValueError('Elemental source values must remain unverified')
            target['related_books'] = [dict(title=b['title'],source=b['url'],relationship=b['relationship']) for b in coverage[record['name']]['general_book_references']]
            target['book_keyword_mentions'] = coverage[record['name']]['keyword_mentions']
            continue
        term = terms[record['id']]
        annotation = term['annotation']
        if term.get('is_obsolete') or '3_STAR' not in term.get('in_subset',[]) or term['label'] != record['name']:
            raise ValueError('Compound curation/identity check failed')
        expected = [str(annotation[key][0]) for key in ['generalized_empirical_formula','mass','charge']]
        if [p['value'] for p in record['properties']] != expected:
            raise ValueError('Compound descriptors differ from source')
        target = compounds.get(record['name'].casefold())
        if target is None:
            target = dict(name=f'{record["name"].capitalize()} (compound)',aliases=[f'compound {record["name"]}',f'compound {record["id"]}'],
                overview=f'{record["name"]} is the ChEBI chemical entity {record["id"]}. This profile provides its formula, calculated average molecular mass and formal net charge; these descriptors are not measured engineering-performance values.',
                overview_source=record['source'],verified=True,properties=[])
            collection['materials'].append(target)
        # Preserve conventional formulas already reviewed (e.g. NaCl rather
        # than ChEBI's component notation Cl.Na).
        existing_formula = next((p for p in target['properties'] if p['name']=='Chemical Formula'),None)
        target['properties'] = [existing_formula or record['properties'][0],*record['properties'][1:]]
        for prop in target['properties']:
            if 'CC BY 4.0' not in prop['source']:
                prop['source'] += ' — ChEBI, EMBL-EBI; CC BY 4.0; https://www.ebi.ac.uk/chebi/about/'
        target.update(review_date='2026-10-09',license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
            verification_scope='Curated formula, calculated mass and formal-charge transcription; not independent physical testing.')
    KB.write_text(json.dumps(collection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    digest = hashlib.sha256(KB.read_bytes()).hexdigest()
    totals = dict(total_records=len(collection['materials']),total_properties=sum(len(r['properties']) for r in collection['materials']),
        verified_properties=sum(p['verified'] for r in collection['materials'] for p in r['properties']),
        unverified_element_properties=sum(len(r['properties']) for r in source if r['kind']=='element'))
    audit = dict(review_date='2026-10-09',knowledge_base_sha256=digest,**totals,
        books=provenance['books'],sources=provenance['originals'],
        method='Elemental values staged unverified. Compound fields compared with curated ChEBI annotations; calculated mass explicitly distinguished from physical measurement.',
        limitations=provenance['limitations'])
    (KB.parent/'CHEMISTRY_EXPANSION_REVIEW.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (KB.parent/'element_book_index.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for filename in ['SOURCE_REVIEW.json','REFERENCE_SOURCE_REVIEW.json','CHEMISTRY_SOURCE_REVIEW.json']:
        path = KB.parent/filename
        prior = json.loads(path.read_text(encoding='utf-8'))
        prior['knowledge_base_sha256'] = digest
        if 'total_records' in prior:
            prior.update(totals)
        path.write_text(json.dumps(prior,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(totals))


if __name__ == '__main__':
    main()
