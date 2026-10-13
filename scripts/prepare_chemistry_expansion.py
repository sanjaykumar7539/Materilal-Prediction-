"""Acquire licensed chemistry books/data and prepare source-grounded property QA."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import zipfile

from pypdf import PdfReader
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/chemistry_expansion'
BOOKS = [('Chemistry','Chemistry-LR.pdf','https://openstax.org/details/books/chemistry'),
         ('Chemistry: Atoms First','ChemistryAtomsFirst-LR.pdf','https://openstax.org/details/books/chemistry-atoms-first')]


def main():
    originals = OUT/'originals'
    originals.mkdir(parents=True,exist_ok=True)
    session = requests.Session()
    def fetch(url,filename,params=None):
        path = originals/filename
        if path.exists():
            return path
        with session.get(url,params=params,timeout=(30,180),stream=True) as response:
            response.raise_for_status()
            part = path.with_suffix(path.suffix+'.partial')
            size = 0
            with part.open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk)
                    size += len(chunk)
                    if size//(8*1024*1024) != (size-len(chunk))//(8*1024*1024):
                        print(json.dumps({'download':filename,'bytes':size}),flush=True)
                    if size>160*1024*1024:
                        raise ValueError('Source download exceeds limit')
            part.replace(path)
        return path

    elements = [e for e in json.loads((ROOT/'data/periodic_chemistry/originals/elements.json').read_text(encoding='utf-8'))['elements'] if 1 <= e['number'] <= 118]
    if len(elements) != 118:
        raise ValueError('Expected all 118 elements')
    coverage = {e['name']: {'general_book_references':[], 'keyword_mentions':[]} for e in elements}
    books, corpus = [], []
    for title,filename,link in BOOKS:
        host = 'https://d3bxy9euw4e147.cloudfront.net' if filename=='ChemistryAtomsFirst-LR.pdf' else 'https://assets.openstax.org'
        url = host+'/oscms-prodcms/media/documents/'+filename
        path = fetch(url,filename)
        cache = OUT/(filename+'.pages.json')
        if cache.exists():
            pages = json.loads(cache.read_text(encoding='utf-8'))
        else:
            reader = PdfReader(path)
            pages = []
            for i,page in enumerate(reader.pages):
                pages.append(page.extract_text() or '')
                if (i+1)%100==0:
                    print(json.dumps({'book':title,'extracted_pages':i+1,'total_pages':len(reader.pages)}),flush=True)
            cache.write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
        front = ' '.join('\n'.join(pages[:15]).split())
        if not re.search(r'Creative Commons Attribution 4\.0 International',front):
            raise ValueError(f'{title}: expected CC BY 4.0 edition not confirmed; excluded')
        info = dict(title=title,url=url,attribution_url=link,license='CC BY 4.0',
            authors=('Edward J. Neth, ' if filename=='ChemistryAtomsFirst-LR.pdf' else '')+'Paul Flowers, Klaus Theopold, Richard Langley and William R. Robinson; OpenStax/Rice University',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pages=len(pages),
            license_evidence=front[front.index('Creative Commons Attribution 4.0 International'):][:260])
        books.append(info)
        for item in coverage.values():
            item['general_book_references'].append({'title':title,'url':url,'attribution_url':link,'relationship':'General chemistry background, not a dedicated book or verified property source for this specific element.'})
        chapter = 1
        for i,text in enumerate(pages[15:],16):
            header = text[:250]+'\n'+text[-250:]
            match = re.search(r'Chapter\s+(\d+)\b',header,re.I)
            if match:
                chapter = int(match.group(1))
            if len(text.strip())>=150:
                corpus.append(dict(source=f'{title} — chapter {chapter}',page=i,text=text))
            for name,item in coverage.items():
                if len(item['keyword_mentions'])<30 and re.search(r'(?<!\w)'+re.escape(name)+r'(?!\w)',text,re.I):
                    item['keyword_mentions'].append({'title':title,'pdf_page':i,'relationship':'Keyword occurrence only; may not be substantive discussion.'})
        print(json.dumps({'book_ready':title,'pages':len(pages),'license':'CC BY 4.0'}),flush=True)

    terms = {}
    # Keep the 20 previously selected common compounds first.
    for path in (ROOT/'data/periodic_chemistry/originals').glob('term_*.json'):
        for term in json.loads(path.read_text()).get('_embedded',{}).get('terms',[]):
            terms[term['obo_id']] = term
    for page in range(32):
        if len(terms)>=200:
            break
        path = fetch('https://www.ebi.ac.uk/ols4/api/ontologies/chebi/terms',f'chebi_terms_{page}.json',{'size':500,'page':page})
        for term in json.loads(path.read_text()).get('_embedded',{}).get('terms',[]):
            annotation = term.get('annotation',{})
            if not term.get('is_obsolete') and '3_STAR' in term.get('in_subset',[]) and annotation.get('generalized_empirical_formula') and annotation.get('mass') and annotation.get('charge'):
                terms.setdefault(term['obo_id'],term)
                if len(terms)>=200:
                    break
        print(json.dumps({'compound_candidates':len(terms),'source_page':page}),flush=True)
    if len(terms)<100:
        raise ValueError('Insufficient curated compound records; inspect source pagination')

    records = []
    def property_record(name,description,value,unit,source,conditions,kind='reference',verified=False):
        return dict(name=name,description=description,value=str(value),unit=unit,source=source,
            conditions=conditions,data_kind=kind,verified=verified,applicable=True,formula_applicable=False)
    for e in elements:
        source = e['source']
        props = []
        for key,name,unit,definition in [('atomic_mass','Reported Atomic Mass','u','Source-reported atomic-mass datum; not necessarily a standard atomic weight.'),
            ('electron_configuration_semantic','Electron Configuration','orbital notation','Source electron-configuration notation; a leading asterisk marks an unconfirmed configuration.'),
            ('period','Periodic Table Period','dimensionless integer','Row position in the periodic table.'),
            ('group','Periodic Table Group','dimensionless integer','Source group assignment; conventions for f-block assignments may differ.'),
            ('melt','Melting Point','K','Solid-liquid transition temperature; for carbon and arsenic the source instead reports the triple-point temperature.'),
            ('boil','Boiling Point','K','Source-reported liquid-gas transition temperature.'),
            ('density','Density',e.get('density_unit'),'Mass per unit volume at the stated source conditions.'),
            ('molar_heat','Molar Heat Capacity','J/(mol·K)','Heat per amount of substance and temperature increment under the source constraints.')]:
            value = e.get(key)
            if value is None or unit is None or (isinstance(value,float) and not math.isfinite(value)):
                continue
            if key=='density' and unit not in ('g/L','g/cm3'):
                continue
            uncertain = (e['number']>92 and key in ('melt','boil','density','molar_heat')) or (key=='electron_configuration_semantic' and str(value).startswith('*'))
            conditions = 'Community source aggregation; not independently verified; processing/purity and test method unspecified.'
            if uncertain:
                conditions = 'Measurement/confirmation status unconfirmed; heavy-element source values may be theoretical. Do not treat as verified experimental data. '+conditions
            if key=='density' and e['phase']=='Gas':
                conditions += ' Gas density: 0 °C and 1 atm, per dataset documentation.'
            elif key in ('melt','boil'):
                conditions += ' Pressure not individually established by this record.'
            if key=='melt' and e['number'] in (6,33):
                name='Triple-Point Temperature'
                conditions += ' Carbon/arsenic source exception: do not interpret this as ordinary atmospheric melting.'
            props.append(property_record(name,definition,value,unit,source,conditions,'unconfirmed' if uncertain else 'source-aggregated'))
        records.append(dict(name=e['name'],kind='element',atomic_number=e['number'],properties=props,source=source))
    for term in list(terms.values())[:200]:
        annotation = term['annotation']
        formula = annotation['generalized_empirical_formula'][0]
        mass, charge = annotation['mass'][0],annotation['charge'][0]
        if not math.isfinite(float(mass)) or float(mass)<=0 or not math.isfinite(float(charge)):
            raise ValueError('Invalid molecular property')
        source = term['iri']
        props = [property_record('Chemical Formula','Source formula notation; a dot separates components.',formula,'chemical formula',source,'Specific ChEBI entity, not a concentration or phase-dependent measurement.','identity',True),
            property_record('Calculated Molecular Mass','Average mass calculated from the specified chemical composition.',mass,'Da',source,'ChEBI calculated average mass; not an experimentally measured bulk engineering property.','calculated',True),
            property_record('Net Charge','Formal net charge of the specified chemical entity.',charge,'e',source,'Source entity charge in units of the elementary charge, not an electrochemical potential.','identity',True)]
        records.append(dict(name=term['label'],kind='compound',id=term['obo_id'],source=source,properties=props))
    OUT.mkdir(exist_ok=True)
    (OUT/'property_records.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'book_corpus.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in corpus),encoding='utf-8')
    (OUT/'element_book_index.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    facts = []
    for record in records:
        for p in record['properties']:
            context = f'Local source record: material={record["name"]}; property={p["name"]}; value={p["value"]}; unit={p["unit"]}; data_kind={p["data_kind"]}; conditions={p["conditions"]}'
            answer = f'### 1. General Overview\n{record["name"]}: {p["data_kind"]} source record for {p["name"]}.\n\n### 2. Engineering Profile\n* **{p["name"]}:**\n  * **Knowledge:** {p["description"]}\n  * **Typical Value / Range:** {p["value"]} (Conditions: {p["conditions"]})\n  * **Unit / Scale:** {p["unit"]}'
            # prepare() uses question+answer; supplied context is explicitly part
            # of the question so the model must learn source-grounded reading.
            facts.append(dict(question=f'Using this local record, report {p["name"]} for {record["name"]}.\n{context}',answer=answer,
                entity=record['name'],kind=record['kind'],property=p['name'],data_kind=p['data_kind'],source=p['source'],conditions=p['conditions']))
    rng = random.Random(42)
    by_entity = defaultdict(list)
    for fact in facts:
        by_entity[fact['entity']].append(fact)
    split = {name:[] for name in ['train','validation','test']}
    held = []
    for name,values in sorted(by_entity.items()):
        rng.shuffle(values)
        if len(values)>1:
            held.append(values.pop())
        split['train'].extend(values)
    rng.shuffle(held)
    split['test'] = held[:64]
    split['validation'] = held[64:128]
    split['train'].extend(held[128:])
    with zipfile.ZipFile(OUT/'property_sft.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,values in split.items():
            archive.writestr(f'split_dataset/reviewed_qa/{name}.jsonl',''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in values))
            archive.writestr(f'split_dataset/rag_knowledge/{name}.jsonl','')
    report = dict(elements=118,compounds=len(records)-118,properties=len(facts),books=books,
        element_properties=sum(len(r['properties']) for r in records if r['kind']=='element'),
        unconfirmed_properties=sum(p['data_kind']=='unconfirmed' for r in records for p in r['properties']),
        split_counts={name:len(values) for name,values in split.items()},
        license='Element property data CC BY-SA 3.0; ChEBI CC BY 4.0; exact downloaded OpenStax editions CC BY 4.0.',
        evaluation_scope='Held-out property questions with source values supplied in context; known entities. Measures source-grounded reading, not closed-book scientific recall or unseen-entity generalization.',
        limitations=['Element physical values are source-aggregated, not independently verified; heavy-element measurement/confirmation status explicitly labelled unconfirmed rather than fabricated as experimental or computed.',
            'Compound mass is calculated; formula and charge are identity data, not measured engineering performance.',
            'Book index separates keyword occurrences from general background; not all elements have dedicated discussion or books.',
            'Chapter splits within related textbooks can share underlying facts; no independent-book generalization claim.',
            'A bounded subset of up to 200 compounds is not all known compounds.'],
        originals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in originals.iterdir() if p.is_file() and not p.name.endswith('.partial')})
    (OUT/'PROVENANCE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil_text=(ROOT/'data/periodic_chemistry/ATTRIBUTION.md').read_text(encoding='utf-8')
    (OUT/'ATTRIBUTION.md').write_text(shutil_text+'\n\n## Chemistry books\n\n'+'\n\n'.join(f'{b["title"]} — {b["authors"]}. Download for free at {b["attribution_url"]}. Exact PDF: {b["url"]}. CC BY 4.0. Adaptation: text extracted by page and grouped by chapter.' for b in books),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['elements','compounds','properties','element_properties','unconfirmed_properties','split_counts']}),flush=True)


if __name__ == '__main__':
    main()
