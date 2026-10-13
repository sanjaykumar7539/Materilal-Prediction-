"""Import a bounded, manually reviewed collection from licensed local editions.

No network calls or model training. Generic aliases deliberately require the
word 'generic', so a request for an unknown alloy cannot select these records.
"""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'domains/materials/verified_records.json'
DATE = '2026-10-08'
# Material, table-12.1 Young modulus (GPa), table-1.3 heat capacity,
# table-1.5 conductivity. Empty cells remain omitted, never extrapolated.
ROWS = [
    ('Aluminium', 'Aluminum', '70', '900', '220'),
    ('Copper', 'Copper', '110', '387', '390'),
    ('Iron', 'Iron', '210', '452', None),
    ('Lead', 'Lead', '16', '128', None),
    ('Nickel', 'Nickel', '210', None, None),
    ('Gold', 'Gold', None, '129', '318'),
    ('Silver', 'Silver', None, '235', '420'),
    ('Diamond', 'Diamond', None, None, '2000'),
    ('Glass', 'Glass', None, '840', '0.84'),
    ('Water', 'Water', None, '4186', '0.6'),
    ('Ethanol', 'Ethanol', None, '2450', None),
    ('Ice', 'Ice', None, '2090', '2.2'),
    ('Wood', 'Wood', None, '1700', '0.08–0.16'),
    ('Cork', 'Cork', None, None, '0.042'),
    ('Glycerin', 'Glycerin', None, '2410', None),
    ('Mercury', 'Mercury', None, '139', None),
    ('Polystyrene foam', 'Polystyrene foam', None, None, '0.010'),
]


def main():
    manifest = json.loads((ROOT / 'data/open_materials/LICENSE_MANIFEST.json').read_text(encoding='utf-8'))
    books = {}
    for volume, info in enumerate(manifest, 1):
        path = ROOT / f'data/open_materials/originals/volume_{volume}.pdf'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info['sha256']:
            raise ValueError(f'Source bytes changed: {path}')
        reader = PdfReader(path)
        declaration = ' '.join((reader.pages[3].extract_text() or '').split())
        if 'Creative Commons Attribution 4.0 International License (CC BY 4.0)' not in declaration:
            raise ValueError(f'Expected edition license missing: {path}')
        books[volume] = (info, reader)

    # Retain the reviewed table evidence locally for repeatable checks.
    evidence = {}
    for volume, page, label in [(1, 614, '12.1'), (2, 32, '1.3'), (2, 49, '1.5')]:
        text = books[volume][1].pages[page-1].extract_text(extraction_mode='layout')
        if f'Table {label}' not in text:
            raise ValueError(f'Wrong source page for table {label}')
        evidence[label] = text

    def source(volume, page, table, row):
        info = books[volume][0]
        return f"{info['url']} — PDF page {page}, Table {table}, {row} row; {info['authors']}, OpenStax/Rice University; CC BY 4.0; Download for free at {info['attribution_url']}"

    records = []
    for name, row, modulus, heat, conductivity in ROWS:
        properties = []
        def add(title, description, value, unit, condition, volume, page, table, source_row):
            if value is None:
                return
            # This is a membership check, not a replacement for the manual
            # review of columns, units and footnotes documented in the audit.
            if source_row.split(' (')[0] not in evidence[table]:
                raise ValueError(f'Source row missing: {source_row}')
            properties.append(dict(name=title, description=description, value=value,
                unit=unit, conditions=condition + ' Textbook reference approximation; grade/composition, processing and test method not specified; not a specification limit or design allowable.',
                source=source(volume, page, table, source_row), verified=True,
                applicable=True, formula_applicable=False))
        add("Young's Modulus", 'Ratio of uniaxial elastic stress to elastic strain in the linear regime.',
            modulus, 'GPa', 'Approximate small-strain elastic modulus; temperature not specified in Table 12.1.', 1, 614, '12.1', row)
        heat_condition = 'Constant-volume specific heat at 25 °C, as stated in the table footnote.'
        if name == 'Water':
            heat_condition = 'Liquid water at 15.0 °C; constant-volume value as labelled by the table footnote.'
        elif name == 'Ice':
            heat_condition = 'Ice; average over −50 °C to 0 °C; constant-volume value as labelled by the table footnote.'
        heat_row = 'Iron, steel' if name == 'Iron' else row
        add('Specific Heat Capacity', 'Heat per unit mass and temperature increment under the stated constraints.',
            heat, 'J/(kg·K)', heat_condition, 2, 32, '1.3', heat_row)
        conductivity_row = 'Glass (average)' if name == 'Glass' else row
        conductivity_condition = 'Representative thermal conductivity near 0 °C.'
        if name == 'Water':
            conductivity_condition += ' Liquid phase.'
        elif name == 'Ice':
            conductivity_condition += ' Solid ice.'
        elif name == 'Polystyrene foam':
            conductivity_condition += ' Foam, not solid polystyrene; density not specified.'
        add('Thermal Conductivity', 'Coefficient relating conductive heat flux to a temperature gradient.',
            conductivity, 'W/(m·K)', conductivity_condition,
            2, 49, '1.5', conductivity_row)
        aliases = [f'generic {name.lower()}', f'generic {row.lower()}']
        if name == 'Glycerin':
            aliases.append('generic glycerol')
        info = books[1 if modulus else 2][0]
        records.append(dict(name=f'{name} (generic reference)', aliases=list(dict.fromkeys(aliases)),
            overview=f'Generic textbook reference for {name.lower()}, covering ' + ', '.join(p['name'].lower() for p in properties) + '. These published approximations describe the named material category; no specific commercial grade, purity or processing condition is established.',
            overview_source=properties[0]['source'], verified=True, properties=properties,
            review_date=DATE, verification_scope='Reviewed source transcription and table conditions; not independent measurement or grade certification.',
            license='CC BY 4.0', license_url='https://creativecommons.org/licenses/by/4.0/',
            attribution=f"{info['authors']}; OpenStax, Rice University; Download for free at {info['attribution_url']}",
            modification='Selected numerical table entries adapted into structured records; modulus converted from 10^10 Pa to GPa and temperature-difference units normalized to K.'))
    collection = json.loads(DEST.read_text(encoding='utf-8'))
    names = {r['name'] for r in records}
    collection['materials'] = [r for r in collection['materials'] if r['name'] not in names] + records
    DEST.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    audit = dict(review_date=DATE, records_added=len(records), properties_added=sum(len(r['properties']) for r in records),
        total_records=len(collection['materials']), total_properties=sum(len(r['properties']) for r in collection['materials']),
        knowledge_base_sha256=digest, sources=manifest, table_evidence=evidence,
        method='Reviewed table columns, row identities, numerical values and footnotes in the exact local PDFs; confirmed original PDF hashes and embedded licenses.',
        limitations=['Generic reference data only; no grade certification or independent laboratory measurements.',
            'Elastic table provides no measurement temperature; this is explicitly preserved.',
            'Ambiguous thermal rows such as Steel iron and Concrete brick were omitted.',
            'No interpolation, alloy substitution, inferred strength or training performed.'])
    (DEST.parent / 'REFERENCE_SOURCE_REVIEW.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    prior_path = DEST.parent / 'SOURCE_REVIEW.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    prior['knowledge_base_sha256'] = digest
    prior_path.write_text(json.dumps(prior, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['records_added','properties_added','total_records','total_properties']}))


if __name__ == '__main__':
    main()
