"""Import a bounded, source-reviewed stainless-steel starter collection."""
import hashlib
import json
from pathlib import Path

URL = 'https://www.outokumpu.com/en/products/product-ranges/-/media/files/products/core/outokumpu-core-range-datasheet.pdf'
path = Path('domains/materials/verified_records.json')
collection = json.loads(path.read_text(encoding='utf-8'))
records = []
for grade,en,uns,proof,tensile,summary in [
    ('304','1.4301','S30400','230','540–750','Chromium-nickel austenitic stainless steel used in kitchen equipment and process hardware, valued for corrosion resistance, formability and weldability.'),
    ('304L','1.4307','S30403','220','520–700','Low-carbon austenitic stainless steel derived from 304. Reduced carbide precipitation during welding helps resist intergranular corrosion; used in process equipment and heat exchangers.'),
    ('304LN','1.4311','S30453','290','550–750','Low-carbon, nitrogen-alloyed austenitic steel offering improved strength and low-temperature toughness; used for railway and pressure equipment.'),
    ('305','1.4303','S30500','220','500–650','Nickel-rich austenitic steel selected for deep drawing and complex forming because it work-hardens less than 304.'),
    ('321','1.4541','S32100','220','520–720','Titanium-stabilized austenitic steel used in exhaust and welded equipment for resistance to intergranular corrosion.'),
    ('347','1.4550','S34700','220','520–720','Niobium-stabilized austenitic steel suited to heated equipment and exhaust components requiring intergranular-corrosion resistance.'),
    ('301LN','1.4318','S30153','350','650–850','Low-carbon, nitrogen-alloyed austenitic steel used in lightweight vehicle and railway structures for elevated strength.'),
    ('301','1.4310','S30100','250','600–950','Austenitic steel with high work-hardening capacity, used for springs, conveyor components and mechanically loaded parts.'),
    ('304L (EN 1.4306)','1.4306','S30403','220','520–700','Higher-nickel 304L variant selected for improved formability and deep drawing.'),
    ('201','1.4372','S20100','350','680–880','Low-nickel austenitic steel with strong work hardening, used in household and transport equipment.'),
    ('201LN','1.4372','S20153','350','680–880','Low-nickel austenitic variant used for railway freight and bulk-handling equipment.'),
    ('441','1.4509','S43940','250','430–630','Stabilized nickel-free ferritic stainless steel originally developed for exhaust applications.'),
    ('439M','','S43932','205','415','Titanium- and niobium-stabilized ferritic steel used in exhaust and household equipment.'),
    ('4622','1.4622','S44330','300','430–630','Stabilized nickel-free ferritic steel containing about 21% chromium, used in appliances and tubular products.')]:
    source_grade = '304L' if grade=='304L (EN 1.4306)' else grade
    label = f'Core {source_grade}/{en}' if en else f'Core {source_grade}'
    mechanical = f'Outokumpu {label}; cold-rolled coil and sheet (C); 20 °C; EN 10088-2 table values, not a measured lot result or a universal design allowable.'
    physical = f'Outokumpu {label}; manufacturer physical-property table; 20 °C; datasheet reference value, not a measured lot result.'
    if grade=='439M':
        mechanical=mechanical.replace('EN 10088-2','ASTM A240').replace('Core 439M/;','Core 439M;')
        physical=physical.replace('Core 439M/;','Core 439M;')+' Outokumpu tested values.'
    if grade=='4622':
        mechanical=mechanical.replace('EN 10088-2','EN 10028-7')
        physical+=' Outokumpu tested values.'
    modulus,conductivity,capacity,expansion,resistivity = ('200','15','500','16.0 × 10^-6','0.73')
    if grade in ('201','201LN'):
        capacity,expansion,resistivity = (None,None,'0.70')
    if grade in ('441','439M','4622'):
        modulus,conductivity,capacity,expansion,resistivity = ('220','21' if grade=='4622' else '25','460','10.0 × 10^-6','0.65' if grade=='4622' else '0.60')
    props = []
    for name,description,value,unit,conditions,page,table in [
        ('Yield Strength','Stress at the specified 0.2% permanent strain offset (Rp0.2).',proof,'MPa',mechanical,8,5),
        ('Tensile Strength','Maximum engineering stress reached during a tensile test (Rm).',tensile,'MPa',mechanical,8,5),
        ("Young's Modulus",'Elastic stiffness relating uniaxial stress to elastic strain.',modulus,'GPa',physical,10,7),
        ('Thermal Conductivity','Coefficient relating conductive heat flux to a temperature gradient.',conductivity,'W/(m·K)',physical,10,7),
        ('Specific Heat Capacity','Heat required per unit mass and temperature increment.',capacity,'J/(kg·K)',physical,10,7),
        ('Thermal Expansion Coefficient','Mean relative linear expansion per temperature increment.',expansion,'K^-1',physical.replace('20 °C','20–100 °C interval'),10,7),
        ('Electrical Resistivity','Electrical resistance normalized by conductor geometry.',resistivity,'Ω·mm²/m',physical,10,7)]:
        if value is None:
            continue
        props.append(dict(name=name,description=description,value=value,unit=unit,conditions=conditions,
                          source=f'{URL} — printed page {page}, Table {table}, {label} row',
                          verified=True,applicable=True,formula_applicable=False))
    records.append(dict(name=f'{grade} stainless steel',
                        aliases=[grade]+([f'EN {en}'] if en else [])+[f'UNS {uns}',label],
                        overview=summary,overview_source=f'{URL} — printed page {4 if grade in ("441","439M","4622") else 2 if grade in ("304","304L","304LN","305","304L (EN 1.4306)") else 3}',verified=True,properties=props,
                        review_date='2026-10-08',verification_scope='Source transcription, unit and grade/condition review; no independent physical testing.'))
names = {r['name'] for r in records}
collection['materials'] = [r for r in collection.get('materials',[]) if r.get('name') not in names]+records
path.write_text(json.dumps(collection,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
audit = {'review_date':'2026-10-08','source_url':URL,'publisher':'Outokumpu',
         'records':len(records),'properties':sum(len(r['properties']) for r in records),'knowledge_base_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'method':'Compared source Tables 5 and 7 with transcribed values, units, exact EN grade and product/temperature conditions.',
         'limitations':['Manufacturer reference/specification data are not batch certificates or design allowables.',
                        'No open license is asserted; source attribution preserved, no bulk training corpus or external upload created.',
                        'General 304L alias refers explicitly to EN 1.4307; other 304L variants need their own records.',
                        'No historical facts or unsupported properties were invented.']}
Path('domains/materials/SOURCE_REVIEW.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'materials_in_collection':len(collection['materials']),'reviewed_properties':sum(len(r['properties']) for r in records),'destination':str(path)}))

if __name__ == '__main__':
    pass
