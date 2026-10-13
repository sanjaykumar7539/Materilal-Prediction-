"""Download specific OpenStax editions and admit only PDFs declaring CC BY 4.0."""
import hashlib
import json
from pathlib import Path
import re

from pypdf import PdfReader
import requests

ROOT = Path('data/open_materials')
SOURCES = [
    ('University Physics Volume 1', 'https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/UniversityPhysicsVolume1-LR.pdf',
     'https://openstax.org/details/books/university-physics-volume-1'),
    ('University Physics Volume 2', 'https://d3bxy9euw4e147.cloudfront.net/oscms-prodcms/media/documents/UniversityPhysicsVolume2-LR.pdf',
     'https://openstax.org/details/books/university-physics-volume-2'),
]


def main():
    (ROOT/'originals').mkdir(parents=True,exist_ok=True)
    (ROOT/'books').mkdir(exist_ok=True)
    manifest = []
    for index,(title,url,attribution_url) in enumerate(SOURCES,1):
        path = ROOT/'originals'/f'volume_{index}.pdf'
        if not path.exists():
            response = requests.get(url,timeout=(30,180),stream=True)
            response.raise_for_status()
            partial = path.with_suffix('.partial')
            with partial.open('wb') as handle:
                for chunk in response.iter_content(1024*1024):
                    handle.write(chunk)
            partial.replace(path)
        reader = PdfReader(path)
        front = '\n'.join(page.extract_text() or '' for page in reader.pages[:15])
        flat = re.sub(r'\s+',' ',front)
        declaration = re.search(r'Creative Commons Attribution 4\.0 International.{0,250}',flat,re.I)
        if not declaration:
            raise SystemExit(f'{title}: cannot confirm CC BY 4.0 in this edition; not admitted to training.')
        # Exclude front matter; stable page blocks permit distinct held-out book segments.
        chunks = 0
        for start in range(15,len(reader.pages),20):
            text = '\n\n'.join(page.extract_text() or '' for page in reader.pages[start:start+20])
            if len(text.strip()) < 200:
                continue
            (ROOT/'books'/f'volume_{index}_pages_{start+1:04d}_{min(start+20,len(reader.pages)):04d}.txt').write_text(text,encoding='utf-8')
            chunks += 1
        manifest.append(dict(title=title,authors='Samuel J. Ling, Jeff Sanny, William Moebs',
                             publisher='OpenStax, Rice University',url=url,attribution_url=attribution_url,
                             license='CC BY 4.0',license_url='https://creativecommons.org/licenses/by/4.0/',
                             license_evidence=declaration.group(0),pages=len(reader.pages),text_blocks=chunks,
                             modification='PDF text extracted into 20-page blocks; front matter excluded.',
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(json.dumps({'title':title,'pages':len(reader.pages),'text_blocks':chunks,'license':'CC BY 4.0'}),flush=True)
    (ROOT/'LICENSE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (ROOT/'ATTRIBUTION.md').write_text('# Training text attribution\n\n'+ '\n\n'.join(
        f"{r['title']} by {r['authors']}, OpenStax/Rice University. {r['attribution_url']}\n"
        f"Specific downloaded edition: {r['url']}\nLicense: CC BY 4.0 ({r['license_url']}). "
        f"Changes: {r['modification']}" for r in manifest)+
        '\n\nLater editions may use different licenses. This manifest records the downloaded bytes and embedded declaration; it does not grant rights to OpenStax trademarks.\n',encoding='utf-8')


if __name__ == '__main__':
    main()
