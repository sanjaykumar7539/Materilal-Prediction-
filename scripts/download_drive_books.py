"""Download the grounded public Drive file manifest; originals are preserved."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='data/drive_books/source_manifest.json')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    import requests
    manifest_path = Path(args.manifest)
    root = manifest_path.parent.resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    report_path = root / (manifest_path.stem + '_download_report.json')
    def download(entry):
        target = (root / entry['relative_path']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Download path leaves the data directory')
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = int(entry.get('size') or 0)
        if not target.exists() or (expected and target.stat().st_size != expected):
            temporary = target.with_suffix(target.suffix + '.partial')
            for attempt in range(3):
                try:
                    url = 'https://drive.usercontent.google.com/download'
                    offset = temporary.stat().st_size if temporary.exists() else 0
                    if expected and offset == expected:
                        temporary.replace(target)
                        break
                    headers = {'Range':f'bytes={offset}-'} if offset and offset < expected else {}
                    with requests.get(url, params={'id':entry['id'],'export':'download','confirm':'t'}, headers=headers, stream=True, timeout=(30,120)) as response:
                        response.raise_for_status()
                        if 'text/html' in response.headers.get('Content-Type',''):
                            raise ValueError('Google Drive returned a download page rather than file bytes')
                        append = response.status_code == 206 and offset > 0
                        if append and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                            raise ValueError('Download resumed at an unexpected offset')
                        with temporary.open('ab' if append else 'wb') as output:
                            for chunk in response.iter_content(1024*1024):
                                output.write(chunk)
                    if expected and temporary.stat().st_size != expected:
                        raise ValueError('Download is incomplete')
                    temporary.replace(target)
                    break
                except Exception as error:
                    if attempt == 2:
                        return {'id':entry['id'], 'name':entry['name'], 'status':'failed', 'error':type(error).__name__}
                    time.sleep(3)
        with target.open('rb') as binary:
            digest = hashlib.file_digest(binary, 'sha256').hexdigest()
        return {'id':entry['id'], 'name':entry['name'], 'relative_path':entry['relative_path'], 'status':'ready', 'bytes':target.stat().st_size, 'sha256':digest}
    results = []
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 8))) as executor:
        futures = [executor.submit(download, entry) for entry in manifest['files']]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            report_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
            print(json.dumps({'completed':len(results), 'total':len(futures), 'name':result['name'], 'status':result['status']}), flush=True)
    failures = [r for r in results if r['status'] != 'ready']
    print(json.dumps({'ready':len(results)-len(failures), 'failed':len(failures)}), flush=True)
    if failures:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
