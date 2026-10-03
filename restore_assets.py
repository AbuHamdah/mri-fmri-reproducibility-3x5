"""Restore byte-exact release assets; requires Python 3.10+ and authenticated gh."""
import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path)
    parser.add_argument('--verify-only', action='store_true', help='Verify archive contents without writing model/data files')
    parser.add_argument('--offline', action='store_true', help='Use archives already present in assets-dir')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'ARTIFACTS_MANIFEST.json').read_text())
    assets = (args.assets_dir or root / '.release_assets').resolve()
    assets.mkdir(parents=True, exist_ok=True)
    for item in manifest['assets']:
        archive = assets / item['name']
        if not archive.exists():
            if args.offline:
                raise FileNotFoundError(archive)
            subprocess.run(['gh','release','download',manifest['release'],'--repo',manifest['repository'],
                            '--pattern',item['name'],'--dir',str(assets)], check=True)
        if archive.stat().st_size != item['bytes'] or sha(archive) != item['sha256']:
            raise ValueError('Archive verification failed: ' + item['name'])
    opened = {}
    cache = OrderedDict()
    try:
        for i, item in enumerate(manifest['files'], 1):
            target = (root / item['path']).resolve()
            if not target.is_relative_to(root) or target == root:
                raise ValueError('Unsafe output path')
            if not args.verify_only and target.exists():
                if target.is_file() and sha(target) == item['sha256']:
                    continue
                raise ValueError('Refusing to overwrite a different file: ' + item['path'])
            temporary = target.with_name(target.name + '.restore.tmp')
            stream = None
            h = hashlib.sha256(); size = 0
            try:
                if not args.verify_only:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    stream = temporary.open('xb')
                for chunk in item['chunks']:
                    data = cache.get(chunk)
                    if data is None:
                        asset = manifest['chunks'][chunk]
                        if asset not in opened:
                            opened[asset] = zipfile.ZipFile(assets / asset)
                        data = opened[asset].read(chunk)
                        if hashlib.sha256(data).hexdigest() != chunk:
                            raise ValueError('Chunk verification failed')
                        cache[chunk] = data
                        if len(cache) > 128:
                            cache.popitem(last=False)
                    else:
                        cache.move_to_end(chunk)
                    h.update(data); size += len(data)
                    if stream:
                        stream.write(data)
                if h.hexdigest() != item['sha256'] or size != item['bytes']:
                    raise ValueError('Reconstructed file mismatch: ' + item['path'])
                if stream:
                    stream.close(); stream = None
                    temporary.replace(target)
            finally:
                if stream:
                    stream.close()
            if i % 100 == 0 or i == len(manifest['files']):
                print(f'Verified {i}/{len(manifest["files"])} binary files', flush=True)
    finally:
        for archive in opened.values():
            archive.close()
    print('All release assets verified' if args.verify_only else 'All release assets restored')


if __name__ == '__main__':
    main()
