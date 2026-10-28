"""Seal all package files and build a portable ZIP (outside the package itself).

python -B scripts/r8c_package.py
No production input is changed. Existing package ZIPs are never overwritten.
Server result directories should be archived separately after real execution.
"""
import zipfile
from r8c_common import HERE, seal, sha


def main():
    target=HERE.parent/'R8C_MEMORY_RECONCILIATION_SERVER_V1.zip'
    if target.exists(): raise SystemExit('package exists; choose a new version before packaging again')
    seal(HERE)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(HERE.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts: z.write(p,HERE.name+'/'+p.relative_to(HERE).as_posix())
    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None: raise RuntimeError('ZIP CRC error')
    digest=sha(target); target.with_suffix('.zip.sha256').write_text(digest+'  '+target.name+'\n',encoding='utf-8')
    print(str(target)); print(digest)


if __name__=='__main__': main()
