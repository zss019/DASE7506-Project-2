import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];m=json.loads((root/'MANIFEST.json').read_text())
for name,h in m['files'].items():
 p=root/name
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise SystemExit('Hash mismatch: '+name)
print('Verified',len(m['files']),'release files; checkpoint bytes:',m['checkpoint_bytes'])
