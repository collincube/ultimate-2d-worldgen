"""Check public snapshot hygiene using only the standard library; skip Git metadata."""
from pathlib import Path
import os
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '.local', '.venv', 'venv', '__pycache__'}
TEXT = {'.py', '.md', '.json', '.html', '.txt', '.toml', '.yaml', '.yml'}
PATTERNS = {
    'machine-specific path': re.compile(r'/(?:Users|home|mnt)/[\w.-]+/'),
    'private artifact link': re.compile(r'https://claude[.]ai/code/artifact/'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'credential-like token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9_-]{32,})\b'),
}


def main():
    findings = []
    count = 0
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        for name in sorted(names):
            path = Path(directory) / name
            rel = path.relative_to(ROOT)
            count += 1
            if path.is_symlink():
                findings.append(f'{rel}: symlink; review target before copying')
                continue
            if name in {'.DS_Store', 'Thumbs.db', 'Desktop.ini'} or name.startswith('._'):
                findings.append(f'{rel}: OS metadata')
            if name == '.env' or (name.startswith('.env.') and name != '.env.example'):
                findings.append(f'{rel}: local environment file')
            if rel.parts[:2] == ('.github', 'workflows'):
                findings.append(f'{rel}: hosted workflow')
            if path.suffix not in TEXT:
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                findings.append(f'{rel}: text is not UTF-8')
                continue
            for line, text in enumerate(content.splitlines(), 1):
                for label, pattern in PATTERNS.items():
                    if pattern.search(text):
                        findings.append(f'{rel}:{line}: {label}')
    for finding in findings:
        print(finding)
    print(f'Checked {count} files; {len(findings)} findings. Git metadata excluded.')
    print('This heuristic scan does not certify secret absence or inspect history.')
    return bool(findings)


if __name__ == '__main__':
    raise SystemExit(main())
