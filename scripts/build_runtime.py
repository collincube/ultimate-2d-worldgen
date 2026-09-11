"""Rebuild runtime artifacts with portable source references; run from any directory."""
from pathlib import Path
import json
import sys

BUNDLE = Path(__file__).resolve().parents[1] / 'tilegen_full_stack_bundle'
sys.path.insert(0, str(BUNDLE))
import tilegen_lowend_runtime as runtime


def main():
    output = BUNDLE / 'tilegen_runtime_output'
    summary = runtime.build_runtime(BUNDLE / 'tilegen_output', output)
    # The legacy builder uses absolute paths while rendering. Serialize paths
    # relative to the bundle root so the exported catalog survives relocation.
    catalog = output / 'module_library.json'
    modules = json.loads(catalog.read_text(encoding='utf-8'))
    for module in modules:
        for key in ('source_render', 'source_layout_json'):
            if module[key]:
                module[key] = Path(module[key]).relative_to(BUNDLE).as_posix()
    catalog.write_text(json.dumps(modules, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
