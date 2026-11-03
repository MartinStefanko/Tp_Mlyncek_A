"""Generate JSON Schemas of the contract into schemas/.

    venv/Scripts/python.exe tools/export_schemas.py           # write files
    venv/Scripts/python.exe tools/export_schemas.py --check   # exit 1 if files are outdated
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from qgen.contract.schemas import SCHEMAS, build_schema  # noqa: E402

SCHEMA_DIR = REPO_ROOT / "schemas"


def render(schema: dict) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="Only check that the files are up to date.")
    parser.add_argument("--out", type=Path, default=SCHEMA_DIR)
    args = parser.parse_args(argv)

    outdated = []
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (file_name, _model, _description) in SCHEMAS.items():
        text = render(build_schema(name))
        path = args.out / file_name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                outdated.append(file_name)
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
            print(f"Wrote {path}")

    if outdated:
        print("Outdated schemas (run tools/export_schemas.py): " + ", ".join(outdated))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
