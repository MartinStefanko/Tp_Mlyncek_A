import json
import re


def safe_load_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'(\[.*\]|\{.*\})', text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    raise
