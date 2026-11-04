import json
import re


def safe_load_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        m = re.search(r'(\[.*\]|\{.*\})', text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # Step 6: re-raise the original JSONDecodeError. Before, a bare `raise` stood outside
        # the except block and raised RuntimeError("No active exception to reraise").
        raise error
