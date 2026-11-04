from context_builder import parse_source_refs
from json_load import safe_load_json
from llm_client import generate_with_retry


def infer_prerequisites(lo_list, model="gemini-2.5-flash-lite", client=None, verbose=True, document_language="sk"):
    if not lo_list:
        return lo_list

    groups = {}
    source_signature_by_id = {}
    for obj in lo_list:
        signature = get_source_signature(obj)
        source_signature_by_id[obj.get("id")] = signature
        groups.setdefault(signature, []).append(obj)

    mapping = {}
    for signature, group in groups.items():
        if len(group) < 2:
            for obj in group:
                mapping[obj.get("id")] = []
            continue
        group_mapping = infer_prerequisites_for_group(
            group,
            signature=signature,
            model=model,
            client=client,
            verbose=verbose,
            document_language=document_language,
        )
        for obj in group:
            mapping[obj.get("id")] = group_mapping.get(obj.get("id"), [])

    for obj in lo_list:
        pid = obj.get("id")
        allowed_signature = source_signature_by_id.get(pid, tuple())
        raw_prereq = mapping.get(pid, [])
        obj["prerekvizity"] = [
            prereq_id
            for prereq_id in raw_prereq
            if prereq_id != pid and source_signature_by_id.get(prereq_id, tuple()) == allowed_signature
        ]
    if verbose:
        print("Prerekvizity boli doplnene.")
    return lo_list


def infer_prerequisites_for_group(group, signature, model="gemini-2.5-flash-lite", client=None, verbose=True, document_language="sk"):
    summary_lines = []
    for obj in group:
        acts = obj.get("odporúčané_aktivity", [])
        if not isinstance(acts, (list, tuple)):
            acts = [acts]
        zadania = obj.get("odporúčané_zadania", [])
        if not isinstance(zadania, (list, tuple)):
            zadania = [zadania]
        summary_lines.append(
            f"id: {obj.get('id')}, vzdelávací_objekt: {obj.get('vzdelávací_objekt')}, "
            f"bloom_level: {obj.get('bloom_level')}, citovane_dokumenty: {list(signature)}, "
            f"odporucane_aktivity: {acts}, odporucane_zadania: {zadania}"
        )
    summary_text = "\n".join(summary_lines)

    if document_language == "en":
        prompt = f"""
You are a teacher. Infer prerequisite relationships between the following learning objectives.
All of these objectives belong to the same document group: {list(signature)}.
Return JSON: an array of objects {{"id": number, "prerekvizity": [list of ids that must come before it]}}.
Do not create cycles.
Use only ids from the list below.
Keep the JSON field name exactly as "prerekvizity".

Learning objectives:
{summary_text}
"""
    else:
        prompt = f"""
Si ucitel. Pre nasledujuci zoznam vzdelavacich cielov navrhni dopln vsetky prerekvizity medzi nimi.
Vsetky tieto ciele patria do rovnakej skupiny dokumentov: {list(signature)}.
Vrat JSON: pole objektov {{"id": cislo, "prerekvizity": [zoznam id, ktore musia byt predtym]}}.
Nedavaj cykly.
Pouzivaj iba id zo zoznamu cielov, ktore su uvedene nizsie.

Zoznam cielov:
{summary_text}
"""
    try:
        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        parsed = safe_load_json(response.text if response else "")
        if isinstance(parsed, dict):
            parsed = [parsed]
    except Exception as e:
        if verbose:
            print(f"Prerekvizity zlyhali pre skupinu {list(signature)}: {e}")
        return {}

    valid_ids = {obj.get("id") for obj in group}
    mapping = {}
    if not isinstance(parsed, list):
        return mapping

    for item in parsed:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id not in valid_ids:
            continue
        raw_prereq = item.get("prerekvizity", [])
        if not isinstance(raw_prereq, list):
            raw_prereq = []
        mapping[item_id] = [
            prereq_id
            for prereq_id in raw_prereq
            if isinstance(prereq_id, int) and prereq_id in valid_ids and prereq_id != item_id
        ]
    return mapping


def get_source_signature(lo):
    source_ids = {
        source_id
        for source_id, _ in parse_source_refs(lo.get("citovane_zdroje", []))
        if source_id
    }
    return tuple(sorted(source_ids))
