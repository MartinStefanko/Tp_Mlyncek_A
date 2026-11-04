import time

from context_builder import format_segment_label, make_source_ref
from json_load import safe_load_json
from llm_client import generate_with_retry


def generate_learning_objects(
    segmenty,
    batch_size=10,
    model="gemini-2.5-flash-lite",
    client=None,
    verbose=True,
    document_language="sk",
):
    vsetky_lo = []
    next_id = 1

    if verbose:
        print("Generujem vzdelávacie ciele.")

    start_total = time.perf_counter()
    batch_num = 1
    for start in range(0, len(segmenty), batch_size):
        batch = segmenty[start:start + batch_size]
        parts = []
        start_batch = time.perf_counter()
        for seg in batch:
            parts.append(f"[{format_segment_label(seg)}]\n{seg.get('text', '')}")
        combined_text = "\n\n".join(parts)

        prompt = build_lo_generation_prompt(combined_text, document_language)

        response = generate_with_retry(prompt, client=client, model=model, verbose=verbose)
        response_text = response.text if response else ""

        try:
            parsed_seg = safe_load_json(response_text)
        except Exception as e:
            if verbose:
                print("Nepodarilo sa parsovať JSON odpoveď:", e)
            continue

        if not parsed_seg:
            continue

        if isinstance(parsed_seg, dict):
            parsed_seg = [parsed_seg]

        end_batch = time.perf_counter()
        if verbose:
            print(f"Čas generovania batchu {batch_num}: {end_batch - start_batch:.2f} s")

        batch_num += 1
        for obj in parsed_seg:
            obj["id"] = next_id
            next_id += 1

        missing = []
        for obj in parsed_seg:
            sources = obj.get("citovane_zdroje")
            if sources is None:
                missing.append(obj)
                continue
            if isinstance(sources, (list, tuple, set)):
                if not any(str(v).strip() for v in sources):
                    missing.append(obj)
            elif not str(sources).strip():
                missing.append(obj)

        if missing:
            parts_with_pages = []
            for seg in batch:
                page = seg.get("page")
                source_id = seg.get("source_id")
                text = seg.get("text", "")
                parts_with_pages.append(f"[zdroj {make_source_ref(source_id, page)}]\n{text}")
            batch_text_with_pages = "\n\n".join(parts_with_pages)

            lo_summary = []
            for obj in missing:
                lo_summary.append(
                    f"id: {obj.get('id')}, "
                    f"vzdelavaci_objekt: {obj.get('vzdelávací_objekt', '')}, "
                    f"bloom_level: {obj.get('bloom_level', '')}, "
                    f"odporucane_zadania: {obj.get('odporúčané_zadania', '')}"
                )
            lo_summary_text = "\n".join(lo_summary)

            prompt_missing = build_missing_sources_prompt(lo_summary_text, batch_text_with_pages, document_language)
            try:
                response_missing = generate_with_retry(prompt_missing, client=client, model=model, verbose=verbose)
                parsed_missing = safe_load_json(response_missing.text if response_missing else "")
                if isinstance(parsed_missing, dict):
                    parsed_missing = [parsed_missing]
            except Exception as e:
                parsed_missing = []
                if verbose:
                    print(f"Doplnenie citovanych zdrojov zlyhalo: {e}")

            src_map = {}
            if isinstance(parsed_missing, list):
                for row in parsed_missing:
                    if not isinstance(row, dict):
                        continue
                    row_id = row.get("id")
                    if row_id is None:
                        continue

                    raw_sources = row.get("citovane_zdroje")
                    if isinstance(raw_sources, (list, tuple, set)):
                        normalized = [str(v).strip() for v in raw_sources if str(v).strip()]
                    elif raw_sources is None:
                        normalized = []
                    else:
                        text = str(raw_sources).strip()
                        if not text:
                            normalized = []
                        elif "," in text:
                            normalized = [p.strip() for p in text.split(",") if p.strip()]
                        else:
                            normalized = [text]

                    if normalized:
                        src_map[row_id] = normalized

            for obj in parsed_seg:
                sources = obj.get("citovane_zdroje")
                has_sources = False
                if isinstance(sources, (list, tuple, set)):
                    has_sources = any(str(v).strip() for v in sources)
                elif sources is not None:
                    has_sources = bool(str(sources).strip())

                if not has_sources:
                    filled = src_map.get(obj.get("id"))
                    if filled:
                        obj["citovane_zdroje"] = filled

        for obj in parsed_seg:
            vsetky_lo.append(obj)

    end_total = time.perf_counter()
    if verbose:
        print(f"Generovanie LO dokončené. Celkový počet LO: {len(vsetky_lo)}")
        print(f"Celkový čas generovania: {end_total - start_total:.2f} s")

    return vsetky_lo


def build_lo_generation_prompt(combined_text, document_language):
    if document_language == "en":
        return f"""
You are a teacher. Extract measurable learning objectives from the following material.

Return ONLY valid JSON: an array of objects with the following fields:
id (unique identifier 1, 2, 3...),
vzdelávací_objekt (main point/objective, at most two words; keep the field name exactly as written),
bloom_level (must be one of these exact Slovak values: Zapamätať si, Pochopiť, Aplikovať, Analyzovať, Hodnotiť, Vytvoriť),
odporúčané_aktivity (short list; keep the field name exactly as written),
odporúčané_zadania (short assignment text with at most one active verb in imperative form; plain text, not a JSON list),
citovane_zdroje (must cite a specific document and page; use format ["D1:1", "D2:5"]).

Important:
- Keep JSON field names exactly as specified.
- Keep bloom_level values in the Slovak controlled vocabulary above.
- Write the content of the learning objective, activities, and assignments in English because the source document is in English.
- If you identify a section that looks like front matter (table of contents, introduction, etc.), ignore it and create no learning objectives for it.

Teaching material:
\"\"\"{combined_text}\"\"\"
"""
    return f"""
Si učiteľ. Na základe nasledujúceho materiálu extrahuj merateľné vzdelávacie ciele.

Output only valid JSON: an array of objects with the following fields:
id (jedinečný identifikátor 1, 2, 3...),
vzdelávací_objekt (hlavný bod/cieľ, najviac dve slová),
bloom_level (jedno z: Zapamätať si, Pochopiť, Aplikovať, Analyzovať, Hodnotiť, Vytvoriť),
odporúčané_aktivity (krátky zoznam),
odporúčané_zadania - v jednej úlohe maximálne jedno aktívne sloveso a slovesá nech sú v imperatíve (krátky popis) (vystup ma byt ako suvisly text ziadne [] a tak),
citovane_zdroje - každý vzdelávací objekt MUSÍ mať citovaný konkrétny dokument a stranu. Toto pole nesmie byť prázdne. Použi formát ["D1:1", "D2:5"] podľa značiek dokumentov v texte.

V prípade, že identifikuješ časť materiálu ktorá má štruktúru začiatku dokumentu (napríklad obsah dokumentu, úvod atď.)
tak túto časť ignoruj a nevytváraj pre ňu žiadne vzdelávacie objekty.

Vyučovací materiál:
\"\"\"{combined_text}\"\"\"
"""


def build_missing_sources_prompt(lo_summary_text, batch_text_with_pages, document_language):
    if document_language == "en":
        return f"""
You are a teacher. Fill the field citovane_zdroje for the following learning objectives where it is missing.
Use only sources visible in the text with labels like [zdroj D1:1].
If you are unsure, return at least the single most relevant source. Use the format ["D1:1", "D2:5"].

Input learning objectives:
{lo_summary_text}

Material text:
\"\"\"{batch_text_with_pages}\"\"\"

Return ONLY valid JSON as an array of objects:
[
    {{"id": 1, "citovane_zdroje": ["D1:12", "D2:13"]}}
]
"""
    return f"""
                Si ucitel. Doplň pole citovane_zdroje pre nasledujuce LO, kde je prazdne. 
                Pouzi iba zdroje, ktore su viditelne v texte so znacenkou [zdroj D1:1].
                Ak si nie si isty, vrat aspon najrelevantnejsi jeden zdroj. Pouzi format ["D1:1", "D2:5"].

                Vstupne LO:
                {lo_summary_text}

                Text materialu:
                \"\"\"{batch_text_with_pages}\"\"\"

                Vrat LEN validny JSON ako pole objektov:
                [
                    {{"id": 1, "citovane_zdroje": ["D1:12", "D2:13"]}}
                ]
                """
