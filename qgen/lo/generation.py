import time

from qgen.context_builder import format_segment_label, make_source_ref
from qgen.config import Settings, setting_or_default
from qgen.llm.parallel import run_batches
from qgen.llm.retry import generate_json
from qgen.progress import STAGE_LEARNING_OBJECTS, batch_count, counting_progress, report_progress


def generate_learning_objects(
    segmenty,
    batch_size=None,
    model=None,
    llm=None,
    verbose=True,
    document_language="sk",
    progress_callback=None,
    *,
    settings: Settings,
):
    batch_size = setting_or_default(batch_size, "lo_batch_size", settings)
    model = setting_or_default(model, "llm_model_lo", settings)
    total_batches = batch_count(len(segmenty), batch_size)
    vsetky_lo = []
    next_id = 1

    if verbose:
        print("Generujem vzdelávacie ciele.")

    start_total = time.perf_counter()
    davky = [segmenty[start:start + batch_size] for start in range(0, len(segmenty), batch_size)]

    # Step 13b, phase A: the generation calls of all batches at the same time (at most
    # QGEN_LLM_MAX_CONCURRENCY). The ids are given below in the order of the batches.
    def generate_batch(batch):
        parts = []
        start_batch = time.perf_counter()
        for seg in batch:
            parts.append(f"[{format_segment_label(seg)}]\n{seg.get('text', '')}")
        combined_text = "\n\n".join(parts)

        prompt = build_lo_generation_prompt(combined_text, document_language)

        # Step 6: invalid JSON no longer skips the batch (`except Exception: continue`);
        # generate_json repeats the call once and then raises LLMError. items_key: the answer
        # must be {"items": [...]}; any other shape is invalid JSON too, never an empty batch.
        parsed = generate_json(
            llm, prompt, role="lo", model=model, verbose=verbose, settings=settings, items_key="items"
        )
        return parsed, time.perf_counter() - start_batch

    report_progress(progress_callback, STAGE_LEARNING_OBJECTS, 0, total_batches)
    vysledky = run_batches(
        [lambda batch=batch: generate_batch(batch) for batch in davky],
        settings=settings,
        on_done=counting_progress(progress_callback, STAGE_LEARNING_OBJECTS, total_batches),
    )

    batch_num = 1
    spracovane = []  # (batch, parsed_seg, missing) in the order of the batches
    for batch, (parsed_seg, trvanie) in zip(davky, vysledky):
        if not parsed_seg:
            continue

        if verbose:
            print(f"Čas generovania batchu {batch_num}: {trvanie:.2f} s")

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
        spracovane.append((batch, parsed_seg, missing))

    # Phase B: the calls for missing sources; their prompts list the LO ids, so they start
    # only after the ids are given.
    def fill_missing_sources(batch, missing):
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
        # Step 6: a failed call no longer means "no sources filled" (parsed_missing = []);
        # LLMError propagates.
        return generate_json(
            llm, prompt_missing, role="lo", model=model, verbose=verbose, settings=settings, items_key="items"
        )

    s_chybajucimi = [entry for entry in spracovane if entry[2]]
    doplnene = run_batches(
        [lambda batch=batch, missing=missing: fill_missing_sources(batch, missing)
         for batch, _, missing in s_chybajucimi],
        settings=settings,
    )
    doplnene_podla_batchu = {id(entry[1]): parsed for entry, parsed in zip(s_chybajucimi, doplnene)}

    for batch, parsed_seg, missing in spracovane:
        if missing:
            parsed_missing = doplnene_podla_batchu[id(parsed_seg)]

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

    report_progress(progress_callback, STAGE_LEARNING_OBJECTS, total_batches, total_batches)
    end_total = time.perf_counter()
    if verbose:
        print(f"Generovanie LO dokončené. Celkový počet LO: {len(vsetky_lo)}")
        print(f"Celkový čas generovania: {end_total - start_total:.2f} s")

    return vsetky_lo


def build_lo_generation_prompt(combined_text, document_language):
    if document_language == "en":
        return f"""
You are a teacher. Extract measurable learning objectives from the following material.
Extract ALL separately teachable parts of the material in this batch (definitions, formulas,
procedures, examples, relationships), not a representative selection. Each such part is a
separate learning objective.

Return ONLY a valid JSON object {{"items": [...]}} where "items" is an array of objects with the following fields:
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
Z materiálu vytiahni VŠETKY samostatne vyučiteľné časti učiva (definície, vzorce, postupy,
príklady, súvislosti), nie reprezentatívny výber. Každá takáto časť je samostatný vzdelávací objekt.

Output only a valid JSON object {{"items": [...]}} where "items" is an array of objects with the following fields:
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

Return ONLY a valid JSON object {{"items": [...]}} where "items" is an array of objects:
{{"items": [
    {{"id": 1, "citovane_zdroje": ["D1:12", "D2:13"]}}
]}}
"""
    return f"""
                Si ucitel. Doplň pole citovane_zdroje pre nasledujuce LO, kde je prazdne. 
                Pouzi iba zdroje, ktore su viditelne v texte so znacenkou [zdroj D1:1].
                Ak si nie si isty, vrat aspon najrelevantnejsi jeden zdroj. Pouzi format ["D1:1", "D2:5"].

                Vstupne LO:
                {lo_summary_text}

                Text materialu:
                \"\"\"{batch_text_with_pages}\"\"\"

                Vrat LEN validny JSON objekt {{"items": [...]}}, kde "items" je pole objektov:
                {{"items": [
                    {{"id": 1, "citovane_zdroje": ["D1:12", "D2:13"]}}
                ]}}
                """
