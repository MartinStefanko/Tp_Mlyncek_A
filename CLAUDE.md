# Mlynček A – prerábka na Docker API službu

## Cieľ
Prerobiť Streamlit aplikáciu v `Hlavna_cast/` na samostatnú službu (FastAPI + RQ worker + Redis) v Dockeri,
ktorú volá Laravel backend ECEH. Špecifikácia: `docs/` (hlavný dokument + Prílohy A–D). Pri nejasnosti
platí špecifikácia; ak s ňou niečo nesedí, zastav sa a spýtaj sa, nevymýšľaj.

## Rozhodnutia nad rámec špecifikácie (platia prednostne)
- `job.json` v `/data/jobs/{id}/` je zdroj pravdy o úlohe; Redis je iba fronta a cache priebehu.
  Zápis atomicky (tmp súbor → fsync → rename). Pri štarte worker znova zaradí úlohy queued/running, ktoré nie sú vo fronte.
- Webhook: `X-QGen-Timestamp` a podpis sa počítajú pri KAŽDOM pokuse nanovo; `X-QGen-Delivery` a `occurred_at` ostávajú.
- `llm_error` je kód chyby úlohy (status failed), nie HTTP odpoveď API.
- Opakovania webhookov cez RQ `enqueue_in`, bez samostatného kontajnera scheduler.
- Webhooky (krok 11): samostatná RQ fronta `qgen-webhooks` a kontajner `webhook-worker`
  (rovnaký image, `python -m service.webhook_worker`, `with_scheduler=True`), aby doručenie
  nečakalo na bežiacu úlohu. Odchýlka od spec (4 kontajnery namiesto 3). Najprv zápis
  job.json, potom zaradenie webhooku s record z toho zápisu. job.cancelled posiela iba API
  (DELETE), worker pri zrušení nič. worker_lost z recovery posiela job.failed.
  /health: HTTP 503 iba pri nedostupnom Redise, inak 200; stav workerov iba v tele
  (`status` ok/degraded, `webhook_worker` "disabled" pri vypnutých webhookoch).
- Čistenie HTML knižnicou `nh3`.
- Replay poskytovateľ (krok 12): `QGEN_LLM_PROVIDER/QGEN_EMBED_PROVIDER=replay` iba na vývoj;
  check_config ho neodmieta, ale vždy varuje (bez QGEN_ENV); /v1/info ukazuje poskytovateľov
  a `models.replay`. `QGEN_MAINTENANCE_INTERVAL_S` (predvolene 600) skracuje údržbu pre dev
  (replay compose 15 s), inak by úloha zabitého workera čakala na obnovu až 10 min. Podľa neho
  sa nastaví aj `maintenance_interval` a `worker_ttl` RQ (`rq_worker_options`; pri 600 s
  predvolené hodnoty RQ), lebo zámok plánovača zabitého workera RQ prevezme až pri svojej údržbe.

## Pravidlá
- Python 3.12, pytest. Testy NIKDY nevolajú skutočné LLM ani embedding API – používajú falošný/replay klient.
- Pri presune a refaktore NEMENIŤ logiku pipeline (prompty, prahy, filtrovanie), pokiaľ to krok výslovne nežiada.
- Existujúce slovenské názvy premenných v pôvodných moduloch nechať; nový kód písať s anglickými názvami.
- Výstup `units` nesmie obsahovať žiadne polia navyše (Pydantic `extra="forbid"`).
- Mimo `qgen/llm/` sa nesmie importovať žiadna knižnica výrobcu LLM.
- Kľúče iba z prostredia (`.env`), nikdy v kóde ani v logoch. Logy neobsahujú text dokumentu.
- Na konci každej úlohy: spusti testy, vypíš zhrnutie zmien (súbory, nové funkcie, otvorené body).
- Testy spúšťaj cez venv/Scripts/python.exe -m pytest (globálny Python nemá závislosti).
- LLMError musí prebublať z run_pipeline; worker ju v kroku 10 zmení na failed s error.code llm_error.

- Terminológia (podľa stránky Databáza, Viktor Bejtic): learning object (LO) v ECEH = TEMATICKÝ
  CELOK Mlynčeka (v kóde Section) – skupina otázok s názvom, tvrdením a typom. ECEH ho ukladá do
  tabuľky learning_objects a presúva medzi hodinami (learning_object_unit). Learning object NIE JE
  jedna otázka; otázka/úloha je položka units[].questions[]. Väčšie jednotky dokumentu pre
  split topics = témy (topic). Interné funkcie pôvodných modulov sa zatiaľ nepremenúvajú.
- Do databázy ECEH ide IBA formát units (GET /v1/jobs/{id}/result). Report je interná
  diagnostika kvality (sections, question_meta, quality, warnings, timings); Laravel ho nemusí
  volať ani ukladať. Tvar reportu sa bez dôvodu nemení.

## Cieľ generovania a referenčná nahrávka
- Cieľ generovania: čo najviac otázok, pokrytie všetkých tematických celkov.
  per_section je minimum, nie strop.
- Replay testy (test ekvivalencie, test_cli, test_progress, replay časti workera) stoja na
  referenčnej nahrávke baseline_v5/ (gemini-3.1-flash-lite + gemini-3.5-flash). Meraná bola
  s commitom dff7c00 (metrics.json: git_dirty false; nápovedy z kódu, opravený filter otázok).
  Očakávané hodnoty, modely, nastavenia otázok (metrics.json "questions" → "settings", prah
  duplicít 0.94) aj súbežnosť (run.llm_max_concurrency 8) berie tests/baseline_data.py
  iba z jej súborov, nie z predvolených hodnôt kódu. docker-compose.replay.yml
  a tools/resilience_check.py používajú tiež baseline_v5. V gite je len metrics.json, bez lokálnej
  nahrávky sa testy preskočia. Zmena promptu = nová nahrávka (tools/baseline.py), nie úprava testov.
- baseline/ (gemini-2.5), baseline_v3/ (starý formát otázok) a baseline_v4/ (krok 13, prompty
  s nápovedami) sú iba historické záznamy; žiadny test ich nepoužíva. Všetky štyri sú chránené
  v tools/baseline.py.

## Požiadavky na ďalšie kroky (z kroku 8)
- Krok 9: `delete_job` (DELETE hotovej úlohy aj mazanie po TTL) musí odstrániť aj rezerváciu
  external_ref v Redise, inak by nový POST s rovnakým external_ref vrátil zmazanú úlohu.
- Krok 10 (RQ `JobQueue.contains`): True iba ak úloha čaká vo fronte alebo beží na ŽIVOM workeri.
  Úloha, ktorej worker spadol (ostala v registri bežiacich úloh), sa nepočíta – inak by ju recovery
  nikdy neobnovila.
- Krok 10 (worker): `InvalidTransition` pri zápise completed/failed, keď je úloha medzitým
  cancelled, je normálne ukončenie – zalogovať (info), nie chyba.

## Rozhodnutia z kroku 13 (typy otázok)
- Všetkých 14 promptov (aj LO pipeline) vracia objekt `{"items": [...]}` (`generate_json(items_key="items")`);
  JSON režim zapnutý pri každom volaní. Iný tvar = neplatný JSON (opakovanie, potom LLMError).
- Interný formát otázky = polia `Question` + metadáta (qgen/questions/types.py). Pravidlá typov sú
  iba raz: `qgen.contract.units.type_rule_errors`.
- Kontrola `is_correct` pri choice je v kontrole vernosti (faithfulness, má text zdroja);
  `choices_ok: false` → otázka vyradená (`wrong_choice_marking`) s varovaním.
- Generovanie po QGEN_QUESTION_GENERATION_BATCH_SIZE (3) celkoch; odpoveď odrezaná na limite
  tokenov (OutputTruncatedError) rozdelí batch na polovicu; jeden celok, ktorý sa nezmestí = LLMError.
- Anthropic: temperature sa neposiela vôbec (ani extra_body).
- Doplňujúce kolá: QGEN_QUESTION_EXTRA_ROUNDS=1.
- Prah duplicít QGEN_QUESTION_DEDUP_THRESHOLD=0.94 (baseline_v4 nahratá s 0.95; dvojice
  0.947–0.949 boli tiež preformulované otázky). baseline_v5 je nahratá s 0.94.

## Požiadavky na ďalšie kroky (z kroku 13)
- Krok 15–16: poradie v hodine choice → open → no_answer (patrí k rozdeleniu do hodín).

## Rozhodnutia z kroku 13b (paralelné volania)
- Súbežné batche iba cez `qgen.llm.parallel.run_batches` (QGEN_LLM_MAX_CONCURRENCY; 1 = bez vlákien).
  Task iba volá LLM; stav pipeline, id a progress_callback mení len hlavné vlákno v poradí batchov.
- Otázky: kolá generovania idú po fázach cez všetky batche; `renumber_generation` vráti id a poradie
  sekvenčného behu (batch po batchi). Hraničný prípad: otázka s lo_id celku z iného batchu by
  v dedup/zozname existujúcich otázok mohla mať iné poradie – v baseline_v4 ani v5 sa nevyskytuje.

## Požiadavky na ďalšie kroky (z kroku 13b)
- HOTOVÉ v kroku 13c: nápoveda obsahuje iba odkaz na názov dokumentu a strany, vytvára ju kód
  zo zdrojov otázky, nie LLM (požiadavka tímu; ruší pravidlo, že nápovedy nesmú odkazovať na strany).

## Rozhodnutia z kroku 13c (nápovedy)
- LLM nápovedy nepíše ani nehodnotí: žiadny prompt (generovanie, doplňujúce kolo, pokrytie,
  skóre kvality vrátane {items_text}) o nich nič neobsahuje; "hints" z odpovede LLM sa ignoruje.
  Zrušené: mená „Ľahká/Ťažká nápoveda“, varovania hint_page_reference (aj kontrola zadania)
  a hints_missing. Relevancia k celku (embedding) nápovedu nepoužíva.
- Vytvára ju `qgen/questions/hints.py` (`build_source_hints`) na konci pipeline otázok (krok 5,
  po attach_source_names). Strany: vlastný `source` otázky (iba strany, ktoré v dokumente sú,
  a dokument s názvom); ak nie je platný žiadny, `citovane_zdroje` jej tematického celku;
  inak `hints: []`. Názov dokumentu = `source_name` segmentov = InputFile.name (nie D1.pdf).
- Formát: jedna nápoveda `{"name": "Zdroj", "url": "<p>eko.pdf, strana 14</p>"}`; rozsah
  „strany 14 – 15“, nesúvislé „strany 14, 16“ (mix „strany 3 – 5, 9“), viac dokumentov v poradí
  nahrania oddelené „; “ („eko.pdf, strana 4; uvod.pdf, strany 2 – 3“). EN dokument: „Source“,
  „page 14“ / „pages 14 – 15“. Text je HTML-escapovaný v `<p>`. Šablóny QGEN_HINT_NAME_SK/EN,
  QGEN_HINT_PAGE_SK/EN (1 strana), QGEN_HINT_PAGES_SK/EN ({file}, {pages} povinné).
- Limity ECEH: name ≤ 50 (skrátené na koniec slova), url ≤ 2000 – odoberajú sa posledné behy
  strán dokumentu s najviac behmi („, …“), potom celé dokumenty („; …“), nakoniec znaky.
- options.questions.hints: 0 = bez nápovedy, ≥ 1 = jedna nápoveda so zdrojom (rozsah 0–2 a
  predvolené 2 ostali kvôli kontraktu). Výstup (qgen/output) ju len pustí cez nh3. Report bez zmeny.

## Oprava filtra otázok (po kroku 13c)
- `filter_items_variant_b`: každý prah platí samostatne – skóre kvality < QGEN_QUESTION_MIN_SCORE,
  vernosť < QGEN_QUESTION_MIN_FAITHFULNESS_SCORE alebo zodpovedateľnosť <
  QGEN_QUESTION_MIN_ANSWERABILITY_SCORE (aj chýbajúce skóre) → vyradená, dôvod low_score.
  Predtým bolo kvalita OR zodpovedateľnosť (variant B z pôvodného kódu), takže prechádzali otázky
  so skóre 2 (napr. nesprávna vzorová odpoveď) aj so zodpovedateľnosťou 1. Celok pod per_section
  po filtri dostane doplnenie pokrytia ako doteraz. Filter tematických celkov (iba vernosť) bol v poriadku.
- baseline_v5 je nahratá až s touto opravou (commit dff7c00).

## Rozhodnutia z kroku 14 (výstup units)
- Krok output je `qgen/output/` (units.py, limits.py, sanitize.py, report.py, check.py,
  `build_output` v `__init__.py`); neimportuje nič zo `service/` (test_import_guard).
  `service/jobs/legacy_output.py` zmazaný. Zatiaľ jeden unit, title = názvy PDF, description =
  zoznam tematických celkov (rozdelenie do hodín je krok 15–16).
- Dĺžkové limity ECEH v znakoch (`len`): otázka s príliš dlhým name, description alebo choices[i]
  dostane JEDNO volanie LLM (rola questions, JSON `{"items": [{"field", "text"}]}`, jazyk dokumentu,
  iba príliš dlhé polia s limitmi). Text sa nahradí iba v poli s rovnakým field → poradie choices
  a is_correct sa nemenia. Volania cez `run_batches`; progress_callback("output", …) po každom.
- Nápovedy sa LLM neposielajú (robí ich kód v limitoch). name stále dlhší → cut_at_word;
  description / choice / hint stále dlhší → otázka vyradená, varovanie `field_too_long`
  (pole, dĺžka, limit, bez textu). Zlyhané volanie (LLMError) → `shortening_failed`, úloha beží ďalej.
- hints.url cez nh3 (p, br, strong, em, ul, ol, li, code, pre, sub, sup; bez atribútov; obsah
  script/style zahodený). Nápoveda bez textu po čistení (alebo bez name) → vyhodená, `invalid_hint`,
  otázka ostáva. Varovania jednej otázky idú spolu, v poradí otázok, za diagnostikou pipeline.
- Worker pred uložením volá `check_output`: Pydantic + JSON schéma (units aj report) + každá otázka
  má práve jeden question_meta a jej section_id je v sections. Zlyhanie → internal_error
  (stage output, error_type OutputCheckError); v logu iba cesty a druhy chýb, nie hodnoty.
  `jsonschema` je preto v requirements-service.txt.
- Otvorené: bod 1 špecifikácie (limity polí priamo v prompte generovania) = zmena promptu, teda
  nová nahrávka; zatiaľ nie.

## Rozhodnutia z kroku 15 a 16 (rozdelenie do hodín)
- `qgen/split/` (sections, order, topics, syllabus, lessons) sú čisté funkcie bez LLM a embeddingov
  (test_import_guard). Volania (témy, plán sylabu, názvy hodín) sú v `qgen/output/split.py`
  (`run_split`), prompty v `qgen/output/prompts.py`, rola LLM `split`.
- Krok 16, rozhodnutie Martina (variant A): počet hodín sa NEURČUJE zo súčtu časov otázok
  (otázky sú zásobník; eko.pdf 560 min by dalo 12 hodín). Kapacita hodiny neexistuje.
  `lessons` vyžaduje `lesson_count`; `lesson_minutes` ostáva v kontrakte (Príloha A), kontroluje
  sa rozsah, ale nemá vplyv a nedopĺňa sa. `QGEN_DEFAULT_LESSON_MINUTES` zrušené.
- Poradie: Kahn, zhoda = (číslo dokumentu, prvá strana, id) ako lo_page_sort_key, D10 za D9.
  Cyklus: v cyklickej komponente celok s najskoršou stranou stratí prerekvizity z komponenty.
- Témy: vektory tém a celkov z `analyze_topic_coverage` (parameter `vectors`) →
  `LOPipelineResult.embeddings` (nové id, neserializuje sa). Žiadne nové volanie. Téma celku
  = najpodobnejšia (spec c; Section.topic a question_meta.topic v reporte).
- Úseky tém: Viterbi, Σ podobnosť − 0,1 × počet zmien témy (`TOPIC_SWITCH_PENALTY`).
- lessons: DP Σ T² (= čo najrovnomernejšie), rez vnútri úseku témy +0,02 · priemer²; zhoda →
  skorší rez. lesson_count > počet celkov → zníži (`lesson_count_reduced`).
- topics: hodina = úsek témy; úsek < 50 % mediánu sa spojí s menším susedom, úsek > 3 × medián
  sa rozdelí na round(T / medián) častí. Bez tém → 1 hodina + `no_document_topics`.
  eko.pdf: 8 hodín; lessons s lesson_count 3: celky 1–15, 16–32, 33–42.
- `lessons_unbalanced`: hodina > 50 % od priemeru (lessons, topics). Zrušené: lesson_time_deviation,
  lesson_over_capacity, section_over_capacity, lesson_count_increased.
- Sylabus (rozhodnutie Martina, režim instructions nebude): `split.syllabus` (zoznam) ALEBO
  `split.syllabus_text` (voľný text ≤ 4000), práve jedno. Text LLM iba prevedie na interný
  `SyllabusPlan` (hodiny so stranami ALEBO section_ids, skipped_section_ids, alebo iba
  lesson_count → lessons, alebo iba lesson_minutes → topics + `syllabus_lesson_minutes_ignored`).
  Text je v prompte v `<<< >>>` s pravidlom ignorovať pokyny. Neplatný plán → jedno opakovanie
  s chybami (cesty a druhy), potom záloha topics + `syllabus_plan_invalid`; LLMError tiež.
  Neznáme id a duplicity = varovania. Strany sylabu s D-indexom nad počet súborov → 400.
  Podobnosť celok × hodina sylabu: vektory celkov z pipeline + 1 embedding hodín.
  Doladenie: sections_text v prompte plánu má stĺpec prerekvizít a pravidlo, že celok nesmie byť
  skôr ako jeho prerekvizita. Poistka IBA pre plán z textu: taký celok sa presunie do hodiny
  najneskoršej prerekvizity (v poradí a, takže ťahá závislé celky), `prerequisite_moved`; zoznam
  od učiteľa sa nemení (ostáva `prerequisite_order`).
- Názov a popis hodiny (bod f): LLM pre všetky hodiny naraz po `QGEN_SPLIT_TITLE_BATCH_SIZE`
  (10), AJ v režime single (rozhodnutie Martina: nová hodina v ECEH by sa inak volala podľa PDF).
  Pri každej hodine riadok „Témy:“ (single: všetky témy dokumentu) a pravidlo, že názov a popis
  musia vystihnúť celý obsah hodiny, nielen začiatok (single na eko.pdf vystihoval iba prvé celky).
  Sylabus: názov učiteľa/plánu, LLM iba pre chýbajúci popis. Text cez nh3 bez značiek,
  cut_at_word 255 / 4000. Zlyhanie → názvy tém, inak „prvý celok a ďalšie“, popis = zoznam
  celkov, `lesson_titles_failed`. Úloha kvôli splitu nikdy nezlyhá.
- Výstup: stage split → output. Otázky v hodine: poradie celkov, choice → open → no_answer.
  Otázky vynechaných celkov nie sú v units ani v question_meta (celky ostávajú v sections).
  Hodina bez prijatej otázky sa vynechá (`empty_lesson`); žiadna hodina → 1 prázdny unit +
  `no_questions` (posledné varovanie). Varovania: diagnostika pipeline, split, otázky.
  `timings.split`.
- Nahrávky: pipeline sa nemenila, baseline_v5 ostáva referenciou. Split má krátke nadstavbové
  nahrávky (tools/baseline.py `--base baseline_v5 --split JSON`): baseline_v5_single, _topics,
  _lessons3, _syllabus_text (chránené). Replay = `ReplayClient([v5, nadstavba])`,
  test `tests/output/test_split_replay.py` (bez nahrávky sa preskočí). Worker replay test a
  test_cli --split odpovedajú na názvy hodín falošným klientom (`TitlesLLM`).
- Windows: `atomic_write_bytes` opakuje `os.replace` pri PermissionError (antivírus drží súbor),
  najviac 10×; na Linuxe nie.
- Kódy varovaní: prerequisite_cycle, prerequisite_order, prerequisite_moved, lesson_count_reduced, lessons_unbalanced,
  no_document_topics, empty_lesson, no_questions, syllabus_unknown_section,
  syllabus_duplicate_section, syllabus_skipped_sections, syllabus_fallback, syllabus_plan_invalid,
  syllabus_lesson_minutes_ignored, lesson_titles_failed.
