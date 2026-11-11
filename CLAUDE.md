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
- Report je súčasť integrácie, nie iba diagnostika: Laravel ho ukladá celý do generations.report
  (jsonb), z report.learning_objects[] plní tabuľku learning_objects (external_id = id LO
  z reportu), pri otázke ukladá generation_questions.lo_ref (id learning objektu) a z kľúčových
  slov, Bloomovej úrovne a témy (anchor) robí tagy (tags + taggables, source = llm).

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

## Požiadavky na ďalšie kroky (z kroku 10)
- Krok 14: `service/jobs/legacy_output.py` je dočasný – nahradiť skutočným krokom output.

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
- Krok 14: poradie v hodine choice → open → no_answer, dĺžkové limity a skracovanie.
- Krok 14: report použije názvy zo stránky Databáza – `learning_objects[]` namiesto `sections`
  a väzba otázky cez `lo_ref` namiesto `section_id`. Kontrakt a schéma reportu
  (qgen/contract/report.py, exportované schémy) sa upravia v kroku 14; dovtedy sa kód reportu nemení.

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
  predvolené 2 ostali kvôli kontraktu). legacy_output ju len pustí cez nh3. Report bez zmeny.

## Oprava filtra otázok (po kroku 13c)
- `filter_items_variant_b`: každý prah platí samostatne – skóre kvality < QGEN_QUESTION_MIN_SCORE,
  vernosť < QGEN_QUESTION_MIN_FAITHFULNESS_SCORE alebo zodpovedateľnosť <
  QGEN_QUESTION_MIN_ANSWERABILITY_SCORE (aj chýbajúce skóre) → vyradená, dôvod low_score.
  Predtým bolo kvalita OR zodpovedateľnosť (variant B z pôvodného kódu), takže prechádzali otázky
  so skóre 2 (napr. nesprávna vzorová odpoveď) aj so zodpovedateľnosťou 1. Celok pod per_section
  po filtri dostane doplnenie pokrytia ako doteraz. Filter tematických celkov (iba vernosť) bol v poriadku.
- baseline_v5 je nahratá až s touto opravou (commit dff7c00).
