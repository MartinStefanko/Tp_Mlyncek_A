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

- Terminológia: LO z pôvodného kódu = tematický celok (v kóde Section). Learning object = jedna
  otázka alebo úloha vo výstupe (význam ECEH). Väčšie jednotky dokumentu pre split topics = témy (topic).
  Interné funkcie pôvodných modulov sa zatiaľ nepremenúvajú.

## Cieľ generovania a referenčná nahrávka
- Cieľ generovania: čo najviac otázok, pokrytie všetkých tematických celkov.
  per_section je minimum, nie strop.
- Replay testy (test ekvivalencie, test_cli, test_progress) stoja na referenčnej nahrávke
  baseline_v3/ (gemini-3.1-flash-lite + gemini-3.5-flash, commit 42a7cd1). Očakávané hodnoty
  a modely (tests/baseline_data.py) sa berú iba z jej súborov; v gite je len metrics.json,
  bez lokálnej nahrávky sa testy preskočia. Zmena promptu = nová nahrávka (tools/baseline.py),
  nie úprava testov.
- baseline/ (gemini-2.5) je iba historický záznam; žiadny test ho nepoužíva.

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

## Požiadavky na ďalšie kroky (z kroku 11)
- Krok 13: nápovedy nesmú odkazovať na strany ani časti dokumentu („pozrite si stranu 14“) –
  študent v ECEH PDF nemá. Nápoveda má navádzať obsahom.
