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
- Opakovania webhookov cez RQ `enqueue_in` (worker s `--with-scheduler`), bez samostatného kontajnera scheduler.
- Čistenie HTML knižnicou `nh3`.

## Pravidlá
- Python 3.12, pytest. Testy NIKDY nevolajú skutočné LLM ani embedding API – používajú falošný/replay klient.
- Pri presune a refaktore NEMENIŤ logiku pipeline (prompty, prahy, filtrovanie), pokiaľ to krok výslovne nežiada.
- Existujúce slovenské názvy premenných v pôvodných moduloch nechať; nový kód písať s anglickými názvami.
- Výstup `units` nesmie obsahovať žiadne polia navyše (Pydantic `extra="forbid"`).
- Mimo `qgen/llm/` sa nesmie importovať žiadna knižnica výrobcu LLM.
- Kľúče iba z prostredia (`.env`), nikdy v kóde ani v logoch. Logy neobsahujú text dokumentu.
- Na konci každej úlohy: spusti testy, vypíš zhrnutie zmien (súbory, nové funkcie, otvorené body).

- Terminológia: LO z pôvodného kódu = tematický celok (v kóde Section). Learning object = jedna
  otázka alebo úloha vo výstupe (význam ECEH). Väčšie jednotky dokumentu pre split topics = témy (topic).
  Interné funkcie pôvodných modulov sa zatiaľ nepremenúvajú.