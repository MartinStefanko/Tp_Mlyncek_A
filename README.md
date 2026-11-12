# QGen – služba na generovanie otázok z PDF (Mlynček A)

Služba z jedného alebo viacerých PDF dokumentov vytvorí otázky a úlohy pre výučbu
a rozdelí ich do vyučovacích hodín. Volá ju backend ECEH (Laravel) cez HTTP API.
Pôvodne to bola Streamlit aplikácia; teraz je to samostatná služba v Dockeri.

- **Nasadenie na server:** [docs/deployment.md](docs/deployment.md) (pre správcu servera).
- **Integrácia do Laravelu:** [docs/integration.md](docs/integration.md) (pre backend ECEH).
- **Špecifikácia:** `docs/` (hlavný dokument a Prílohy A–D).

## Obsah

1. [Čo služba robí](#čo-služba-robí)
2. [Rýchly štart](#rýchly-štart)
3. [Režimy rozdelenia do hodín](#režimy-rozdelenia-do-hodín)
4. [Príklady volaní (curl, PowerShell)](#príklady-volaní)
5. [Webhooky](#webhooky)
6. [Chyby](#chyby)
7. [Replay, smoke test a overenie odolnosti](#replay-smoke-test-a-overenie-odolnosti)
8. [Nahrávky (baseline)](#nahrávky-baseline)
9. [Vývoj a testy](#vývoj-a-testy)
10. [Mock API](#mock-api)

## Čo služba robí

Jedno volanie `POST /v1/jobs` (PDF + nastavenia) spustí všetky kroky:

| Krok (`stage`) | Čo robí |
|---|---|
| `extract` | text z PDF (pypdf), voliteľne OCR obrázkov (EasyOCR) |
| `language` | jazyk dokumentu (sk / en) |
| `learning_objects`, `lo_evaluation` | tematické celky (menšie časti učiva) a ich kontrola voči textu |
| `questions`, `q_evaluation` | otázky typov `choice`, `open`, `no_answer`, hodnotenie kvality, vernosti a zodpovedateľnosti, filter |
| `split` | rozdelenie do hodín (poradie podľa prerekvizít, témy dokumentu, sylabus) |
| `output` | formát `units` pre ECEH, dĺžkové limity, čistenie HTML, report |

Laravel potom iba číta stav (`GET /v1/jobs/{id}`, webhooky) a výsledok
(`GET /v1/jobs/{id}/result`). Spracovanie nezávisí od spojenia: úloha dobehne aj po
zatvorení stránky a výsledok je dostupný 7 dní.

Štyri kontajnery (`docker-compose.yml`):

```
Laravel ──HTTP──► api ──► redis (fronta) ◄── worker ──► LLM / embedding API (Gemini, …)
   ▲                 │                         │
   └── webhooky ── webhook-worker ◄─────────────┘   /data (volume): job.json, PDF, výsledky
```

- `api` – FastAPI, kontrola vstupu, API kľúč, stav a výsledky.
- `worker` – RQ worker, celá pipeline; obnoví nedokončené úlohy z `job.json`.
- `webhook-worker` – doručuje webhooky, aby nečakali na bežiacu úlohu.
- `redis` – fronta a priebeh. Zdroj pravdy o úlohe je `job.json` vo volume `/data`.

LLM a embeddingy sa volajú cez API zvoleného poskytovateľa (predvolene Gemini). Poskytovateľ
sa mení iba premennými prostredia (`QGEN_LLM_PROVIDER`, `QGEN_EMBED_PROVIDER`); podporované sú
Gemini, OpenAI-kompatibilné servery (OpenAI, Mistral, vLLM, Ollama …) a Anthropic.

## Rýchly štart

Potrebujete Docker s Compose 2.24+. Všetky príkazy sa spúšťajú z koreňa projektu.

### a) Bez kľúča a bez nákladov (replay)

Odpovede LLM sa čítajú z lokálnej nahrávky `baseline_v5/` (funguje iba s `eko.pdf`):

```bash
cp .env.example .env
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f docker-compose.replay.yml up --build
```

API kľúč je `replay-dev-key`. Najjednoduchšie je spustiť smoke test, ktorý pošle úlohu sám
([smoke test](#smoke-test)).

### b) So skutočným Gemini (lokálne)

```bash
cp .env.example .env      # vyplňte GEMINI_API_KEY, voliteľne QGEN_API_KEYS
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

- API: `http://localhost:8000`, dokumentácia OpenAPI: `http://localhost:8000/docs`.
- `docker-compose.dev.yml` zverejní port, nastaví `QGEN_ENV=development` (chýbajúci kľúč je
  iba varovanie) a vytvorí lokálnu sieť namiesto siete ECEH.
- Rýchlejší build bez OCR: `QGEN_INSTALL_OCR=false` v `.env`.
- Logy: `docker compose logs -f worker`. Zastavenie: `docker compose down` (s `-v` zmaže aj dáta).

Spracovanie `eko.pdf` (69 strán) trvá asi 3–5 minút.

### c) Produkcia

`docker compose up -d --build` – postup, premenné, siete a zálohy sú v
[docs/deployment.md](docs/deployment.md).

## Režimy rozdelenia do hodín

Nastavenie `options.split`. Jeden unit vo výsledku = jedna vyučovacia hodina. Počet hodín sa
neurčuje zo súčtu časov otázok (otázky sú zásobník pre učiteľa). Vo všetkých režimoch platí:
žiadna prázdna hodina a tematický celok nie je skôr ako jeho prerekvizita.

| Režim | `split` | Výsledok | Súvisiace varovania |
|---|---|---|---|
| `single` (predvolený) | `{"mode": "single"}` | jedna hodina, názov a popis od LLM | – |
| `lessons` | `{"mode": "lessons", "lesson_count": 3}` | presne N hodín (1–100), čo najrovnomernejšie, rezy radšej na hraniciach tém | `lesson_count_reduced`, `lessons_unbalanced` |
| `topics` | `{"mode": "topics"}` | hodina = téma dokumentu; malé témy spojí, veľké rozdelí | `no_document_topics`, `lessons_unbalanced` |
| `syllabus` + zoznam | `{"mode": "syllabus", "syllabus": [{"title": "Úrokovanie", "pages": "D1:1-20"}, {"title": "Anuity", "pages": "D1:21-40"}]}` | hodiny učiteľa: `title`, voliteľne `description` a `pages` (`D1` = prvý súbor); celky mimo strán idú k najpodobnejšej hodine | `syllabus_unmentioned_sections`, `syllabus_unknown_section`, `syllabus_duplicate_section`, `prerequisite_order` |
| `syllabus` + text | `{"mode": "syllabus", "syllabus_text": "3 hodiny: najprv úrokovanie, potom anuity, perpetuity vynechaj"}` | LLM prevedie voľný text (≤ 4000 znakov) na zoznam hodín; názov z plánu ostáva, popis vždy napíše LLM podľa celej hodiny; nespomenuté celky idú k najpodobnejšej hodine | `syllabus_unmentioned_sections`, `syllabus_skipped_sections`, `syllabus_plan_invalid`, `syllabus_fallback`, `syllabus_lesson_minutes_ignored`, `prerequisite_moved` |

- `syllabus` vyžaduje práve jedno z `syllabus` a `syllabus_text`.
- Ak sa sylabus nedá použiť, služba rozdelí hodiny podľa tém (`topics`) a pridá varovanie;
  úloha kvôli rozdeleniu nikdy nezlyhá.
- `lesson_minutes` sa prijme vo všetkých režimoch, ale nemá vplyv.
- Názvy a popisy hodín píše LLM. V režime `syllabus` ostáva názov hodiny; pri zozname od učiteľa
  aj jeho popis (LLM píše iba chýbajúci), pri `syllabus_text` píše popis vždy LLM. Ak volanie
  zlyhá, použijú sa názvy tém (`lesson_titles_failed`).

Ďalšie nastavenia (`language`, `extraction.ocr`, `questions.*`) sú v
[docs/integration.md](docs/integration.md#3-formát-options) a v `GET /v1/schema/options`.

## Príklady volaní

V príkladoch je adresa `http://localhost:8000` a kľúč `<kluc>` (`QGEN_API_KEYS`; pri replay
`replay-dev-key`). Pole súborov sa môže volať `files` alebo `files[]`.

### curl (bash, Git Bash, Linux, macOS)

```bash
# Vytvorenie úlohy -> 202 a job_id
curl -X POST http://localhost:8000/v1/jobs -H "X-API-Key: <kluc>" \
  -F "files=@tests/fixtures/eko.pdf" \
  -F 'options={"external_ref": "test-1", "split": {"mode": "lessons", "lesson_count": 3}}'

# Rovnaký request znova -> 200, rovnaké job_id, "replayed": true
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>                    # stav
curl -H "X-API-Key: <kluc>" "http://localhost:8000/v1/jobs?external_ref=test-1"       # hľadanie
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/result             # units
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/report             # report
curl -X DELETE -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>          # zrušenie / zmazanie
curl -X POST -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/retry      # po failed, do 24 h
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/schema/lesson                    # JSON schéma units
curl http://localhost:8000/health                                                     # bez kľúča
```

### PowerShell (Windows)

V PowerShelli 5.1 je `curl` alias na `Invoke-WebRequest` – volajte **`curl.exe`**.
Úvodzovky v JSON sú najväčší problém: PowerShell 5.1 ich pri odovzdaní programu odstráni.
Najspoľahlivejšie je dať `options` do súboru a poslať ho cez `<`:

```powershell
# options.json (UTF-8):  {"external_ref": "test-1", "split": {"mode": "syllabus", "syllabus_text": "3 hodiny: najprv úrokovanie, potom anuity"}}
'{"external_ref": "test-1", "split": {"mode": "topics"}}' | Set-Content -Encoding utf8 options.json

curl.exe -X POST http://localhost:8000/v1/jobs -H "X-API-Key: <kluc>" `
  -F "files=@tests/fixtures/eko.pdf" -F "options=<options.json"
```

`-F "options=<options.json"` pošle obsah súboru ako text poľa (`@` by ho poslal ako súbor).
Bez súboru treba úvodzovky v JSON zdvojiť so spätnou lomkou:

```powershell
curl.exe -X POST http://localhost:8000/v1/jobs -H "X-API-Key: <kluc>" `
  -F "files=@tests/fixtures/eko.pdf" -F "options={\`"external_ref\`": \`"test-2\`"}"
```

GET a DELETE sú jednoduchšie cez `Invoke-RestMethod` (vráti objekt):

```powershell
$h = @{ "X-API-Key" = "<kluc>" }
$job = Invoke-RestMethod http://localhost:8000/v1/jobs/<job_id> -Headers $h
$job.status; $job.stage; $job.progress
Invoke-RestMethod http://localhost:8000/v1/jobs/<job_id>/result -Headers $h | ConvertTo-Json -Depth 10 | Set-Content -Encoding utf8 units.json
Invoke-RestMethod -Method Delete http://localhost:8000/v1/jobs/<job_id> -Headers $h
```

(PowerShell 7 má aj `Invoke-RestMethod -Form`, PowerShell 5.1 nie.)

## Webhooky

Ak má úloha `callback_url`, služba naň pošle `job.stage_changed` (najviac raz na krok) a na
konci `job.completed`, `job.failed` alebo `job.cancelled`. Telo obsahuje iba stav (`job_id`,
`external_ref`, `status`, `stage`, `progress`, `error`, `occurred_at`), nie výsledok.

- Podpis: `X-QGen-Signature: sha256=<HMAC-SHA256(QGEN_WEBHOOK_SECRET, "{X-QGen-Timestamp}.{telo}")>`.
  Overenie v PHP: [docs/integration.md](docs/integration.md#overenie-podpisu-v-php).
- Ak odpoveď nie je 2xx do `QGEN_WEBHOOK_TIMEOUT_S` (5 s), služba to skúsi znova po 10 s,
  60 s a 300 s. Pri každom pokuse je nový timestamp a podpis; `X-QGen-Delivery` a telo ostávajú.
  Potom doručenie vzdá (log `webhook gave up`); stav úlohy sa nemení.
- Hostiteľ z `callback_url` musí byť v `QGEN_WEBHOOK_ALLOWED_HOSTS` (kontrola pri vytvorení
  úlohy aj pred každým pokusom). Prázdne = webhooky vypnuté, úloha s `callback_url` dostane 400.
- Stav zapíše worker najprv do `job.json`, až potom zaradí webhook. `job.cancelled` posiela API
  (DELETE).

Vyskúšanie bez Laravelu:

1. Do `.env` doplňte `QGEN_WEBHOOK_ALLOWED_HOSTS=host.docker.internal` a `QGEN_WEBHOOK_SECRET=lokalny-tajny-kluc`.
2. Spustite službu (rýchly štart b). Po zmene `.env` treba `up -d` (kontajnery načítajú nové hodnoty).
3. Na počítači spustite prijímač (tajný kľúč si načíta z `.env`; `--fail 1` vráti na prvý webhook 500):

   ```bash
   venv/Scripts/python.exe tools/webhook_receiver.py --port 9000
   ```

4. Pošlite úlohu s `"callback_url": "http://host.docker.internal:9000/webhook"`.

Prijímač vypíše každý webhook a výsledok kontroly podpisu. Logy doručovania:
`docker compose logs -f webhook-worker`.

## Chyby

Chyba API má tvar `{"error": {"code": "...", "message": "...", "details": {...}}}`.

| HTTP | `code` | Kedy |
|---|---|---|
| 400 | `invalid_options` | neplatné `options`, nepovolený `callback_url`, OCR v obraze bez OCR, sylabus odkazuje na nenahratý súbor |
| 401 | `unauthorized` | chýba alebo nesedí `X-API-Key` |
| 404 | `job_not_found` / `not_found` | úloha neexistuje (aj po DELETE a po 7 dňoch) / neznáma cesta |
| 405 | `method_not_allowed` | zlá HTTP metóda |
| 409 | `result_not_ready` | `/result`, `/report` pri `queued` / `running` |
| 409 | `job_not_completed` | `/result`, `/report` pri `failed` / `cancelled` |
| 409 | `job_not_retryable` | retry mimo stavu `failed` alebo po 24 h |
| 413 | `file_too_large` | súbor nad `QGEN_MAX_UPLOAD_MB`, priveľa strán alebo súborov |
| 415 | `unsupported_file` | nie je PDF, nedá sa otvoriť, je zaheslovaný |
| 422 | `validation_error` | chýba časť requestu (napr. `files`) |
| 429 | `too_many_jobs` | viac ako `QGEN_MAX_ACTIVE_JOBS` aktívnych úloh, hlavička `Retry-After` |
| 500 | `internal_error` | neočakávaná chyba |
| 503 | – | `/health`: Redis nie je dostupný |

Chyba **úlohy** (stav `failed`, pole `error` v stave a webhooku) má vlastné kódy:
`llm_error` (poskytovateľ LLM zlyhal aj po opakovaniach), `worker_lost` (worker dvakrát spadol),
`extraction_failed` (z PDF sa nedal získať text), `internal_error` (iná chyba, aj časový limit
`QGEN_JOB_TIMEOUT_S`). Úlohu `failed` možno do 24 h zopakovať cez `POST /v1/jobs/{id}/retry`.

Varovania (`warnings` v reporte) úlohu nezastavia; zoznam a texty pre učiteľa sú v
[docs/integration.md](docs/integration.md#7-varovania-ktoré-môže-zobraziť-učiteľovi).

## Replay, smoke test a overenie odolnosti

**Replay** (iba vývoj): `QGEN_LLM_PROVIDER=replay` a `QGEN_EMBED_PROVIDER=replay` odpovedajú
z nahrávky `llm_calls.jsonl` namiesto API. Žiadny kľúč, žiadne náklady, rovnaký výsledok.
`docker-compose.replay.yml` to nastaví spolu s modelmi a nastaveniami nahrávky `baseline_v5`.
Funguje iba `eko.pdf` bez OCR; iný dokument skončí `llm_error`. Worker pri štarte zapíše
výrazné varovanie `REPLAY` a `GET /v1/info` ukáže `"replay": true`. Názvy hodín v nahrávke nie
sú, preto výsledok má varovanie `lesson_titles_failed`. Replay je povolený iba pri
`QGEN_ENV=development`; pri `production` služba nenaštartuje.

### Smoke test

`tools/smoke_test.py` overí finálny Docker: build, štart štyroch kontajnerov, `/health` 200
do 60 s, healthchecky, beh pod používateľom bez root práv, OCR (modely v obraze / 400 v obraze
bez OCR), jednu úlohu od POST po `completed`, výsledok voči JSON schéme, časy odpovedí
(< 200 ms) a na konci `docker compose down -v`. Používa vlastný projekt `qgen-smoke` a port 18000.

```bash
venv/Scripts/python.exe tools/smoke_test.py                       # obraz s OCR, replay
venv/Scripts/python.exe tools/smoke_test.py --install-ocr false   # menší obraz bez OCR
venv/Scripts/python.exe tools/smoke_test.py --real                # skutočné Gemini z .env (stojí peniaze)
```

Potrebuje Docker, `.env`, `eko.pdf` (`tests/fixtures/`) a bez `--real` nahrávku `baseline_v5/`.
Build s OCR trvá prvýkrát asi 3 minúty.

### Overenie odolnosti

`tools/resilience_check.py` overí odolnosť v kontajneroch (špecifikácia, kap. 9) na replay:

| Scenár | Postup | Očakávaný výsledok |
|---|---|---|
| a | reštart `api` a `redis` hneď po 202 | úloha dobehne `completed` |
| a2 | worker stojí, po 202 `FLUSHALL` v Redise, štart workera | worker obnoví úlohu z job.json, `completed` |
| b | `docker kill` workera počas `running`, štart | `completed`, `attempts` = 2 |
| c | dvakrát kill počas behu | `failed` s `worker_lost`, webhook `job.failed`, PDF ostali |
| d | DELETE počas behu | `cancelled`, `job.cancelled` práve raz, PDF zmazané, `/result` 409 |
| e | `QGEN_LLM_REPLAY_FAIL=1` | `failed` s `llm_error`; po vypnutí `POST /retry` → `completed`, rovnaké `job_id` |
| f | opakovaný POST s rovnakým `external_ref` | 200, rovnaké `job_id` |
| g | prijímač vráti 500, medzitým reštart `webhook-worker` | webhook doručený na 3. pokus, rovnaké `X-QGen-Delivery` |

```bash
venv/Scripts/python.exe tools/resilience_check.py                 # všetky scenáre, asi 15 min
venv/Scripts/python.exe tools/resilience_check.py --only b,d --keep-up
```

Porty 8000 a 9000 musia byť voľné. Replay compose nastaví `QGEN_MAINTENANCE_INTERVAL_S=15`,
aby sa úloha zabitého workera vrátila do fronty rýchlo (predvolene 600 s). Lehoty 24 h a 7 dní
overujú unit testy s posunutými hodinami (`tests/worker/test_retention.py`, `tests/jobs/test_cleanup.py`).

## Nahrávky (baseline)

`tools/baseline.py` spustí pipeline na `eko.pdf`, nahrá všetky volania LLM
(`llm_calls.jsonl`) a zapíše metriky. V gite je iba `metrics.json`; nahrávka obsahuje text
dokumentu a ostáva lokálne. Chránené nahrávky sa dajú prepísať iba s `--force`.

| Priečinok | Modely | Čo meria / prečo existuje |
|---|---|---|
| `baseline/` | gemini-2.5-flash-lite + 2.5-flash | Pôvodný stav (Streamlit kód): 41 celkov, 79 otázok, 11 tém, vernosť otázok 4,73, 330 s. Referencia „kvalita sa nezhorší“. Historická, chránená. |
| `baseline_v2/` | gemini-3.1-flash-lite + 3.5-flash | Prvý beh na lacnejších nových modeloch s pôvodnými promptmi: iba 25 celkov a 22 otázok – problém lacného modelu, ktorý viedol k úprave generovania (minimum na celok, pokrytie). 91 s. |
| `baseline_v3/` | 3.1-flash-lite + 3.5-flash | Po úprave generovania, starý formát otázok: 47 celkov, 132 otázok, 287 s. Historická, chránená. |
| `baseline_v4/` | 3.1-flash-lite + 3.5-flash | Krok 13: nové typy otázok (choice / open / no_answer), prompty s nápovedami, sekvenčne: 167 otázok, 461 s. Historická, chránená. |
| `baseline_v4_parallel/` | 3.1-flash-lite + 3.5-flash | Krok 13b: paralelné volania, `QGEN_LLM_MAX_CONCURRENCY=4`: 138 s (zrýchlenie 3,3× oproti v4). |
| `baseline_v4_par8/` | 3.1-flash-lite + 3.5-flash | To isté so súbežnosťou 8: 141 s – ďalšie zvýšenie už nezrýchľuje (limit poskytovateľa). |
| `baseline_v5/` | 3.1-flash-lite + 3.5-flash | **Referencia** (commit dff7c00): nápovedy z kódu, opravený filter otázok, súbežnosť 8. 42 celkov, 159 otázok, pokrytie tém 100 %, vernosť 4,93, 159 s. Používajú ju replay testy, replay compose, smoke test a resilience check. |
| `baseline_v5_single/`, `_topics/`, `_lessons3/`, `_syllabus_text/` | – | Krátke nadstavbové nahrávky nad v5 (`--base baseline_v5 --split …`): iba volania kroku split (názvy hodín, plán sylabu) pre jednotlivé režimy. Test `tests/output/test_split_replay.py`. |

Zmena promptu alebo modelu = nová nahrávka (`tools/baseline.py --out baseline_v6`), nie úprava testov.

## Vývoj a testy

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt     # služba + OCR + pytest (Windows: venv\Scripts\...)
venv/Scripts/python.exe -m pytest                 # všetky testy (asi 1–2 min)
```

- Testy nikdy nevolajú skutočné LLM ani embedding API (falošní klienti a replay). Testy na
  nahrávke `baseline_v5/` sa bez nej preskočia.
- RQ worker potrebuje `fork`, preto beží iba v Dockeri (Linux). Testy používajú `SimpleWorker`
  a `fakeredis`.
- Mimo `qgen/llm/` sa nesmie importovať knižnica výrobcu LLM (`tests/test_import_guard.py`).
- Pravidlá a rozhodnutia pre vývoj sú v `CLAUDE.md`.

### Pipeline z príkazového riadka (CLI)

```bash
venv/Scripts/python.exe -m qgen.cli eko.pdf --out vystup
venv/Scripts/python.exe -m qgen.cli a.pdf b.pdf --out vystup --ocr --language sk
venv/Scripts/python.exe -m qgen.cli eko.pdf --out vystup --split '{"mode": "topics"}'
```

`--split` spustí aj rozdelenie do hodín a zapíše `units.json` a `report.json`.

### Ďalšie nástroje

| Nástroj | Účel |
|---|---|
| `tools/baseline.py` | nahrávka a metriky na referenčnom dokumente |
| `tools/smoke_llm.py` | jedno skutočné volanie LLM a embeddingu s aktuálnou konfiguráciou |
| `tools/smoke_test.py` | smoke test Dockeru (pozri vyššie) |
| `tools/resilience_check.py` | odolnosť v kontajneroch |
| `tools/webhook_receiver.py` | lokálny prijímač webhookov s kontrolou podpisu |
| `tools/export_schemas.py` | vygeneruje `schemas/*.json` z Pydantic modelov |
| `tools/dedup_report.py` | podobnosť otázok (kalibrácia prahu duplicít) |

### Štruktúra projektu

```
qgen/            jadro (bez webu): contract/ (Pydantic modely), extraction/, lo/ (tematické celky),
                 questions/, split/ (rozdelenie do hodín, čisté funkcie), output/ (units, limity,
                 nh3, report), llm/ (rozhranie a adaptéry), config.py (všetky premenné)
service/         FastAPI (main.py, routes/), jobs/ (job.json, fronta, worker, obnova, mazanie),
                 webhooks/, mock/, worker.py, webhook_worker.py, healthcheck.py, startup.py
schemas/         JSON schémy units, report, options (generované)
tools/           nástroje (tabuľka vyššie)
tests/           unit, kontraktové, replay a integračné testy
docs/            špecifikácia, deployment.md, integration.md
Dockerfile, Dockerfile.mock, docker-compose*.yml, .env.example
```

Popis jednotlivých súborov: `popis_suborov..txt`.

## Mock API

Mock má rovnaké endpointy a formáty ako skutočná služba, ale úlohy iba simuluje (žiadne LLM,
žiadny Redis) a vždy vráti ten istý príklad výsledku. Slúži na vývoj integrácie v Laraveli.

```bash
cp .env.example .env
docker compose -f docker-compose.mock.yml up --build      # http://localhost:8000/docs
```

- Úloha trvá `QGEN_MOCK_DURATION_S` (30 s) a prejde všetkými krokmi; webhooky posiela rovnako
  ako služba (podpis, opakovania).
- `external_ref` začínajúci `mock-fail-`: prvý pokus skončí `failed` s `llm_error`, `POST /retry`
  potom uspeje. `mock-slow-`: úloha trvá 5× dlhšie.
- Rozdiely: výsledok je vždy rovnaký, úlohy sú iba v pamäti (reštart ich zmaže), PDF sa
  nekontroluje ani neukladá, `extraction.ocr=true` sa vždy prijme.

Prechod na skutočnú službu: [docs/integration.md](docs/integration.md#1-prechod-z-mocku-na-skutočnú-službu).
