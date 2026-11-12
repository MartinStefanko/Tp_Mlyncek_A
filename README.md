> Tento súbor popisuje pôvodnú Streamlit verziu (Hlavna_cast/) a aktualizuje sa v poslednom kroku prerábky.

# Nástroj na automatické generovanie otázok a úloh

Tento projekt predstavuje nástroj pre automatické generovanie vzdelávacích objektov, otázok, úloh, odpovedí a nápovedí z PDF dokumentov pomocou veľkého jazykového modelu.

Aplikácia je vytvorená v jazyku Python a používa webové rozhranie Streamlit. Na generovanie a hodnotenie výstupov využíva veľký jazykový model Gemini cez API kľúč.

## Hlavné funkcie

- nahratie PDF dokumentu cez webové rozhranie,
- extrakcia textu z PDF dokumentu,
- OCR spracovanie textu z obrázkov v dokumente,
- generovanie vzdelávacích objektov,
- zhlukovanie podobných vzdelávacích objektov,
- dopĺňanie prerekvizít medzi vzdelávacími objektmi,
- validácia a filtrácia vygenerovaných výstupov,
- generovanie otázok a praktických úloh,
- generovanie odpovedí a nápovedí,
- citovanie strán odkiaľ otázky vznikli,
- export výsledkov do JSON, TXT a PDF súborov,
- vizualizácia vzťahov medzi vzdelávacími objektmi.

## Požiadavky

Pred spustením projektu je potrebné mať nainštalované:

- odporúčaná verzia: Python 3.12,
- nástroj `pip`,
- internetové pripojenie,
- API kľúč pre Gemini,
- vo Windowse: Microsoft Visual C++ v14 Redistributable, odporúčané nainštalovať verziu x64 aj x86.

Projekt bol vyvíjaný a testovaný na Python 3.12.6. Použitie Pythonu 3.10 alebo novšieho by malo byť možné, ale odporúča sa Python 3.12.

Projekt bol pripravovaný ako lokálna Streamlit aplikácia.

Skontrolujte si, či máte nainštalované Microsoft Visual C++ v14 Redistributable x64 a x86.

Inštalačný súbor je možné stiahnuť z oficiálnej stránky Microsoftu:

```text
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
```

Vo Windowse je možné Microsoft Visual C++ v14 Redistributable nainštalovať aj cez `winget` a to nasledovne:

```bash
winget install --id Microsoft.VCRedist.2015+.x64 -e
winget install --id Microsoft.VCRedist.2015+.x86 -e
```

## Inštalácia

Najskôr si stiahnite alebo naklonujte projekt a otvorte jeho koreňový priečinok v termináli.

Vytvorte virtuálne prostredie (Windows):

```bash
python -m venv venv
```

Pre Linux je potrebova najprv nainštalovať knižnicu venv.

Aktivujte virtuálne prostredie.

Vo Windows:

```bash
venv\Scripts\activate
```

V Linuxe alebo macOS:

```bash
source venv/bin/activate
```

Nainštalujte potrebné knižnice:

```bash
pip install -r requirements.txt
```

## Nastavenie API kľúča

Projekt používa model Gemini, preto je potrebné nastaviť API kľúč.

V koreňovom priečinku projektu vytvorte súbor s názvom:

```text
.env
```

Do súboru `.env` vložte riadok:

```env
GEMINI_API_KEY=vas_api_kluc
```

Hodnotu `vas_api_kluc` nahraďte vlastným API kľúčom pre Gemini.

Príklad:

```env
GEMINI_API_KEY=AIza...
```

Súbor `.env` nesmie byť zverejnený v repozitári, pretože obsahuje súkromný API kľúč.

## Spustenie aplikácie

Po nainštalovaní závislostí a nastavení API kľúča spustite aplikáciu príkazom:

```bash
streamlit run Hlavna_cast/app.py
```

Po spustení sa v termináli zobrazí lokálna adresa aplikácie, napríklad:

```text
http://localhost:8501
```

Túto adresu otvorte vo webovom prehliadači.

## Použitie aplikácie

1. Otvorte aplikáciu v prehliadači.
2. Prejdite na časť určenú na nahratie dokumentu.
3. Nahrajte jeden alebo viac PDF dokumentov.
4. Spustite extrakciu a spracovanie dokumentov.
5. Aplikácia automaticky vygeneruje vzdelávacie objekty.
6. Následne sa vygenerujú otázky, úlohy, odpovede a nápovedy.
7. Výsledky je možné zobraziť v aplikácii a stiahnuť vo výstupných formátoch.

## Výstupy

Výstupy sa ukladajú do priečinka:

```text
Hlavna_cast/vystup
```

Medzi hlavné výstupy patria:

- extrahovaný text z dokumentu,
- vzdelávacie objekty,
- otázky a úlohy,
- odpovede a nápovedy,
- PDF a TXT exporty,
- vizualizácia vzťahov medzi vzdelávacími objektmi.


## Riešenie častých problémov

Ak aplikácia vypíše chybu, že chýba `GEMINI_API_KEY`, skontrolujte, či existuje súbor `.env` v koreňovom priečinku projektu a či obsahuje správny riadok:

```env
GEMINI_API_KEY=vas_api_kluc
```

Ak sa aplikácia nespustí kvôli chýbajúcim knižniciam, znova spustite:

```bash
pip install -r requirements.txt
```

Ak sa zobrazí chyba typu `DLL load failed`, vo Windows pravdepodobne chýba Microsoft Visual C++ v14 Redistributable. Nainštalujte obe verzie:

- Microsoft Visual C++ v14 Redistributable x64,
- Microsoft Visual C++ v14 Redistributable x86.

Oficiálna stránka Microsoftu na stiahnutie:

```text
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
```

Príkazy pre inštaláciu Microsoft Visual C++ v14 Redistributable cez `winget`:

```bash
winget install --id Microsoft.VCRedist.2015+.x64 -e
winget install --id Microsoft.VCRedist.2015+.x86 -e
```

Po inštalácii odporúčame zavrieť terminál, otvoriť ho znova, aktivovať virtuálne prostredie a aplikáciu spustiť opäť.

Ak sa príkaz `streamlit` nerozpozná, skontrolujte, či je aktivované virtuálne prostredie.

Ak spracovanie trvá dlhšie, je to očakávané pri väčších PDF dokumentoch, dokumentoch s obrázkami alebo pri väčšom počte generovaných otázok a úloh.

## Poznámky

Projekt vyžaduje internetové pripojenie, pretože generovanie výstupov prebieha cez Gemini API.

Kvalita výstupov závisí od kvality vstupného PDF dokumentu, extrahovaného textu a odpovedí jazykového modelu.

API kľúč uchovávajte iba lokálne v súbore `.env` a nezdieľajte ho verejne.

## Mock API pre backend

Mock API má rovnaké endpointy ako budúca služba (Príloha A). Úlohy však iba simuluje.
Pipeline ani LLM nevolá. Každá hotová úloha vráti rovnaký výsledok: príklad formátu `units`
z kap. 6 a príklad reportu z Prílohy A. Backend ECEH (Laravel) tak môže integráciu robiť
skôr, ako bude pipeline hotová.

### Spustenie

```bash
cp .env.example .env          # nastavte QGEN_API_KEYS, QGEN_WEBHOOK_*
docker compose -f docker-compose.mock.yml up --build
```

API beží na `http://localhost:8000`. Dokumentácia OpenAPI je na `http://localhost:8000/docs`.

Bez Dockeru (z koreňa projektu, vo virtuálnom prostredí):

```bash
pip install -r requirements-mock.txt
uvicorn service.main:app --port 8000
```

Mock číta tieto premenné: `QGEN_API_KEYS`, `QGEN_WEBHOOK_ALLOWED_HOSTS`, `QGEN_WEBHOOK_SECRET`,
`QGEN_WEBHOOK_TIMEOUT_S`, `QGEN_MAX_FILES`, `QGEN_MAX_UPLOAD_MB`, `QGEN_MAX_ACTIVE_JOBS`,
`QGEN_FAILED_INPUT_TTL_HOURS`, `QGEN_OCR_ENABLED` a
`QGEN_MOCK_DURATION_S` (dĺžka jednej úlohy v sekundách, predvolene 30).

### Príklady (curl)

```bash
# Vytvorenie úlohy -> 202 a job_id. Pole súborov môže mať názov files alebo files[].
curl -X POST http://localhost:8000/v1/jobs \
  -H "X-API-Key: <kluc>" \
  -F "files=@eko.pdf" \
  -F 'options={"external_ref": "gen-01JB7Q3K",
               "callback_url": "http://eceh-backend/api/internal/qgen/webhook",
               "split": {"mode": "topics"}}'

# Rovnaký request znova -> 200, rovnaké job_id, "replayed": true
# Stav úlohy
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>
# Hľadanie podľa external_ref
curl -H "X-API-Key: <kluc>" "http://localhost:8000/v1/jobs?external_ref=gen-01JB7Q3K"
# Výsledok a report (iba v stave completed, inak 409)
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/result
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/report
# Zrušenie (queued/running -> 200, cancelled) alebo zmazanie (koncový stav -> 204)
curl -X DELETE -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>
# Opakovanie úlohy v stave failed (do 24 h) -> 202, stav queued
curl -X POST -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/retry
# JSON schémy
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/schema/lesson
```

### Priebeh úlohy

Úloha je prvých 5 % času v stave `queued` (`stage` je `null`). Potom je v stave `running`
a postupne prejde krokmi `extract`, `language`, `learning_objects`, `lo_evaluation`,
`questions`, `q_evaluation`, `split`, `output`. Hodnota `progress` rastie od 0 do 1.
Po `QGEN_MOCK_DURATION_S` sekundách je úloha v stave `completed`.

Ak má úloha `callback_url`, mock pošle webhook `job.stage_changed` (najviac raz na krok)
a na konci `job.completed`, `job.failed` alebo `job.cancelled`. Podpis je
`X-QGen-Signature: sha256=HMAC-SHA256(QGEN_WEBHOOK_SECRET, timestamp + "." + telo)`.
Ak Laravel neodpovie 2xx, mock pokus zopakuje po 10 s, 60 s a 300 s. Pri každom pokuse
pošle nový `X-QGen-Timestamp` a nový podpis. `X-QGen-Delivery` a telo ostanú rovnaké.

### Simulácia chýb a pomalých úloh

Správanie určuje začiatok hodnoty `external_ref`:

| `external_ref` začína | Správanie |
|---|---|
| `mock-fail-` | Prvý pokus skončí v kroku `questions` stavom `failed`, `error.code` = `llm_error`, `retryable_until` = čas chyby + 24 h. `POST /retry` úlohu spustí znova a druhý pokus skončí `completed`. |
| `mock-slow-` | Úloha trvá 5× dlhšie (predvolene 150 s). Vhodné na test zrušenia a pollingu. |
| iné | Normálny priebeh. |

Príklad: `"external_ref": "mock-fail-gen-42"`.

### Rozdiely oproti skutočnej službe

- Výsledok a report sú vždy rovnaké. Obsah PDF sa nečíta.
- Úlohy sú iba v pamäti. Reštart kontajnera ich zmaže. Neexistuje `job.json` ani Redis.
- Počet strán ani heslo PDF sa nekontrolujú. `extraction.ocr = true` sa vždy prijme.
- Výsledky sa nemažú po 7 dňoch. PDF sa neukladá.
- Webhooky posiela API proces, nie worker. Opakovania sa pri reštarte stratia.

### Skutočný backend

`QGEN_BACKEND=real` zapne skutočný backend: úlohy v `QGEN_DATA_DIR/jobs/{id}/job.json`,
rezervácia `external_ref` a fronta v Redise (`QGEN_REDIS_URL`). API kontroluje aj počet strán,
heslo PDF a dostupnosť OCR. Úlohy spracuje worker. Spustenie: pozri časť
[Lokálne spustenie skutočnej služby](#lokálne-spustenie-skutočnej-služby).

## Lokálne spustenie skutočnej služby

Skutočná služba má štyri kontajnery: `redis`, `api` (FastAPI), `worker` (RQ, pipeline) a
`webhook-worker` (RQ, iba doručovanie webhookov). API a worker zdieľajú volume `/data`
(job.json, PDF, výsledky). Worker spracuje úlohu celou pipeline a
volá skutočné LLM API podľa `.env`. Obraz je zatiaľ bez OCR (`INSTALL_OCR=false`); úlohu s
`extraction.ocr=true` API odmietne. RQ worker potrebuje `fork`, preto beží iba v Dockeri
(Linux), nie priamo vo Windowse.

```bash
cp .env.example .env     # vyplňte GEMINI_API_KEY (alebo iného poskytovateľa), voliteľne QGEN_API_KEYS
docker compose -f docker-compose.dev.yml up --build
```

`QGEN_BACKEND=real`, `QGEN_DATA_DIR=/data` a `QGEN_REDIS_URL` nastaví compose sám.

- `http://localhost:8000/health` vráti HTTP 200, kým API vie prijímať úlohy (Redis je
  dostupný); 503 iba pri nedostupnom Redise. V tele je `"worker"` a `"webhook_worker"`
  (`"ok"` / `"error"`; `"webhook_worker": "disabled"`, keď je `QGEN_WEBHOOK_ALLOWED_HOSTS`
  prázdne) a celkový `"status"`: `"ok"`, alebo `"degraded"`, ak niektorá časť nie je ok.
- `http://localhost:8000/docs`: pri **POST /v1/jobs** kliknite na *Try it out*. Do `files`
  pridajte `Hlavna_cast/ucebny_material/eko.pdf`. Do `options` zadajte napríklad
  `{"external_ref": "test-1"}`. Ak je nastavený `QGEN_API_KEYS`, vyplňte aj `x-api-key`.
  Kliknite na *Execute*. Odpoveď 202 obsahuje `job_id`.
- Stav: **GET /v1/jobs/{job_id}** (stage a progress). Spracovanie eko.pdf trvá asi 5 minút.
  Po stave `completed` sú výsledky na **GET /v1/jobs/{job_id}/result** a **/report**.

To isté cez curl:

```bash
curl -X POST http://localhost:8000/v1/jobs -H "X-API-Key: <kluc>"   -F "files=@Hlavna_cast/ucebny_material/eko.pdf" -F 'options={"external_ref": "test-1"}'
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>
curl -H "X-API-Key: <kluc>" http://localhost:8000/v1/jobs/<job_id>/result
```

Logy workera: `docker compose -f docker-compose.dev.yml logs -f worker`.
Zastavenie: `docker compose -f docker-compose.dev.yml down` (s `-v` zmaže aj dáta).

### Webhooky

Ak má úloha v options `callback_url`, služba naň pošle `job.stage_changed` (najviac raz na
krok) a na konci `job.completed`, `job.failed` alebo `job.cancelled`. Worker (a pri DELETE
API) najprv zapíše stav do job.json a potom zaradí webhook do fronty `qgen-webhooks`.
Doručuje ho samostatný kontajner `webhook-worker`, preto nečaká na bežiacu úlohu. Ak
odpoveď nie je 2xx do `QGEN_WEBHOOK_TIMEOUT_S`, skúsi to znova po 10 s, 60 s a 300 s. Potom
doručenie vzdá (log `webhook gave up`); stav úlohy sa nemení. Hostiteľ sa kontroluje pri
vytvorení úlohy aj pred každým pokusom.

Vyskúšanie bez Laravelu:

1. Do `.env` doplňte:

   ```env
   QGEN_WEBHOOK_ALLOWED_HOSTS=host.docker.internal
   QGEN_WEBHOOK_SECRET=lokalny-tajny-kluc
   ```

2. Spustite službu (`docker compose -f docker-compose.dev.yml up --build`). Po zmene `.env`
   reštartujte kontajnery (`up -d`), aby načítali nové hodnoty.
3. Na počítači (mimo Dockeru) spustite prijímač. Tajný kľúč si načíta z `.env`:

   ```bash
   venv/Scripts/python.exe tools/webhook_receiver.py --port 9000
   ```

   `--fail 1` vráti na prvý webhook 500, takže uvidíte opakovanie po 10 s.
4. Vytvorte úlohu s `callback_url` na `host.docker.internal`:

   ```bash
   curl -X POST http://localhost:8000/v1/jobs -H "X-API-Key: <kluc>" \
     -F "files=@Hlavna_cast/ucebny_material/eko.pdf" \
     -F 'options={"external_ref": "test-hook", "callback_url": "http://host.docker.internal:9000/webhook"}'
   ```

Prijímač vypíše každý webhook (udalosť, stav, krok, priebeh, výsledok kontroly podpisu) a
odpovie 200, pri neplatnom podpise 401. `host.docker.internal` je adresa počítača z
kontajnera (Docker Desktop ju má, v Linuxe ju pridáva `extra_hosts` v compose). Logy
doručovania: `docker compose -f docker-compose.dev.yml logs -f webhook-worker`.

### Overenie odolnosti

Skript `tools/resilience_check.py` overí odolnosť skutočnej služby v kontajneroch
(špecifikácia, kap. 9: test odolnosti, test 202, test mazania). Nevolá LLM API: worker
odpovedá z nahrávky `baseline_v5/llm_calls.jsonl` (override `docker-compose.replay.yml`,
`QGEN_LLM_PROVIDER=replay`, `QGEN_EMBED_PROVIDER=replay`). Každé volanie je umelo spomalené
(`QGEN_LLM_REPLAY_DELAY_S`, predvolene 1 s), aby sa dal kontajner zabiť počas behu.

Potrebujete Docker, lokálnu nahrávku `baseline_v5/` a `eko.pdf` (v `tests/fixtures/` alebo
`Hlavna_cast/ucebny_material/`). Porty 8000 a 9000 musia byť voľné. Skript sám spustí
kontajnery, prijíma webhooky (port 9000) a na konci zavolá `docker compose down` (volume
ostanú). Beží asi 15 minút, nie je súčasťou `pytest`. Názov hodiny (krok 16) v nahrávke
baseline_v5 nie je: replay ho odmietne a služba použije záložný názov s varovaním
`lesson_titles_failed`, úloha sa dokončí.

```bash
venv/Scripts/python.exe tools/resilience_check.py                 # všetky scenáre
venv/Scripts/python.exe tools/resilience_check.py --only b,d --keep-up
```

| Scenár | Postup | Očakávaný výsledok |
|---|---|---|
| a | reštart `api` a `redis` hneď po 202 | úloha dobehne `completed` |
| a2 | worker stojí, po 202 `FLUSHALL` v Redise, štart workera | worker obnoví úlohu z job.json, `completed` |
| b | `docker kill` workera počas `running`, štart | `completed`, `attempts` = 2 |
| c | dvakrát kill počas behu | `failed` s `worker_lost`, webhook `job.failed`, PDF ostali |
| d | DELETE počas behu | `cancelled`, `job.cancelled` práve raz, PDF zmazané, `/result` 409 |
| e | `QGEN_LLM_REPLAY_FAIL=1` (zlyhá generovanie otázok) | `failed` s `llm_error`, PDF ostali; po vypnutí `POST /retry` → `completed`, rovnaké `job_id` |
| f | opakovaný POST s rovnakým `external_ref` počas behu a po dokončení | 200, rovnaké `job_id` |
| g | prijímač vráti 500, medzitým reštart `webhook-worker` | webhook doručený na 3. pokus, rovnaké `X-QGen-Delivery` |

Na konci skript vypíše tabuľku OK/CHYBA. Ak niektorý scenár zlyhá, vráti exit kód 1.

Úloha zabitého workera sa vráti do fronty až po vypršaní jeho kľúča v Redise (asi 90 s)
a pri najbližšej údržbe. Replay compose preto nastaví `QGEN_MAINTENANCE_INTERVAL_S=15`
(predvolene 600 s). Lehoty 24 h (PDF po `failed`) a 7 dní (výsledok) sa v kontajneroch
čakať nedajú. Overujú ich unit testy s posunutými hodinami (`tests/worker/test_retention.py`,
`tests/jobs/test_cleanup.py`).

Replay je iba na vývoj. Worker pri štarte zapíše do logu výrazné varovanie `REPLAY` a
`GET /v1/info` ukáže `"llm_provider": "replay"` a `"replay": true`. So službou v režime
replay funguje iba `eko.pdf` bez OCR. Pri inom dokumente úloha skončí `llm_error`.

### Časový limit a výstup

Časový limit jednej úlohy je `QGEN_JOB_TIMEOUT_S` (predvolene 3600 s).

Do databázy ECEH ide iba formát `units` (`/result`). Report (`/report`) je interná diagnostika
kvality; Laravel ho nemusí volať. Výstup zatiaľ obsahuje jednu hodinu (rozdelenie do hodín
príde neskôr). Krok výstupu (`qgen/output/`) dodrží dĺžkové limity ECEH: príliš dlhé zadanie,
názov alebo možnosť služba raz pošle LLM na skrátenie. Názov, ktorý je stále dlhý, skráti na
koniec slova. Inú otázku, ktorá je stále príliš dlhá, vyradí s varovaním `field_too_long`.
HTML nápovedy prečistí (nh3). Pred uložením worker overí units aj report modelom, JSON schémou
a vzájomnou zhodou.

## Pipeline z príkazového riadka (CLI)

CLI spustí extrakciu, tematické celky a otázky. Výsledky a reporty zapíše do priečinka
v pôvodnom formáte (`learning_objects.json`, `questions.json`, `reports/*.txt`). S `--split`
spustí aj rozdelenie do hodín a výstup služby a zapíše navyše `units.json` a `report.json`.

```bash
venv/Scripts/python.exe -m qgen.cli eko.pdf --out vystup
venv/Scripts/python.exe -m qgen.cli a.pdf b.pdf --out vystup --ocr --language sk
venv/Scripts/python.exe -m qgen.cli eko.pdf --out vystup --split '{"mode": "topics"}'
```

- `--ocr` zapne OCR obrázkov. Potrebuje knižnicu `easyocr`. Predvolená hodnota je `QGEN_OCR_ENABLED` (false).
- `--language auto|sk|en` určí jazyk dokumentu. Pri `auto` (predvolené) jazyk zistí LLM.
- `--split JSON` je `options.split` (napr. `{"mode": "lessons", "lesson_count": 3}`).
- `--verbose` vypíše aj vlastné hlásenia pipeline.

Všetky nastavenia (modely, prahy, veľkosti batchov, limity) sú v `qgen/config.py`
a v `.env.example`. Predvolené modely sú tie, s ktorými vznikol baseline.
