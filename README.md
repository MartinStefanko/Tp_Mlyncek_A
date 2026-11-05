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
`QGEN_FAILED_INPUT_TTL_HOURS`, `QGEN_DEFAULT_LESSON_MINUTES`, `QGEN_OCR_ENABLED` a
`QGEN_MOCK_DURATION_S` (dĺžka jednej úlohy v sekundách, predvolene 30).

### Príklady (curl)

```bash
# Vytvorenie úlohy -> 202 a job_id. Pole súborov môže mať názov files alebo files[].
curl -X POST http://localhost:8000/v1/jobs \
  -H "X-API-Key: <kluc>" \
  -F "files=@eko.pdf" \
  -F 'options={"external_ref": "gen-01JB7Q3K",
               "callback_url": "http://eceh-backend/api/internal/qgen/webhook",
               "split": {"mode": "topics", "lesson_minutes": 45}}'

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

### Skutočný backend (rozpracované)

`QGEN_BACKEND=real` zapne skutočný backend: úlohy v `QGEN_DATA_DIR/jobs/{id}/job.json`,
rezervácia `external_ref` a fronta v Redise (`QGEN_REDIS_URL`). API kontroluje aj počet strán,
heslo PDF a dostupnosť OCR. Worker zatiaľ nie je (krok 10), preto úlohy ostanú v stave `queued`.
Potrebuje `requirements.txt` (pypdf, redis, qgen/extraction).

## Lokálne spustenie skutočnej služby

Skutočná služba má tri kontajnery: `redis`, `api` (FastAPI) a `worker` (RQ). API a worker
zdieľajú volume `/data` (job.json, PDF, výsledky). Worker spracuje úlohu celou pipeline a
volá skutočné LLM API podľa `.env`. Obraz je zatiaľ bez OCR (`INSTALL_OCR=false`); úlohu s
`extraction.ocr=true` API odmietne. RQ worker potrebuje `fork`, preto beží iba v Dockeri
(Linux), nie priamo vo Windowse.

```bash
cp .env.example .env     # vyplňte GEMINI_API_KEY (alebo iného poskytovateľa), voliteľne QGEN_API_KEYS
docker compose -f docker-compose.dev.yml up --build
```

`QGEN_BACKEND=real`, `QGEN_DATA_DIR=/data` a `QGEN_REDIS_URL` nastaví compose sám.

- `http://localhost:8000/health` vráti `"worker": "ok"`, keď beží aspoň jeden worker.
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

Časový limit jednej úlohy je `QGEN_JOB_TIMEOUT_S` (predvolene 3600 s). Výstup `units` je
zatiaľ dočasný: jedna hodina, všetky otázky typu `open` (krok 14 doplní nové typy a split).

## Pipeline z príkazového riadka (CLI)

CLI spustí extrakciu, tematické celky a otázky. Výsledky a reporty zapíše do priečinka.
Formát výstupu je zatiaľ pôvodný (`learning_objects.json`, `questions.json`, `reports/*.txt`).

```bash
venv/Scripts/python.exe -m qgen.cli eko.pdf --out vystup
venv/Scripts/python.exe -m qgen.cli a.pdf b.pdf --out vystup --ocr --language sk
```

- `--ocr` zapne OCR obrázkov. Potrebuje knižnicu `easyocr`. Predvolená hodnota je `QGEN_OCR_ENABLED` (false).
- `--language auto|sk|en` určí jazyk dokumentu. Pri `auto` (predvolené) jazyk zistí LLM.
- `--verbose` vypíše aj vlastné hlásenia pipeline.

Všetky nastavenia (modely, prahy, veľkosti batchov, limity) sú v `qgen/config.py`
a v `.env.example`. Predvolené modely sú tie, s ktorými vznikol baseline.
