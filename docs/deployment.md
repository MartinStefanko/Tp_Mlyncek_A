# Nasadenie služby QGen

Návod pre nasadenie služby na server. Služba generuje z PDF otázky rozdelené do hodín
a volá ju iba backend ECEH (Laravel). Podrobnosti o API sú v Prílohe A a v
[integration.md](integration.md).

## Obsah

1. [Čo beží](#čo-beží)
2. [Požiadavky](#požiadavky)
3. [Prvé nasadenie](#prvé-nasadenie)
4. [Premenné prostredia](#premenné-prostredia)
5. [Siete a adresy](#siete-a-adresy)
6. [Volumes a zálohy](#volumes-a-zálohy)
7. [Logy](#logy)
8. [Stav služby: /health a healthchecky](#stav-služby-health-a-healthchecky)
9. [Aktualizácia na novú verziu](#aktualizácia-na-novú-verziu)
10. [Výkon a škálovanie](#výkon-a-škálovanie)
11. [Lokálny model (vLLM, Qwen)](#lokálny-model-vllm-qwen)
12. [Veľkosť obrazu s OCR a bez OCR](#veľkosť-obrazu-s-ocr-a-bez-ocr)
13. [Bezpečnosť – kontrolný zoznam](#bezpečnosť--kontrolný-zoznam)
14. [Riešenie problémov](#riešenie-problémov)

## Čo beží

`docker-compose.yml` spustí štyri kontajnery z jedného obrazu (okrem Redisu):

| Kontajner | Príkaz | Úloha | Siete |
|---|---|---|---|
| `api` | `uvicorn service.main:app` | Prijíma requesty od Laravelu, vracia stav a výsledky. Port 8000 iba v sieti. | `qgen`, `eceh` (alias `qgen-api`) |
| `worker` | `python -m service.worker` | Spracuje úlohu (volá LLM a embeddingy). Plánovač RQ: mazanie po TTL, obnova úloh. | `qgen` |
| `webhook-worker` | `python -m service.webhook_worker` | Posiela webhooky na Laravel a opakuje ich. | `qgen`, `eceh` |
| `redis` | `redis-server --appendonly yes` | Fronta a priebeh úloh. | `qgen` |
| `llm` (voliteľný) | vLLM | Lokálny model, profil `local-llm`. | `qgen` |

Zdroj pravdy o úlohe je súbor `job.json` vo volume `/data`. Redis je iba fronta. Ak Redis
stratí dáta, worker úlohy obnoví z `job.json`.

## Požiadavky

- Linux server s Dockerom a Docker Compose **2.24 alebo novším** (`docker compose version`).
- Odchádzajúce pripojenie na internet z kontajnera `worker` (API Gemini), ak nebeží lokálny model.
- Disk: obraz s OCR približne 2,8 GB (pozri [veľkosť obrazu](#veľkosť-obrazu-s-ocr-a-bez-ocr)), dáta
  úloh desiatky MB na úlohu (PDF sa po dokončení mažú, výsledky po 7 dňoch).
- RAM: približne 1 GB na worker s OCR (EasyOCR sa načíta až pri úlohe s OCR), inak menej.
- GPU iba pre voliteľný lokálny model.

## Prvé nasadenie

1. Skopírujte projekt na server (`git clone …`).
2. Vytvorte `.env`:

   ```bash
   cp .env.example .env
   ```

3. V `.env` vyplňte aspoň:

   | Premenná | Hodnota |
   |---|---|
   | `GEMINI_API_KEY` | kľúč Gemini (alebo iného poskytovateľa) |
   | `QGEN_API_KEYS` | servisný kľúč pre Laravel (vygenerujte nižšie) |
   | `QGEN_WEBHOOK_ALLOWED_HOSTS` | hostiteľ Laravelu z `callback_url` (napr. `eceh-app`) |
   | `QGEN_WEBHOOK_SECRET` | tajný kľúč webhookov (vygenerujte nižšie) |
   | `QGEN_ECEH_NETWORK` | názov Docker siete ECEH |

   Kľúč aj tajný kľúč vygenerujete takto (každý zvlášť):

   ```bash
   python3 -c "import secrets; print(secrets.token_hex(32))"
   ```

   Rovnaký `QGEN_API_KEYS` a `QGEN_WEBHOOK_SECRET` dostane Laravel.
4. Overte, že sieť ECEH existuje (`docker network ls`). Ak nie, vytvorte ju: `docker network create eceh`.
5. Spustite službu:

   ```bash
   docker compose up -d --build
   ```

6. Overte stav (z kontajnera, port nie je zverejnený):

   ```bash
   docker compose ps                                   # všetky 4 kontajnery "healthy"
   docker compose exec api python -m service.healthcheck api
   docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
   ```

`QGEN_ENV=production` (predvolené) **nedovolí štart** bez `QGEN_API_KEYS` a pri zapnutých
webhookoch bez `QGEN_WEBHOOK_SECRET`. Kontajner potom skončí a v logu je dôvod
(`not started (QGEN_ENV=production)`).

## Premenné prostredia

Všetky premenné sú v `.env.example` aj s komentármi. Kód ich číta v `qgen/config.py`.
Stĺpec „Prod.“: **áno** = treba nastaviť pri nasadení.

### Prostredie, prístup a webhooky

| Premenná | Predvolená hodnota | Prod. | Význam |
|---|---|---|---|
| `QGEN_ENV` | `production` | | `production`: služba nenaštartuje bez kľúča / tajného kľúča alebo s poskytovateľom `replay`. `development`: iba varovanie (lokálne). |
| `QGEN_API_KEYS` | prázdne | **áno** | Povolené kľúče v hlavičke `X-API-Key`, oddelené čiarkou. |
| `QGEN_WEBHOOK_ALLOWED_HOSTS` | prázdne | **áno** | Hostitelia povolení v `callback_url` (`host` alebo `host:port`). Prázdne = webhooky vypnuté. |
| `QGEN_WEBHOOK_SECRET` | prázdne | **áno** | Tajný kľúč pre podpis `X-QGen-Signature` (HMAC-SHA256). Laravel má rovnaký. |
| `QGEN_WEBHOOK_TIMEOUT_S` | `5` | | Časový limit jedného doručenia webhooku (s). |
| `QGEN_BACKEND` | `real` (compose) | | `real` = skutočná služba, `mock` = simulácia (`docker-compose.mock.yml`). Nastaví compose. |
| `QGEN_MOCK_DURATION_S` | `30` | | Iba mock: dĺžka simulovanej úlohy (s). |

### LLM a embeddingy

| Premenná | Predvolená hodnota | Prod. | Význam |
|---|---|---|---|
| `QGEN_LLM_PROVIDER` | `gemini` | | `gemini`, `openai_compatible` (OpenAI, Mistral, vLLM, Ollama …), `anthropic`. |
| `GEMINI_API_KEY` | – | **áno** (Gemini) | Kľúč Gemini (LLM aj embeddingy). |
| `ANTHROPIC_API_KEY` | – | | Kľúč Anthropic (iba LLM). |
| `QGEN_LLM_BASE_URL` | – | | `openai_compatible`: adresa vrátane `/v1`, napr. `http://llm:8001/v1`. |
| `QGEN_LLM_API_KEY` | – | | `openai_compatible`: kľúč (lokálny server ho nepotrebuje). |
| `QGEN_LLM_MODEL` | `gemini-3.1-flash-lite` | | Predvolený model (kontroly, témy, rozdelenie do hodín). |
| `QGEN_LLM_MODEL_LO` | = `QGEN_LLM_MODEL` | | Tematické celky, prerekvizity. |
| `QGEN_LLM_MODEL_QUESTIONS` | = `QGEN_LLM_MODEL` | | Generovanie otázok. |
| `QGEN_LLM_MODEL_EVAL` | `gemini-3.5-flash` | | Skóre kvality otázok (silnejší model). Pri inom poskytovateľovi ho treba zmeniť. |
| `QGEN_LLM_MAX_TOKENS` | `8192` | | Anthropic: `max_tokens` volania. |
| `QGEN_LLM_MAX_OUTPUT_TOKENS` | `32768` | | Limit výstupu pri generovaní otázok. |
| `QGEN_LLM_TEMPERATURE` | `0.2` | | Teplota (Anthropic ju nedostane). |
| `QGEN_LLM_MAX_CONCURRENCY` | `4` | | Súbežné volania LLM jedného workera. |
| `QGEN_LLM_RETRIES` | `5` | | Pokusy jedného volania (429, 5xx, timeout). |
| `QGEN_LLM_RETRY_DELAY_S` | `3.0` | | Čakanie medzi pokusmi (s), pri 429 exponenciálne. |
| `QGEN_LLM_TIMEOUT_S` | `120` | | Časový limit jedného volania (s). |
| `QGEN_EMBED_PROVIDER` | `gemini` | | `gemini` alebo `openai_compatible`; nezávislé od LLM. |
| `QGEN_EMBED_BASE_URL` | – | | `openai_compatible`: adresa vrátane `/v1`. |
| `QGEN_EMBED_API_KEY` | = `QGEN_LLM_API_KEY` | | `openai_compatible`: kľúč embeddingov. |
| `QGEN_EMBED_MODEL` | `gemini-embedding-001` | | Embedding model; prahy sú kalibrované preň. |
| `QGEN_EMBED_MAX_BATCH` | `100` | | Textov na jedno volanie. |

### Replay (iba vývoj)

| Premenná | Predvolená hodnota | Význam |
|---|---|---|
| `QGEN_LLM_REPLAY_PATH` | – | Nahrávka `llm_calls.jsonl` pre poskytovateľa `replay`. Pri `QGEN_ENV=production` služba s replay nenaštartuje. |
| `QGEN_LLM_REPLAY_DELAY_S` | `0` | Spomalenie každého volania (s). |
| `QGEN_LLM_REPLAY_FAIL` | `false` | `true` = generovanie otázok zlyhá (test opakovania). |

### Pipeline (prahy, otázky, nápovedy, batche)

Tieto hodnoty netreba meniť. Zmena mení kvalitu výstupu a vyžaduje novú referenčnú nahrávku.

| Premenná | Predvolená hodnota | Význam |
|---|---|---|
| `QGEN_CLUSTER_SIM_THRESHOLD` | `0.82` | Zhlukovanie tematických celkov (kosínová podobnosť). |
| `QGEN_CLUSTER_REQUIRE_SOURCE_OVERLAP` | `false` | Zhlukovať iba celky so spoločnou stranou. |
| `QGEN_TOPIC_MATCH_THRESHOLD` | `0.72` | Téma je pokrytá celkom. |
| `QGEN_LO_MIN_FAITHFULNESS_SCORE` | `3` | Filter tematických celkov (vernosť 1–5). |
| `QGEN_QUESTION_MIN_SCORE` | `3` | Filter otázok: skóre kvality. |
| `QGEN_QUESTION_MIN_FAITHFULNESS_SCORE` | `3` | Filter otázok: vernosť. |
| `QGEN_QUESTION_MIN_ANSWERABILITY_SCORE` | `3` | Filter otázok: zodpovedateľnosť. |
| `QGEN_QUESTION_FAITHFUL_SCORE` | `4` | Iba report: príznak „faithful“. |
| `QGEN_QUESTION_ANSWERABLE_SCORE` | `4` | Iba report: príznak „answerable“. |
| `QGEN_QUESTION_EXTRA_ROUNDS` | `1` | Doplňujúce kolá generovania. |
| `QGEN_QUESTION_DEDUP_THRESHOLD` | `0.94` | Podobnosť, od ktorej je otázka duplicitná. |
| `QGEN_QUESTION_CHOICE_MIN_SHARE` | `0.6` | Podiel choice otázok v prompte. |
| `QGEN_QUESTION_PRACTICAL_BLOOM` | `apply,analyze,evaluate,create` | Úrovne celkov s praktickou úlohou. |
| `QGEN_QUESTION_NO_ANSWER_BLOOM` | `evaluate,create` | Úrovne povolené pre `no_answer`. |
| `QGEN_HINT_NAME_SK` / `_EN` | `Zdroj` / `Source` | Názov nápovedy. |
| `QGEN_HINT_PAGE_SK` / `_EN` | `{file}, strana {pages}` / `{file}, page {pages}` | Text nápovedy, jedna strana. |
| `QGEN_HINT_PAGES_SK` / `_EN` | `{file}, strany {pages}` / `{file}, pages {pages}` | Text nápovedy, viac strán. |
| `QGEN_LO_BATCH_SIZE` | `10` | Strán na jedno volanie generovania celkov. |
| `QGEN_QUESTION_GENERATION_BATCH_SIZE` | `3` | Celkov na jedno volanie generovania otázok. |
| `QGEN_QUESTION_EVAL_BATCH_SIZE` | `20` | Otázok na jedno volanie hodnotenia. |
| `QGEN_QUESTION_GENERATION_ATTEMPTS` | `3` | Pokusy generovania otázok. |
| `QGEN_QUESTION_EVAL_ATTEMPTS` | `2` | Pokusy hodnotenia otázok. |
| `QGEN_CHECK_BATCH_ATTEMPTS` | `2` | Pokusy kontrol vernosti a zodpovedateľnosti. |
| `QGEN_CONTEXT_MAX_CHARS` | `8000` | Zdrojový text pre jeden celok alebo otázku. |
| `QGEN_LANGUAGE_SAMPLE_MAX_CHARS` | `12000` | Text na zistenie jazyka. |
| `QGEN_TOPIC_SOURCE_MAX_CHARS` | `20000` | Text na zistenie tém. |
| `QGEN_SPLIT_TITLE_BATCH_SIZE` | `10` | Hodín na jedno volanie názvov hodín. |

### Predvolené options, limity, uchovanie, fronta

| Premenná | Predvolená hodnota | Význam |
|---|---|---|
| `QGEN_OCR_ENABLED` | `false` | Predvolené OCR obrázkov, ak ho úloha nezadá. |
| `QGEN_MAX_FILES` | `10` | Najviac súborov v úlohe. |
| `QGEN_MAX_UPLOAD_MB` | `50` | Najväčšia veľkosť nahrávky (MB). |
| `QGEN_MAX_PAGES` | `300` | Najviac strán spolu. |
| `QGEN_MAX_ACTIVE_JOBS` | `10` | Najviac úloh `queued` + `running`; ďalšia dostane 429. |
| `QGEN_DATA_DIR` | `/data` | Úložisko úloh (volume). |
| `QGEN_RESULT_TTL_DAYS` | `7` | Ako dlho ostane výsledok. |
| `QGEN_FAILED_INPUT_TTL_HOURS` | `24` | Ako dlho ostane PDF po `failed` (pre retry). |
| `QGEN_REDIS_URL` | `redis://redis:6379/0` | Redis (nastaví compose). |
| `QGEN_JOB_TIMEOUT_S` | `3600` | Časový limit jednej úlohy (s). eko.pdf trvá asi 3–5 min. |
| `QGEN_MAINTENANCE_INTERVAL_S` | `600` | Údržba workera (mazanie, obnova úloh), s. |

### Nasadenie (číta docker compose)

| Premenná | Predvolená hodnota | Význam |
|---|---|---|
| `QGEN_WORKERS` | `1` | Počet kontajnerov `worker`. |
| `QGEN_INSTALL_OCR` | `true` | `false` = menší obraz bez OCR; úloha s `extraction.ocr=true` dostane 400. Platí pri builde. |
| `QGEN_ECEH_NETWORK` | `eceh` | Názov existujúcej Docker siete ECEH. |
| `QGEN_API_PORT` | `8000` | Zverejnený port (iba `docker-compose.remote.yml` a `docker-compose.dev.yml`). |
| `QGEN_API_BIND` | `0.0.0.0` | Adresa rozhrania pre zverejnený port (`docker-compose.remote.yml`). |
| `QGEN_LOCAL_MODEL` | – | Model kontajnera `llm` (Hugging Face názov). |
| `QGEN_VLLM_IMAGE` | `vllm/vllm-openai:v0.11.0` | Obraz vLLM. |

## Siete a adresy

### Variant A: ECEH a služba na jednom serveri (odporúčané)

Kontajnery `api` a `webhook-worker` sú pripojené aj do siete ECEH (`QGEN_ECEH_NETWORK`).
Port 8000 nie je zverejnený (`expose`), takže z internetu sa k službe nedá dostať.

- Laravel volá službu na adrese **`http://qgen-api:8000`** (alias v sieti ECEH; názov
  `api` by sa mohol zraziť so službou ECEH).
- Laravel posiela v `options.callback_url` adresu svojho kontajnera v sieti ECEH, napr.
  `http://eceh-app/api/qgen/webhook`.
- Hostiteľ z `callback_url` (`eceh-app`) musí byť v `QGEN_WEBHOOK_ALLOWED_HOSTS`. Inak API
  úlohu odmietne (400 `invalid_options`). Port sa zapisuje, ak nie je predvolený:
  `eceh-app:8080`.
- Worker je iba v sieti `qgen`; potrebuje odchádzajúci internet na API Gemini.

### Variant B: ECEH beží na inom serveri (otvorená otázka pre Adelu a Laca)

Spoločná Docker sieť neexistuje. Port API sa zverejní iba pre server ECEH:

```bash
docker compose -f docker-compose.yml -f docker-compose.remote.yml up -d --build
```

`docker-compose.remote.yml` zverejní `QGEN_API_BIND:QGEN_API_PORT` (predvolene
`0.0.0.0:8000`) a sieť `eceh` nahradí lokálnou sieťou (externá sieť netreba).

1. **Firewall:** port povoľte iba z IP adresy servera ECEH, napr.
   `ufw allow from <IP ECEH> to any port 8000 proto tcp`. Docker obchádza pravidlá `ufw`
   pre zverejnené porty; preto nastavte `QGEN_API_BIND` na adresu internej siete (VPN,
   privátna sieť medzi servermi), alebo pravidlo dajte do reťazca `DOCKER-USER` v iptables.
2. **Reverse proxy s TLS** (odporúčané, ak spojenie ide cez verejnú sieť): nastavte
   `QGEN_API_BIND=127.0.0.1` a pred službu dajte nginx/Caddy s HTTPS, ktoré pustí iba IP ECEH.
   API kľúč by inak išiel nešifrovane.
3. **Adresa pre Laravel:** `https://qgen.<doména>` (proxy) alebo `http://<IP servera>:8000`.
4. **Webhooky:** `callback_url` mieri na verejnú adresu ECEH, napr.
   `https://eceh.<doména>/api/qgen/webhook`; do `QGEN_WEBHOOK_ALLOWED_HOSTS` patrí
   `eceh.<doména>`. Kontajner `webhook-worker` potrebuje odchádzajúce spojenie na server ECEH.

## Volumes a zálohy

Compose projekt sa volá `qgen`, preto majú volumes predponu `qgen_`:

| Volume | Obsah | Zálohovať |
|---|---|---|
| `qgen_qgen-data` | `/data/jobs/{id}/` – `job.json`, nahraté PDF, `result.json`, `report.json` | **áno** |
| `qgen_redis-data` | fronta a priebeh | nie (úlohy sa obnovia z `job.json`) |
| `qgen_models` | cache modelov vLLM (iba profil `local-llm`) | nie (dá sa stiahnuť znova) |

Výsledky sa po 7 dňoch aj tak mažú, preto stačí denná záloha:

```bash
# záloha
docker run --rm -v qgen_qgen-data:/data:ro -v "$PWD":/backup alpine \
  tar czf /backup/qgen-data-$(date +%F).tar.gz -C /data .
# obnova (služba zastavená: docker compose stop)
docker run --rm -v qgen_qgen-data:/data -v "$PWD":/backup alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/qgen-data-2026-11-12.tar.gz -C /data"
```

Súbory patria používateľovi s UID 1000 (kontajner nebeží ako root). Pri obnove ich
vlastníka nemeňte.

`docker compose down` volumes ponechá. `docker compose down -v` ich **zmaže** aj s
výsledkami.

## Logy

```bash
docker compose logs -f api worker webhook-worker
docker compose logs --since 1h worker
```

- Rotácia: `json-file`, 10 MB × 5 súborov na kontajner (nastavené v `docker-compose.yml`).
- Log obsahuje `job_id`, `external_ref`, krok (`stage`), typ chyby a časy. Neobsahuje text
  dokumentu, API kľúče ani tajný kľúč webhookov.
- Dôležité správy: `worker not started` / `not started (QGEN_ENV=production)` (konfigurácia),
  `webhook gave up` (Laravel neprijal webhook ani po opakovaniach), `REPLAY` (vývojový
  poskytovateľ – v produkcii nesmie byť).

## Stav služby: /health a healthchecky

`GET /health` (bez API kľúča):

- **200**: API prijíma úlohy (Redis je dostupný). Telo:
  `{"status": "ok" | "degraded", "components": {"api", "redis", "worker", "webhook_worker"}}`.
  `degraded` znamená, že niektorý worker nebeží; úlohy čakajú vo fronte.
  `"webhook_worker": "disabled"` = webhooky vypnuté.
- **503**: Redis nie je dostupný; API nové úlohy neprijme.

Každý kontajner má vlastný healthcheck (`docker compose ps`, stĺpec STATUS):

| Kontajner | Kontrola |
|---|---|
| `api` | `GET /health` vráti 200 |
| `worker` | v Redise je živý RQ worker fronty `qgen` z tohto kontajnera |
| `webhook-worker` | to isté pre frontu `qgen-webhooks` |
| `redis` | `redis-cli ping` |

Ručne: `docker compose exec worker python -m service.healthcheck worker`.

Po `docker compose up` vráti `/health` 200 do 60 s (meria `tools/smoke_test.py`).

## Aktualizácia na novú verziu

```bash
git pull
docker compose build
docker compose up -d          # vymení kontajnery, volumes ostanú
docker compose ps             # všetko "healthy"
docker compose exec api python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

- Úlohy vo fronte počkajú a spracujú sa po štarte.
- Úloha, ktorá práve beží, sa preruší a spustí znova od začiatku (počíta sa ako ďalší pokus).
  Ak sa preruší druhýkrát, skončí `failed` s kódom `worker_lost`. Preto aktualizujte, keď
  nebeží žiadna úloha (`docker compose logs --tail 20 worker`).
- Verziu ukáže `GET /v1/info` (`version`, potrebuje API kľúč).
- Nová verzia môže pridať premenné; porovnajte `.env` s `.env.example`.

## Výkon a škálovanie

- Jedna úloha (eko.pdf, 69 strán) trvá s Gemini asi 3–5 minút.
- Jeden worker spracuje jednu úlohu naraz. Viac workerov: `QGEN_WORKERS=2` v `.env`, potom
  `docker compose up -d`.
- Súbežné volania LLM spolu = `QGEN_WORKERS × QGEN_LLM_MAX_CONCURRENCY`. Pri limite
  poskytovateľa (HTTP 429) znížte `QGEN_LLM_MAX_CONCURRENCY`.
- `QGEN_MAX_ACTIVE_JOBS` (10) chráni frontu; ďalšia úloha dostane 429 s `Retry-After`.

## Lokálny model (vLLM, Qwen)

Generovanie môže bežať na lokálnom modeli s OpenAI-kompatibilným API. Embeddingy ostávajú
odporúčane cez Gemini: prahy (`QGEN_CLUSTER_SIM_THRESHOLD`, `QGEN_TOPIC_MATCH_THRESHOLD`,
`QGEN_QUESTION_DEDUP_THRESHOLD`) sú kalibrované pre `gemini-embedding-001`.

### Model na tom istom serveri (profil `local-llm`)

Server potrebuje GPU NVIDIA a NVIDIA Container Toolkit. Model s približne 27–32 B parametrami
potrebuje približne 60–70 GB VRAM v plnej presnosti, s kvantizáciou (AWQ/GPTQ, 4 bity) asi 20–24 GB.

```env
QGEN_LOCAL_MODEL=Qwen/Qwen3-32B-AWQ        # príklad, model vyberie Adela
QGEN_LLM_PROVIDER=openai_compatible
QGEN_LLM_BASE_URL=http://llm:8001/v1
QGEN_LLM_MODEL=Qwen/Qwen3-32B-AWQ
QGEN_LLM_MODEL_EVAL=Qwen/Qwen3-32B-AWQ     # musí byť nastavený (inak sa použije gemini model a štart zlyhá)
QGEN_EMBED_PROVIDER=gemini                  # embeddingy ostávajú cez Gemini
GEMINI_API_KEY=...
```

```bash
docker compose --profile local-llm up -d --build
docker compose logs -f llm        # prvý štart sťahuje model (desiatky GB) do volume qgen_models
```

Verziu vLLM nastavíte v `QGEN_VLLM_IMAGE` (overte aktuálnu verziu, ktorá podporuje zvolený
model). Kontajner `llm` je iba v sieti `qgen`.

### Model na samostatnom GPU serveri

Profil `local-llm` sa nespúšťa. Stačí nastaviť adresu servera:

```env
QGEN_LLM_PROVIDER=openai_compatible
QGEN_LLM_BASE_URL=http://gpu-server:8001/v1
QGEN_LLM_API_KEY=...                        # ak ho server vyžaduje
QGEN_LLM_MODEL=<model>
QGEN_LLM_MODEL_EVAL=<model>
```

### Pred ostrou prevádzkou

Iný model treba overiť na slovenskom texte. Spravte novú nahrávku na eko.pdf
(`tools/baseline.py`) a porovnajte pokrytie tém a vernosť s `baseline_v5/metrics.json`
(README, „Nahrávky“). Ak sa zmení aj embedding model, prahy treba nakalibrovať znova.

## Veľkosť obrazu s OCR a bez OCR

Namerané `tools/smoke_test.py` 12. 11. 2026 (na disku = `docker image ls`, komprimovaný = sťahovanie z registra, `docker image inspect`):

| Obraz | Build arg | Veľkosť |
|---|---|---|
| s OCR (predvolené) | `INSTALL_OCR=true` | 2,8 GB na disku (701 MB komprimovaný) |
| bez OCR | `INSTALL_OCR=false` | 1,05 GB na disku (259 MB komprimovaný) |

Obraz s OCR obsahuje torch pre CPU, EasyOCR a modely EasyOCR (stiahnuté pri builde do
`/models`, za behu sa nič nesťahuje). Bez OCR: `QGEN_INSTALL_OCR=false` v `.env`, potom
`docker compose build`. V takom obraze API odmietne úlohu s `extraction.ocr=true`
(400 `invalid_options`, typ `ocr_unavailable`).

## Bezpečnosť – kontrolný zoznam

- [ ] `QGEN_ENV=production` (alebo nenastavené).
- [ ] `QGEN_API_KEYS` – dlhý náhodný kľúč, má ho iba Laravel.
- [ ] `QGEN_WEBHOOK_SECRET` – dlhý náhodný kľúč, má ho iba Laravel.
- [ ] `QGEN_WEBHOOK_ALLOWED_HOSTS` – iba hostiteľ Laravelu.
- [ ] Port 8000 nie je zverejnený (variant A), alebo je obmedzený na server ECEH (variant B).
- [ ] `.env` má práva iba pre správcu (`chmod 600 .env`), nie je v gite.
- [ ] `QGEN_LLM_PROVIDER` a `QGEN_EMBED_PROVIDER` nie sú `replay`.
- [ ] Zálohovaný volume `qgen_qgen-data`.

Obraz neobsahuje tajomstvá (`.dockerignore` púšťa do buildu iba `qgen/`, `service/` a
requirements). Kontajnery bežia pod používateľom `app` (UID 1000), nie ako root.

## Riešenie problémov

| Príznak | Príčina a riešenie |
|---|---|
| `network eceh declared as external, but could not be found` | Sieť ECEH neexistuje alebo má iný názov – `QGEN_ECEH_NETWORK`, alebo variant B. |
| Kontajner sa reštartuje, v logu `not started (QGEN_ENV=production)` | Chýba `QGEN_API_KEYS` alebo `QGEN_WEBHOOK_SECRET`. |
| `worker not started, configuration error` | Chýba kľúč alebo `base_url` poskytovateľa, alebo model `gemini-*` pri inom poskytovateľovi. |
| `/health` 200, ale `"status": "degraded"` | Worker alebo webhook-worker nebeží – `docker compose ps`, `logs worker`. |
| Úloha `failed` s `llm_error` | Poskytovateľ LLM odmietol volanie (kľúč, kvóta, model). `POST /v1/jobs/{id}/retry` do 24 h. |
| Webhooky neprichádzajú | `QGEN_WEBHOOK_ALLOWED_HOSTS`, sieť `webhook-worker`, log `webhook gave up`. Laravel môže stav zistiť aj cez `GET /v1/jobs/{id}`. |
| 429 `too_many_jobs` | Priveľa aktívnych úloh; počkať (`Retry-After`) alebo zvýšiť `QGEN_WORKERS`. |
