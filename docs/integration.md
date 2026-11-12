# Integrácia služby QGen do ECEH (Laravel)

Návod pre backend ECEH: ako prejsť z mocku na skutočnú službu, ktoré endpointy volať,
ako overiť webhook a čo zobraziť učiteľovi. Úplná špecifikácia API je v Prílohe A;
živú dokumentáciu (OpenAPI) má služba na `/docs`.

## 1. Prechod z mocku na skutočnú službu

Mock (`docker-compose.mock.yml`) a skutočná služba majú **rovnaké endpointy, rovnaké
formáty a rovnaké kódy chýb**. Kód v Laraveli sa meniť nemusí. Zmení sa iba konfigurácia:

| | Mock | Skutočná služba |
|---|---|---|
| Adresa | `http://localhost:8000` | `http://qgen-api:8000` (spoločná Docker sieť) alebo adresa od Adely ([deployment.md](deployment.md#siete-a-adresy)) |
| `X-API-Key` | nepovinný, ak je `QGEN_API_KEYS` prázdne | **povinný** (služba bez kľúča nenaštartuje) |
| Webhook secret | ľubovoľný | rovnaký ako `QGEN_WEBHOOK_SECRET` služby |
| `callback_url` | ľubovoľný hostiteľ | iba hostiteľ z `QGEN_WEBHOOK_ALLOWED_HOSTS`, inak 400 `invalid_options` |
| Trvanie úlohy | `QGEN_MOCK_DURATION_S` (30 s) | 3–5 minút na 70-stranový PDF, s OCR dlhšie |
| Výsledok | vždy ten istý príklad | otázky z nahratého PDF |
| Chyby | `mock-fail-`, `mock-slow-` v `external_ref` | skutočné (`llm_error`, `worker_lost`, `extraction_failed`) |
| PDF | kontroluje sa iba hlavička | počet strán, heslo, OCR, limity (`GET /v1/info`) |

Odporúčaný postup:

1. Nastaviť v Laraveli adresu, kľúč a secret (napr. `QGEN_URL`, `QGEN_API_KEY`, `QGEN_WEBHOOK_SECRET` v `.env` Laravelu).
2. Skontrolovať `GET /health` (bez kľúča) a `GET /v1/info` (s kľúčom) – overí adresu aj kľúč.
3. Poslať jednu úlohu s malým PDF a sledovať webhooky aj `GET /v1/jobs/{id}`.
4. Laravel nesmie spoliehať na čas úlohy z mocku: časový limit čakania nastaviť aspoň na
   `QGEN_JOB_TIMEOUT_S` (1 hodina).

## 2. Ktoré endpointy Laravel potrebuje

Všetky okrem `/health` vyžadujú hlavičku `X-API-Key`.

| Endpoint | Kedy | Poznámka |
|---|---|---|
| `POST /v1/jobs` | učiteľ spustí generovanie | multipart: `files` (1–10 PDF) + `options` (JSON text). 202 = úloha prijatá, PDF môžete zmazať. |
| `GET /v1/jobs/{id}` | záloha k webhookom, napr. raz za minútu | `status`, `stage`, `progress` (0–1), `error`. |
| `GET /v1/jobs?external_ref=…` | po timeoute pri POST (nemáte `job_id`) | zoznam úloh s týmto `external_ref`. |
| `GET /v1/jobs/{id}/result` | po `job.completed` | **formát units** – toto ide do databázy ECEH. |
| `DELETE /v1/jobs/{id}` | po uložení výsledku; alebo zrušenie bežiacej úlohy | hotová úloha: 204 a zmazanie; bežiaca: 200 a `cancelled`. |
| `POST /v1/jobs/{id}/retry` | po `failed`, do 24 h | rovnaké `job_id`, stav `queued`. Po 24 h 409 `job_not_retryable` → nová úloha. |
| `GET /v1/jobs/{id}/report` | **voliteľné** | interná diagnostika kvality; Laravel ho nemusí volať ani ukladať. Iba ak chcete učiteľovi ukázať varovania (kap. 7). |

`external_ref` posielajte vždy (`generations.external_ref`). Opakovaný POST s rovnakým
`external_ref` nevytvorí novú úlohu: vráti 200, `"replayed": true` a pôvodné `job_id`.

### Stavy úlohy

`queued` → `running` → `completed` | `failed` | `cancelled`.
Kroky (`stage`): `extract`, `language`, `learning_objects`, `lo_evaluation`, `questions`,
`q_evaluation`, `split`, `output`.

Kódy chyby úlohy (`error.code` pri `failed`):

| Kód | Význam | Čo urobiť |
|---|---|---|
| `llm_error` | poskytovateľ LLM zlyhal aj po opakovaniach | `POST /retry` neskôr |
| `worker_lost` | worker dvakrát spadol počas úlohy | `POST /retry` |
| `extraction_failed` | z PDF sa nedal získať text | iné PDF, prípadne s OCR |
| `internal_error` | chyba služby alebo časový limit | `POST /retry`, pri opakovaní nahlásiť |

## 3. Formát `options`

`options` je JSON text vo formulárovom poli. Všetko okrem `external_ref` je nepovinné.
JSON schéma: `GET /v1/schema/options`.

```json
{
  "external_ref": "gen-123",
  "callback_url": "http://eceh-app/api/qgen/webhook",
  "language": "auto",
  "extraction": { "ocr": false },
  "questions": { "types": ["choice", "open", "no_answer"], "per_section": 3, "choice_options": 4, "hints": 2 },
  "split": { "mode": "topics" }
}
```

| Pole | Predvolené | Význam |
|---|---|---|
| `language` | `auto` | `auto`, `sk`, `en` – jazyk otázok podľa dokumentu. |
| `extraction.ocr` | `false` | OCR obrázkov v PDF (pomalšie). V obraze bez OCR → 400. |
| `questions.types` | všetky tri | Typy otázok. |
| `questions.per_section` | 3 | **Minimum** otázok na tematický celok (1–10), nie strop. |
| `questions.choice_options` | 4 | Počet možností choice otázky (2–10). |
| `questions.hints` | 2 | 0 = bez nápovedy, 1–2 = jedna nápoveda so zdrojom (dokument a strany). |

### Režimy rozdelenia do hodín (`split`)

Jeden unit vo výsledku = jedna vyučovacia hodina. Počet hodín sa **neurčuje** z času otázok;
`lesson_minutes` sa prijme (kontroluje sa rozsah), ale nemá vplyv.

| Režim | JSON | Čo robí |
|---|---|---|
| `single` (predvolený) | `{"mode": "single"}` | Jedna hodina so všetkými otázkami. LLM jej dá názov a popis. |
| `lessons` | `{"mode": "lessons", "lesson_count": 3}` | Presne `lesson_count` hodín (1–100), čo najrovnomernejšie, rezy radšej na hraniciach tém. Viac hodín ako celkov → zníži (`lesson_count_reduced`). |
| `topics` | `{"mode": "topics"}` | Jedna hodina = jedna téma dokumentu; malé témy sa spoja, veľké rozdelia. |
| `syllabus` (zoznam) | pozri nižšie | Hodiny podľa učiteľa: názov, voliteľný popis a strany. |
| `syllabus` (text) | `{"mode": "syllabus", "syllabus_text": "3 hodiny: najprv úrokovanie, potom anuity, perpetuity vynechaj"}` | Voľný text učiteľa (≤ 4000 znakov); LLM ho iba prevedie na zoznam hodín. Nedá sa použiť → záloha `topics` + `syllabus_fallback`. |

Sylabus ako zoznam (`syllabus` alebo `syllabus_text`, práve jedno):

```json
{
  "mode": "syllabus",
  "syllabus": [
    { "title": "Úrokovanie", "pages": "D1:1-20" },
    { "title": "Anuity", "description": "Súčasná a budúca hodnota anuity", "pages": "D1:21-40, D2:1-5" }
  ]
}
```

`pages`: `D1` = prvý nahratý súbor, `D2` = druhý … (podľa poradia v `files`). Strany
nepokryté žiadnou hodinou sa vynechajú (`syllabus_skipped_sections`). `D3` pri dvoch
súboroch → 400 `invalid_options`.

### Príklad volania (PHP, Laravel HTTP klient)

```php
$response = Http::withHeaders(['X-API-Key' => config('qgen.api_key')])
    ->timeout(60)
    ->attach('files', fopen($pdfPath, 'r'), $originalName)
    ->post(config('qgen.url') . '/v1/jobs', [
        'options' => json_encode([
            'external_ref' => $generation->external_ref,
            'callback_url' => route('qgen.webhook'),
            'split' => ['mode' => 'lessons', 'lesson_count' => 3],
        ], JSON_UNESCAPED_UNICODE),
    ]);
// 202 nová úloha, 200 existujúca (replayed), 4xx chyba: $response->json('error.code')
```

`$originalName` sa objaví v nápovedách („eko.pdf, strana 14“), preto posielajte pôvodný názov súboru.

## 4. Webhooky

Pri každom novom kroku (`job.stage_changed`, najviac raz na krok) a na konci
(`job.completed`, `job.failed`, `job.cancelled`) služba pošle `POST` na `callback_url`.

Telo (iba stav, nie výsledok):

```json
{
  "event": "job.completed",
  "job_id": "7f3c2a1e…",
  "external_ref": "gen-123",
  "status": "completed",
  "stage": "output",
  "progress": 1.0,
  "error": null,
  "occurred_at": "2026-11-12T10:15:00Z"
}
```

Hlavičky: `X-QGen-Event`, `X-QGen-Delivery` (rovnaké pri opakovaní – na deduplikáciu),
`X-QGen-Timestamp` (Unix čas v sekundách), `X-QGen-Signature: sha256=<hex>`.

- Odpovedzte **2xx do 5 s**. Inak služba opakuje po 10 s, 60 s a 300 s, potom to vzdá.
  Stav úlohy sa tým nemení; `GET /v1/jobs/{id}` je záloha.
- Webhook spracujte rýchlo (napr. iba zaraďte job do fronty Laravelu, ktorý stiahne `/result`).
- Poradie nie je zaručené pri opakovaniach – rozhoduje `status`, nie poradie správ.

### Overenie podpisu v PHP

Podpis = HMAC-SHA256 z `"{timestamp}.{surové telo}"` tajným kľúčom `QGEN_WEBHOOK_SECRET`.
Použite **surové telo** requestu (nie znova zakódovaný JSON).

```php
public function handle(Request $request)
{
    $secret    = config('qgen.webhook_secret');
    $timestamp = $request->header('X-QGen-Timestamp', '');
    $signature = $request->header('X-QGen-Signature', '');
    $body      = $request->getContent();                       // surové bajty

    if (!ctype_digit($timestamp) || abs(time() - (int) $timestamp) > 300) {
        abort(401, 'stale webhook');                           // ochrana pred zopakovaním
    }
    $expected = 'sha256=' . hash_hmac('sha256', $timestamp . '.' . $body, $secret);
    if (!hash_equals($expected, $signature)) {
        abort(401, 'invalid signature');
    }

    $delivery = $request->header('X-QGen-Delivery');
    if (Cache::add('qgen-delivery:' . $delivery, true, now()->addDay()) === false) {
        return response()->noContent();                        // už spracované
    }

    $payload = $request->json()->all();
    ProcessQgenWebhook::dispatch($payload);                    // napr. po job.completed stiahne /result
    return response()->noContent();                            // 204
}
```

Route webhooku vynechajte z CSRF ochrany (`VerifyCsrfToken::$except`).

## 5. Výsledok (`/result`) a poradie otázok

Výsledok je presne formát units (JSON schéma `GET /v1/schema/lesson`), bez polí navyše:

```
units[]            hodiny v poradí, v akom sa majú učiť
  title            ≤ 255 znakov
  description      ≤ 4000 znakov
  questions[]      otázky hodiny v poradí:
                     1. podľa tematických celkov (prerekvizity sú vždy skôr),
                     2. v rámci hodiny najprv choice, potom open, nakoniec no_answer
    name, description, type (choice | open | no_answer),
    time_to_handle, time_to_explain (sekundy),
    hints[] {name, url}   url je HTML text nápovedy (<p>eko.pdf, strana 14</p>), nie webová adresa
    choices[] {text, is_correct}
```

- `choice`: aspoň 2 možnosti, aspoň 1 správna (môže ich byť viac).
- `open`: práve 1 položka v `choices` = vzorová odpoveď (`is_correct: true`).
- `no_answer`: `choices` je prázdne (esej, úvaha – nevyhodnocuje sa).
- Hodina bez otázky vo výsledku nie je. Ak nevznikla žiadna otázka, výsledok má jeden prázdny unit.
- `hints[].url` je už prečistené HTML (povolené iba `p, br, strong, em, ul, ol, li, code, pre, sub, sup`).

Poradie units a otázok zachovajte pri ukladaní (`learning_object_unit`), je to poradie výučby.

## 6. Chyby API

Každá chyba má tvar `{"error": {"code": "...", "message": "...", "details": {...}}}`.

| HTTP | `code` | Kedy |
|---|---|---|
| 400 | `invalid_options` | zlé `options`, nepovolený `callback_url`, OCR v obraze bez OCR, `D3` v sylabe |
| 401 | `unauthorized` | chýba alebo nesedí `X-API-Key` |
| 404 | `job_not_found` | úloha neexistuje alebo bola zmazaná |
| 409 | `result_not_ready` | `/result` alebo `/report` pri `queued`/`running` |
| 409 | `job_not_completed` | `/result` alebo `/report` pri `failed`/`cancelled` |
| 409 | `job_not_retryable` | retry po 24 h alebo v inom stave ako `failed` |
| 413 | `file_too_large` | priveľký súbor alebo priveľa strán/súborov |
| 415 | `unsupported_file` | nie je PDF, nedá sa otvoriť alebo je zaheslovaný |
| 422 | `validation_error` | chýba časť requestu (napr. `files`) |
| 429 | `too_many_jobs` | priveľa aktívnych úloh, hlavička `Retry-After` |
| 500 | `internal_error` | neočakávaná chyba služby |

## 7. Varovania, ktoré môže zobraziť učiteľovi

Varovania sú iba v reporte (`GET /v1/jobs/{id}/report`, pole `warnings[]` s `code` a
`message`). `message` je technická angličtina, preto zobrazte vlastný text podľa `code`:

| `code` | Navrhnutý text pre učiteľa |
|---|---|
| `lesson_count_reduced` | Dokument nemá dosť obsahu na zvolený počet hodín, počet hodín sa znížil. |
| `lessons_unbalanced` | Hodiny majú nerovnaký rozsah (niektorá je o viac ako polovicu dlhšia alebo kratšia). |
| `no_document_topics` | V dokumente sa nepodarilo nájsť témy, všetky otázky sú v jednej hodine. |
| `empty_lesson` | Niektorá hodina nemala žiadnu otázku a bola vynechaná. |
| `no_questions` | Z dokumentu sa nepodarilo vytvoriť žiadnu otázku. |
| `syllabus_fallback` | Sylabus sa nedal použiť, hodiny sú rozdelené podľa tém dokumentu. |
| `syllabus_plan_invalid` | Text sylabu sa nepodarilo previesť na hodiny, hodiny sú rozdelené podľa tém. |
| `syllabus_skipped_sections` | Časti dokumentu, ktoré sylabus nepokrýva, sú vynechané. |
| `syllabus_unknown_section` / `syllabus_duplicate_section` | Sylabus odkazuje na neexistujúcu alebo opakovanú časť; bola ignorovaná. |
| `syllabus_lesson_minutes_ignored` | Dĺžka hodiny v sylabe sa nepoužíva, hodiny sú podľa tém. |
| `prerequisite_order` | Poradie v sylabe nedodržiava nadväznosť učiva (niečo sa preberá skôr ako jeho základ). |
| `prerequisite_moved` | Niektoré učivo sa presunulo neskôr, aby nasledovalo po svojom základe. |
| `lesson_titles_failed` | Názvy hodín sa nepodarilo vytvoriť, použili sa náhradné. Odporúčame ich upraviť. |
| `field_too_long` | Niektoré otázky boli vynechané, lebo boli príliš dlhé. |

Ostatné kódy (`prerequisite_cycle`, `shortening_failed`, `invalid_hint`, `invalid_question`,
`unknown_bloom_level` a diagnostika pipeline) sú interné; učiteľovi ich netreba ukazovať.

## 8. Kontrolný zoznam integrácie

- [ ] `X-API-Key` pri všetkých volaniach okrem `/health`.
- [ ] `external_ref` pri každom POST; po timeoute POST zopakovať s rovnakým `external_ref`.
- [ ] Overenie podpisu, okno 300 s, deduplikácia podľa `X-QGen-Delivery`.
- [ ] Záložný polling `GET /v1/jobs/{id}`, ak webhook nepríde.
- [ ] Po uložení výsledku `DELETE /v1/jobs/{id}` (inak sa zmaže sám po 7 dňoch).
- [ ] `failed` → ponúknuť „Skúsiť znova“ (`POST /retry`) do `retryable_until`.
- [ ] Ukladať poradie units a otázok.
