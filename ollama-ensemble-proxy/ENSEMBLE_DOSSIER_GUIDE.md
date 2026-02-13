# Ensemble Dossier Approfondi - Guide Complet

Date de reference: 2026-02-07

## 1) Etat actuel de la machine

### 1.1 GPU detectes

- `NVIDIA GeForce RTX 5070 Ti` x2
- VRAM par GPU: `16303 MiB`
- Driver: `570.211.01`

### 1.2 Services actifs

- `ollama.service` (API Ollama locale `:11434`)
- `ollama-ensemble-proxy.service` (proxy OpenAI-compatible `:8001`)
- `open-webui.service` (UI `:4000`)
- `perplexica` (Docker, port publie `:3000`, Searx interne `:8080` dans le conteneur)

### 1.3 Ports utiles

- `4000`: Open-WebUI
- `8001`: Ollama Ensemble Proxy
- `11434`: Ollama API
- `3000`: Perplexica

## 2) Architecture fonctionnelle

Le mode `ensemble-dossier-approfondi` suit ce pipeline:

1. `planner`
2. `search`
3. `outline`
4. `corpus`
5. `ranking`
6. `claims`
7. `verification`
8. `writing`
9. assemblage final (`report.md` + `audit.json`)

Points structurants:

- zero fallback silencieux dans le moteur dossier
- erreurs explicites avec contexte d'etape
- timeout LLM long (run long accepte)
- redaction iterative "type livre" basee sur topics/sommaire
- JSON repair explicite (journalise) quand un modele sort du markdown au lieu de JSON

## 3) Modeles utilises actuellement

### 3.1 IDs exposes au client

- `ensemble-local`
- `ensemble-web-verified`
- `ensemble-code-local`
- `ensemble-code-web-verified`
- `ensemble-dossier-approfondi`

### 3.2 Mapping actuel (fichier `ensemble-proxy.env`)

General ensemble:

- panel: `mistral-small3.2:24b`, `gemma2:27b`, `qwen2.5:32b`
- synthese: `mistral-small3.2:24b`
- verification courte: `mistral-small3.2:24b`

Code ensemble:

- panel code: `qwen2.5-coder:32b`, `deepseek-coder:33b`, `starcoder2:15b-q8_0`
- synthese code: `qwen2.5-coder:32b`
- verification code: `qwen2.5-coder:32b`

Dossier approfondi:

- planner: `qwen2.5:32b`
- extract: `mistral-small3.2:24b`
- verify: `qwen2.5:32b`
- writer: `mistral-small3.2:24b`
- judge: `qwen2.5:32b`

### 3.3 Modeles installes localement (`ollama list`)

Exemples presents:

- `qwen2.5:32b`
- `qwen3:32b`
- `llama3.3:70b`
- `deepseek-r1:32b`
- `mistral-small3.2:24b`
- `qwen2.5-coder:32b`
- `deepseek-coder:33b`
- `starcoder2:15b-q8_0`

## 4) Parametres actuels et explication associee

## 4.1 Configuration complete active

Source: `ensemble-proxy.env`

```env
OLLAMA_BASE_URL=http://127.0.0.1:11434
ENSEMBLE_MODEL_ID=ensemble-local
ENSEMBLE_WEB_MODEL_ID=ensemble-web-verified
ENSEMBLE_CODE_MODEL_ID=ensemble-code-local
ENSEMBLE_CODE_WEB_MODEL_ID=ensemble-code-web-verified
ENSEMBLE_DOSSIER_MODEL_ID=ensemble-dossier-approfondi
ENSEMBLE_PANEL_MODELS=mistral-small3.2:24b,gemma2:27b,qwen2.5:32b
ENSEMBLE_SYNTH_MODEL=mistral-small3.2:24b
ENSEMBLE_VERIFY_MODEL=mistral-small3.2:24b
ENSEMBLE_VERIFY_MAX_TOKENS=450
ENSEMBLE_STRUCTURED_MODEL=mistral-small3.2:24b
ENSEMBLE_CODE_PANEL_MODELS=qwen2.5-coder:32b,deepseek-coder:33b,starcoder2:15b-q8_0
ENSEMBLE_CODE_SYNTH_MODEL=qwen2.5-coder:32b
ENSEMBLE_CODE_VERIFY_MODEL=qwen2.5-coder:32b
ENSEMBLE_CODE_STRUCTURED_MODEL=qwen2.5-coder:32b
ENSEMBLE_TIMEOUT_SECONDS=240
ENSEMBLE_MAX_PARALLEL=1
ENSEMBLE_WEB_MAX_QUERIES=2
ENSEMBLE_WEB_RESULTS_PER_QUERY=4
ENSEMBLE_WEB_MAX_SOURCES=8
ENSEMBLE_WEB_REGION=wt-wt
ENSEMBLE_WEB_SAFESEARCH=moderate
ENSEMBLE_WEB_TIMEOUT_SECONDS=20
ENSEMBLE_DOSSIER_DATA_DIR=/root/codex/ollama-ensemble-proxy/data/dossiers
ENSEMBLE_DOSSIER_PLANNER_MODEL=qwen2.5:32b
ENSEMBLE_DOSSIER_EXTRACT_MODEL=mistral-small3.2:24b
ENSEMBLE_DOSSIER_VERIFY_MODEL=qwen2.5:32b
ENSEMBLE_DOSSIER_WRITER_MODEL=mistral-small3.2:24b
ENSEMBLE_DOSSIER_JUDGE_MODEL=qwen2.5:32b
ENSEMBLE_SEARXNG_BASE_URL=http://172.17.0.2:8080
ENSEMBLE_DOSSIER_MAX_SUBQUESTIONS=6
ENSEMBLE_DOSSIER_MAX_LINKS_PER_SUBQUESTION=240
ENSEMBLE_DOSSIER_FETCH_LIMIT_PER_SUBQUESTION=70
ENSEMBLE_DOSSIER_SHORTLIST_PER_SUBQUESTION=18
ENSEMBLE_DOSSIER_QUERY_VARIANTS=6
ENSEMBLE_DOSSIER_PER_QUERY_RESULTS=80
ENSEMBLE_DOSSIER_WEB_TIMEOUT_SECONDS=25
ENSEMBLE_DOSSIER_WEB_REGION=wt-wt
ENSEMBLE_DOSSIER_WEB_SAFESEARCH=moderate
ENSEMBLE_DOSSIER_CHUNK_SIZE=1400
ENSEMBLE_DOSSIER_CHUNK_OVERLAP=180
ENSEMBLE_DOSSIER_MAX_CLAIMS_PER_SOURCE=6
ENSEMBLE_DOSSIER_LLM_TIMEOUT_SECONDS=7200
ENSEMBLE_DOSSIER_LLM_RETRY_ATTEMPTS=4
ENSEMBLE_DOSSIER_MAX_PARALLEL_FETCH=8
ENSEMBLE_DOSSIER_MAX_PARALLEL_LLM=2
ENSEMBLE_DOSSIER_WRITER_ITERATIONS=8
ENSEMBLE_DOSSIER_WRITER_BATCH_CLAIMS=24
ENSEMBLE_DOSSIER_MIN_WORDS_PER_SECTION=1400
ENSEMBLE_DOSSIER_TARGET_WORDS_PER_SECTION=2600
```

## 4.2 Explication variable par variable

### A) Routage et profils

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Endpoint Ollama source | Si faux, tout le proxy tombe |
| `ENSEMBLE_MODEL_ID` | `ensemble-local` | ID "chat local" | Nom visible Open-WebUI/API |
| `ENSEMBLE_WEB_MODEL_ID` | `ensemble-web-verified` | ID "chat + web" | Nom visible Open-WebUI/API |
| `ENSEMBLE_CODE_MODEL_ID` | `ensemble-code-local` | ID "code local" | Nom visible Open-WebUI/API |
| `ENSEMBLE_CODE_WEB_MODEL_ID` | `ensemble-code-web-verified` | ID "code + web" | Nom visible Open-WebUI/API |
| `ENSEMBLE_DOSSIER_MODEL_ID` | `ensemble-dossier-approfondi` | ID du workflow long | Cle d'entree dossier |

### B) Ensemble general (hors dossier)

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_PANEL_MODELS` | `mistral-small3.2:24b,gemma2:27b,qwen2.5:32b` | Panel multi-modeles | Qualite/diversite vs latence |
| `ENSEMBLE_SYNTH_MODEL` | `mistral-small3.2:24b` | Modele de synthese finale | Style/precision finale |
| `ENSEMBLE_VERIFY_MODEL` | `mistral-small3.2:24b` | Verif rapide | Controle factualite court |
| `ENSEMBLE_VERIFY_MAX_TOKENS` | `450` | Taille max de verif | Cout et granularite |
| `ENSEMBLE_STRUCTURED_MODEL` | `mistral-small3.2:24b` | Requetes structured/tool | Compatibilite Perplexica |
| `ENSEMBLE_TIMEOUT_SECONDS` | `240` | Timeout global chat | Limite requetes non-dossier |
| `ENSEMBLE_MAX_PARALLEL` | `1` | Parallelisme panel | Debit vs VRAM |

### C) Mode code (hors dossier)

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_CODE_PANEL_MODELS` | `qwen2.5-coder:32b,deepseek-coder:33b,starcoder2:15b-q8_0` | Panel code | Qualite patch/codegen |
| `ENSEMBLE_CODE_SYNTH_MODEL` | `qwen2.5-coder:32b` | Synthese code | Qualite fusion code |
| `ENSEMBLE_CODE_VERIFY_MODEL` | `qwen2.5-coder:32b` | Verif code | Rigueur review |
| `ENSEMBLE_CODE_STRUCTURED_MODEL` | `qwen2.5-coder:32b` | Structured code | Tool/function calls |

### D) Web rapide (hors dossier)

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_WEB_MAX_QUERIES` | `2` | Nb requetes web rapides | Couverture vs latence |
| `ENSEMBLE_WEB_RESULTS_PER_QUERY` | `4` | Resultats/requete | Couverture web |
| `ENSEMBLE_WEB_MAX_SOURCES` | `8` | Sources max fusion | Qualite vs cout |
| `ENSEMBLE_WEB_REGION` | `wt-wt` | Region DDG | Relevance geographique |
| `ENSEMBLE_WEB_SAFESEARCH` | `moderate` | Filtre contenu | Securite/relevance |
| `ENSEMBLE_WEB_TIMEOUT_SECONDS` | `20` | Timeout recherche | Robustesse vs attente |

### E) Dossier - stockage et modeles de role

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_DOSSIER_DATA_DIR` | `/root/codex/ollama-ensemble-proxy/data/dossiers` | Checkpoints/run data | Resume/forensic |
| `ENSEMBLE_DOSSIER_PLANNER_MODEL` | `qwen2.5:32b` | Decomposition question | Qualite plan/sous-questions |
| `ENSEMBLE_DOSSIER_EXTRACT_MODEL` | `mistral-small3.2:24b` | Extraction claims | Qualite claims JSON |
| `ENSEMBLE_DOSSIER_VERIFY_MODEL` | `qwen2.5:32b` | Verif claims | Statuts ACCEPTED/UNCERTAIN/REJECTED |
| `ENSEMBLE_DOSSIER_WRITER_MODEL` | `mistral-small3.2:24b` | Redaction longue | Qualite narrative |
| `ENSEMBLE_DOSSIER_JUDGE_MODEL` | `qwen2.5:32b` | Rerank/outline/judge | Rigueur scoring et sommaire |
| `ENSEMBLE_SEARXNG_BASE_URL` | `http://172.17.0.2:8080` | Moteur web principal | Evite rate-limit DDG, collecte massive |

### F) Dossier - exploration web massive

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_DOSSIER_MAX_SUBQUESTIONS` | `6` | Nb sous-questions max | Taille du dossier |
| `ENSEMBLE_DOSSIER_MAX_LINKS_PER_SUBQUESTION` | `240` | Limite brute liens | Couverture et temps |
| `ENSEMBLE_DOSSIER_FETCH_LIMIT_PER_SUBQUESTION` | `70` | Nb liens effectivement telecharges | Cout corpus |
| `ENSEMBLE_DOSSIER_SHORTLIST_PER_SUBQUESTION` | `18` | Sources finalistes par SQ | Qualite vs cout LLM |
| `ENSEMBLE_DOSSIER_QUERY_VARIANTS` | `6` | Variantes de requetes par SQ | Recall web |
| `ENSEMBLE_DOSSIER_PER_QUERY_RESULTS` | `80` | Resultats max par variante | Recall web |
| `ENSEMBLE_DOSSIER_WEB_TIMEOUT_SECONDS` | `25` | Timeout HTTP source | Robustesse fetch |
| `ENSEMBLE_DOSSIER_WEB_REGION` | `wt-wt` | Region moteur web | Relevance locale |
| `ENSEMBLE_DOSSIER_WEB_SAFESEARCH` | `moderate` | Filtre web | Hygiène source |

### G) Dossier - chunks, claims, parallelisme, long-run

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_DOSSIER_CHUNK_SIZE` | `1400` | Taille chunk texte | Granularite claims |
| `ENSEMBLE_DOSSIER_CHUNK_OVERLAP` | `180` | Recouvrement chunks | Evite perte frontiere |
| `ENSEMBLE_DOSSIER_MAX_CLAIMS_PER_SOURCE` | `6` | Claims max/chunk source | Cout claims |
| `ENSEMBLE_DOSSIER_LLM_TIMEOUT_SECONDS` | `7200` | Timeout LLM long | Autorise runs 2h+ |
| `ENSEMBLE_DOSSIER_LLM_RETRY_ATTEMPTS` | `4` | Retries appel LLM | Robustesse |
| `ENSEMBLE_DOSSIER_MAX_PARALLEL_FETCH` | `8` | Fetch HTTP parallele | Vitesse corpus |
| `ENSEMBLE_DOSSIER_MAX_PARALLEL_LLM` | `2` | Jobs LLM paralleles | Goulot principal actuel |

### H) Dossier - redaction iterative type "livre"

| Variable | Valeur | Role | Impact |
|---|---|---|---|
| `ENSEMBLE_DOSSIER_WRITER_ITERATIONS` | `8` | Passes de redaction par topic | Profondeur |
| `ENSEMBLE_DOSSIER_WRITER_BATCH_CLAIMS` | `24` | Claims injectes par passe | Densite evidences |
| `ENSEMBLE_DOSSIER_MIN_WORDS_PER_SECTION` | `1400` | Planche minimum section | Taille finale dossier |
| `ENSEMBLE_DOSSIER_TARGET_WORDS_PER_SECTION` | `2600` | Cible section | Aspect multi-pages |

## 5) Utilisation - Open-WebUI et mode batch

## 5.1 Depuis Open-WebUI (port 4000)

Open-WebUI est deja demarre sur `http://<host>:4000`.

Pour brancher ce proxy:

1. Ajouter un provider OpenAI-compatible dans Open-WebUI
2. Base URL: `http://127.0.0.1:8001/v1` (ou URL reseau equivalence)
3. API key: selon policy Open-WebUI (souvent placeholder si proxy local)
4. Choisir le modele: `ensemble-dossier-approfondi`

## 5.2 Depuis curl (mode batch API)

Creer un run asynchrone:

```bash
curl -sS -X POST http://127.0.0.1:8001/v1/dossier/runs \
  -H 'Content-Type: application/json' \
  -d '{"question":"fait un topo detaille sur l imx8mp","background":true,"resume":false}'
```

Verifier le status:

```bash
RUN_ID=<run_id>
curl -sS "http://127.0.0.1:8001/v1/dossier/runs/$RUN_ID" | jq
```

Suivi en boucle:

```bash
while true; do
  curl -sS "http://127.0.0.1:8001/v1/dossier/runs/$RUN_ID" \
  | jq -r '"state=\(.state) stage=\(.stage) msg=\(.events[-1].message)"'
  sleep 5
done
```

Reprise d'un run:

```bash
curl -sS -X POST \
  "http://127.0.0.1:8001/v1/dossier/runs/$RUN_ID/resume" \
  -H 'Content-Type: application/json' \
  -d '{"background":true}'
```

Recuperer le rapport:

```bash
curl -sS "http://127.0.0.1:8001/v1/dossier/runs/$RUN_ID/report" \
  | jq -r '.report_markdown'
```

Recuperer l'audit complet:

```bash
curl -sS "http://127.0.0.1:8001/v1/dossier/runs/$RUN_ID/audit" | jq
```

## 6) Portage sur une autre machine

## 6.1 Prerequis

- Linux avec GPU NVIDIA
- `ollama` installe et actif
- Python 3.11+ (ou compatible)
- Docker si tu utilises Perplexica/Searx
- Ports ouverts (`4000`, `8001`, `11434`, optionnel `3000`)

## 6.2 Etapes minimales

1. Copier le dossier `ollama-ensemble-proxy`
2. Creer venv + installer deps:

```bash
cd /path/ollama-ensemble-proxy
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

3. Copier et adapter `ensemble-proxy.env`
4. Verifier acces moteur web (`ENSEMBLE_SEARXNG_BASE_URL`)
5. Tirer les modeles Ollama necessaires
6. Installer/activer `ollama-ensemble-proxy.service`
7. Configurer Open-WebUI vers `http://<host>:8001/v1`

## 6.3 Exemple service systemd proxy

Le service actuel:

```ini
[Unit]
Description=Ollama Ensemble OpenAI Proxy
After=network-online.target ollama.service
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/root/codex/ollama-ensemble-proxy
EnvironmentFile=/root/codex/ollama-ensemble-proxy/ensemble-proxy.env
ExecStart=/root/codex/ollama-ensemble-proxy/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8001
Restart=always
RestartSec=2
User=root
Group=root

[Install]
WantedBy=multi-user.target
```

Open-WebUI actuel (port 4000):

```ini
[Service]
ExecStart=/root/venv/bin/open-webui serve --host 0.0.0.0 --port 4000
```

## 7) Comment modifier les modeles

## 7.1 Changer les modeles dans l'env

Modifier `ensemble-proxy.env`, ex:

```env
ENSEMBLE_DOSSIER_PLANNER_MODEL=qwen2.5:72b
ENSEMBLE_DOSSIER_EXTRACT_MODEL=qwen3:32b
ENSEMBLE_DOSSIER_VERIFY_MODEL=deepseek-r1:70b
ENSEMBLE_DOSSIER_WRITER_MODEL=llama3.3:70b
ENSEMBLE_DOSSIER_JUDGE_MODEL=qwen2.5:72b
```

## 7.2 S'assurer que les modeles existent

```bash
ollama pull qwen2.5:72b
ollama pull qwen3:32b
ollama pull deepseek-r1:70b
ollama pull llama3.3:70b
```

## 7.3 Redemarrer

```bash
systemctl restart ollama-ensemble-proxy.service
```

Verifier:

```bash
curl -sS http://127.0.0.1:8001/v1/models | jq
```

## 8) Gain de temps et tuning (cas grosse machine, ex 6x H100)

## 8.1 Ou est le goulot aujourd'hui

Sur un run reel, l'etape `claims` domine souvent le temps.

Observation locale (run `run-1770473320-f93d89069b`):

- de `5/108` a `55/108` sources en ~3922s
- ~`78 s` par source en moyenne (ordre de grandeur)

Principales causes:

- extraction claims = nombreux appels LLM
- verification ensuite sur beaucoup de claims
- parallelisme LLM volontairement bas (`ENSEMBLE_DOSSIER_MAX_PARALLEL_LLM=2`)

## 8.2 Levier prioritaire sur grosse machine

Augmenter en priorite:

1. `ENSEMBLE_DOSSIER_MAX_PARALLEL_LLM` (ex: `8` a `16`)
2. modeles plus rapides/plus robustes JSON pour `extract`/`verify`
3. `ENSEMBLE_DOSSIER_MAX_PARALLEL_FETCH` (ex: `16` ou `24`)

Conserver:

- timeout long (`ENSEMBLE_DOSSIER_LLM_TIMEOUT_SECONDS`)
- retries (`ENSEMBLE_DOSSIER_LLM_RETRY_ATTEMPTS`)
- controle strict JSON/claims

## 8.3 Reglages Ollama serveur recommandes (multi-GPU)

Exemples a tester cote `ollama.service`:

```bash
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=12
OLLAMA_MAX_QUEUE=1024
OLLAMA_FLASH_ATTENTION=1
OLLAMA_KV_CACHE_TYPE=q8_0
```

Notes:

- `OLLAMA_NUM_PARALLEL` augmente le parallelisme par modele, mais augmente aussi la RAM/VRAM contextuelle
- sur multi-GPU, Ollama place un modele sur 1 GPU si ca rentre, sinon il spread sur plusieurs GPUs

## 8.4 Ordre de grandeur du gain 6x H100

Sans engagement contractuel (depends workload/modeles/quantization):

- gain `x4` a `x12` sur temps total run est plausible
- sur etapes purement LLM (`claims`, `verification`, `writing`), gain peut etre encore superieur si parallelisme bien regle
- etapes reseau (`search`, `fetch`) ne scaleront pas comme le compute

## 9) Quels modeles pertinents sur grosse machine

## 9.1 Profil "qualite solide + debit"

- planner/judge: `qwen2.5:72b`
- extract: `qwen3:32b`
- verify: `deepseek-r1:70b`
- writer: `llama3.3:70b`

Pourquoi:

- `qwen2.5:72b` robuste en structuration/planification
- `qwen3:32b` bon compromis debit/qualite extraction
- `deepseek-r1:70b` fort pour raisonnement de verif
- `llama3.3:70b` redaction longue + contexte 128K

## 9.2 Profil "max reasoning"

- planner/judge/verify: `deepseek-r1:70b` (ou mix `qwen2.5:72b` + `deepseek-r1:70b`)
- extract: `qwen3:32b` ou `qwen2.5:72b`
- writer: `qwen2.5:72b` ou `llama3.3:70b`

## 9.3 Profil "extreme" (a reserver)

- `deepseek-r1:671b` ou `qwen3:235b`

A garder en tete:

- tres lourd en VRAM et en cout temps
- utile surtout si tu as besoin de tres haute profondeur de raisonnement
- pas toujours optimal pour le debit d'un pipeline multi-etapes

## 10) Bonnes pratiques operationnelles

1. Garder `search` sur Searx interne (pas DDG direct) pour eviter rate limit.
2. Garder `resume=true` pour reprendre aux checkpoints.
3. Sur run long, monitorer avec `/v1/dossier/runs/{run_id}` plutot que relancer.
4. Ne pas monter trop vite `MAX_PARALLEL_LLM` sans surveiller VRAM/OOM.
5. Verifier la qualite JSON (claims/outline/writer) avant de pousser les modeles "creatifs".

## 11) Checklist de changement de profil

1. Modifier les variables modeles dans `ensemble-proxy.env`.
2. `ollama pull` des nouveaux modeles.
3. Redemarrer `ollama-ensemble-proxy.service`.
4. Lancer un run test court.
5. Comparer:
   - temps par etape
   - nb claims ACCEPTED/UNCERTAIN/REJECTED
   - longueur/qualite dossier
6. Verrouiller un profil stable, puis seulement ensuite augmenter parallelisme.

## 12) Sources officielles utiles

- Ollama library (modeles et tailles):
  - https://ollama.com/library/qwen2.5
  - https://ollama.com/library/llama3.3
  - https://ollama.com/library/deepseek-r1
  - https://ollama.com/library/qwen3
- Ollama FAQ (concurrency, multi-GPU, tuning):
  - https://docs.ollama.com/faq
- Open-WebUI (OpenAI-compatible servers):
  - https://docs.openwebui.com/getting-started/quick-start/starting-with-openai-compatible/
- NVIDIA H100:
  - https://www.nvidia.com/en-us/data-center/h100/

