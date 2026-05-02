# Guide utilisateur — Conversion PDF → Markdown pour Dify

Pipeline de conversion de documents (PDF, DOCX converti en PDF, .pages converti en PDF) vers du Markdown propre, optimisé pour l'ingestion dans une base de connaissance Dify.

Utilise **Ollama Cloud** (abonnement) — coût nul une fois le forfait souscrit.

---

## Prérequis

- **Ollama** installé et connecté à Ollama Cloud (`ollama serve` tourne sur `localhost:11434`)
- **poppler-utils** pour `pdftotext`, `pdftoppm`, `pdfinfo`, `pdfimages`
- **Python 3** (stdlib uniquement, aucune dépendance)

Vérification rapide :
```bash
curl -s http://localhost:11434/api/tags | python3 -m json.tool | head -20
which pdftotext pdftoppm pdfinfo pdfimages
```

---

## Quelle méthode choisir ?

Avant tout, diagnostiquer le PDF pour choisir la bonne méthode.

### 1. Diagnostic rapide

```bash
pdfinfo fichier.pdf | grep -E "^(Pages|File size)"
pdftotext fichier.pdf - | wc -c      # volume de texte extractible
pdfimages -list fichier.pdf | tail -n +3 | awk '$4>500 && $5>500 {print $1}' | sort -u
# ↑ pages contenant des images >500×500 px (vraies images, pas emojis iOS)
```

Interpréter :
- **chars/page > 1500** et **0 image réelle** → PDF texte pur → script `convert_pdf_md.py` (+ `process_huge.py` si > 100 pages)
- **Beaucoup de pages avec images >500 px** → screenshots ou scans → script `process_vision.py`
- **Mixte** → cas par cas

### 2. Arbre de décision

```
        ┌─────────────────────────────────────┐
        │ Le PDF contient-il des screenshots  │
        │ ou scans à transcrire ?             │
        └────────┬────────────────┬───────────┘
                 │ OUI            │ NON
                 ▼                ▼
       process_vision.py      Le texte fait-il
       (vision model          > ~40 KB ?
       sur chaque page)              │
                           ┌─────────┴─────────┐
                           │ NON               │ OUI
                           ▼                   ▼
                   convert_pdf_md.py    process_huge.py
                   (1 appel LLM)        (chunks parallèles)
```

---

## Les 4 scripts

### `convert_pdf_md.py` — conversion simple (un seul appel LLM)

Pour les PDF **texte natif** courts (< ~40 KB de texte extrait, soit ~20 pages de notes).

```bash
# 1. Extraire le texte brut
pdftotext -layout "mon_doc.pdf" /tmp/mon_doc_raw.txt

# 2. Nettoyer + restructurer via LLM
python3 scripts/convert_pdf_md.py /tmp/mon_doc_raw.txt dify_md/mon_doc.md
```

Options :
```bash
python3 scripts/convert_pdf_md.py <input.txt> <output.md> [modèle]
# modèle par défaut : glm-5.1:cloud
```

Le prompt est dans `scripts/prompt_md.txt` — éditer si besoin (corrections d'artefacts spécifiques, autres langues).

---

### `process_huge.py` — PDF texte volumineux (chunking parallèle)

Pour les PDF **texte natif longs** (> 40 KB de texte, typiquement > 100 pages).

Découpe le texte en chunks de ~40 KB (aux frontières de pages), envoie chaque chunk au LLM en parallèle (5 appels simultanés par défaut), puis concatène.

```bash
python3 scripts/process_huge.py "mon_doc.pdf" "dify_md/mon_doc.md" [modèle]
```

**Performance** : ~12 min pour 2 MB de texte (env. 800 pages) avec 5 appels parallèles.

Ajuster `PAR` dans le script (défaut 5) pour plus de parallélisme. Attention aux rate limits Ollama Cloud si multiples jobs simultanés.

---

### `process_vision.py` — PDF avec screenshots/scans

Pour les PDF contenant des **images informatives** (screenshots d'apps, scans de documents manuscrits, captures web). Chaque page est rendue en PNG puis transcrite par un vision model.

```bash
python3 scripts/process_vision.py "mon_doc.pdf" "dify_md/mon_doc.md" [modèle] [parallélisme]
# défauts : qwen3-vl:235b-cloud, 5 threads
```

**Performance** : ~20-30 s par page avec `qwen3-vl:235b-cloud`. Pour 86 pages en PAR=5 : ~7 min.

**Pour un PDF mixte** (certaines pages texte, d'autres scans) : passer le tout en vision donne un résultat homogène. Sinon, traiter par tranches de pages (voir `pdftoppm -f <first> -l <last>`).

---

### `convert_vision.py` — test sur 1 image

Pour debug / test : envoie une image à un vision model et sort la transcription.

```bash
# Rendre une page en image
pdftoppm -f 30 -l 30 -r 120 doc.pdf /tmp/page -png

# Tester la transcription
python3 scripts/convert_vision.py /tmp/page-30.png /tmp/result.md
```

Utile pour valider un prompt avant de lancer le batch complet.

---

## Modèles utilisés

Listés par `curl -s https://ollama.com/api/tags`. Les `:cloud` sont gratuits avec l'abonnement Ollama Cloud.

| Usage | Modèle recommandé | Alternatives |
|---|---|---|
| Restructuration Markdown (texte) | `glm-5.1:cloud` | `deepseek-v3.2`, `kimi-k2.6`, `gpt-oss:120b` |
| Transcription image/screenshot | `qwen3-vl:235b-cloud` | `gemini-3-flash-preview` (si vision confirmée) |

Pour changer de modèle : passer le nom en 3e argument aux scripts.

---

## Customisation du prompt

Le prompt par défaut (`scripts/prompt_md.txt`) est spécialisé pour du **russe** depuis **Apple Notes** (corrige `ĸ` → `к`, retire colonteaux iOS). Règles strictes : pas de paraphrase, pas de résumé, conserve le contenu dans la langue d'origine.

Pour un autre contexte (français, autre source) :
1. Copier `prompt_md.txt` en `prompt_fr.txt`
2. Adapter les instructions (enlever la partie sur les artefacts iOS, préciser la langue)
3. Modifier `convert_pdf_md.py` pour pointer dessus, ou `process_huge.py` (variable `SYSTEM_PROMPT`)

---

## Pipeline complet (exemple avec le dossier `Polina_docs`)

```bash
cd /home/michael/Polina_docs
mkdir -p dify_md

# 1. Fichiers texte pur courts
for f in "Декабрь" "Список дел" "Сегодня и сотка" "file1"; do
  pdftotext -layout "$f.pdf" /tmp/${f}_raw.txt
  python3 scripts/convert_pdf_md.py /tmp/${f}_raw.txt "dify_md/$f.md" &
done
wait

# 2. Fichiers texte volumineux (en parallèle)
python3 scripts/process_huge.py "Лидер таск.pdf"       "dify_md/Лидер таск.md" &
python3 scripts/process_huge.py "База данных.pdf"      "dify_md/База данных.md" &
python3 scripts/process_huge.py "База данных для чата.pdf" "dify_md/База данных для чата.md" &
wait

# 3. Fichier avec screenshots
python3 scripts/process_vision.py "Видение.pdf" "dify_md/Видение.md"

# 4. CSV copié tel quel (Dify gère nativement)
cp TickTick-backup-*.csv dify_md/
```

**Rate limits Ollama Cloud** : ne pas lancer plus de 10-15 appels concurrents en pratique. Si tu fais tourner plusieurs `process_huge.py` en parallèle, chacun consomme `PAR=5` slots.

---

## Ingestion dans Dify

1. **Dify → Knowledge → Create** (nouveau knowledge base)
2. Glisser les fichiers de `dify_md/` dans la zone d'upload
3. Paramètres recommandés :
   - **Chunking** : `Parent-Child` (meilleur retrieval sur gros docs)
   - **Embedding** : modèle multilingue — `bge-m3` ou `nomic-embed-text` (dispo en local via Ollama)
   - **Retrieval** : `Hybrid search` (vector + full-text) — utile pour retrouver des noms propres russes

---

## Dépannage

**Les `print()` Python n'apparaissent pas en temps réel** → stdout bufferisé en background. Lancer avec `python3 -u` pour l'unbuffered output.

**"Connection refused" sur localhost:11434** → vérifier que `ollama serve` tourne. Le démarrer si besoin : `ollama serve &` (ou via systemd selon la config).

**Timeout sur un gros chunk** → augmenter `num_ctx` dans `process_huge.py` (défaut 32768) si le modèle le supporte, ou baisser `TARGET_CHUNK_BYTES`.

**Résultats tronqués** → le modèle a peut-être atteint sa limite de tokens en sortie. Vérifier `eval_count` dans les logs. Baisser `TARGET_CHUNK_BYTES` à 20000-25000.

**Contenu manquant/halluciné** → le prompt demande explicitement de ne pas paraphraser, mais en cas de doute, comparer la taille in/out : l'output doit être ~90-100% de la taille de l'input (en bytes UTF-8). Si < 70%, le modèle a résumé — renforcer le prompt.

**Duplications dans l'output** → vérifier la source : `pdftotext doc.pdf - | grep -c "phrase dupliquée"`. Si présent 2× dans le brut, c'est un artefact du PDF d'origine, pas du pipeline.

---

## Structure du dossier

```
Polina_docs/
├── *.pdf                    # sources
├── *.csv                    # gardé tel quel pour Dify
├── dify_md/                 # sortie prête pour Dify
│   ├── *.md
│   └── *.csv
└── scripts/
    ├── convert_pdf_md.py    # 1 fichier, 1 appel LLM
    ├── process_huge.py      # chunking parallèle
    ├── process_vision.py    # transcription page par page
    ├── convert_vision.py    # 1 image (test/debug)
    ├── prompt_md.txt        # prompt système
    └── USER_GUIDE.md        # ce fichier
```
