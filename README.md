# AIDocGen - Deep Research Dossier Pipeline

**AIDocGen** est une plateforme avancée de génération de dossiers de recherche approfondie, propulsée par un ensemble de modèles LLM locaux (via Ollama). Elle automatise la création de rapports structurés, vérifiés et sourcés sur n'importe quel sujet complexe.

## 🚀 Fonctionnalités Clés

- **Pipeline de Recherche Multi-Étapes** :
  1. **Planning** : Décomposition du sujet en un sommaire détaillé à 3 niveaux.
  2. **Recherche Web** : Extraction de liens via DuckDuckGo, avec fallback automatique sur **SearxNG** et **Wikipedia**.
  3. **Construction du Corpus** : Téléchargement et nettoyage intelligent des pages web (BeautifulSoup).
  4. **Analyse de Preuves** : Extraction massive d'affirmations factuelles (claims).
  5. **Vérification (Fact-Checking)** : Validation de chaque affirmation par un modèle "Juge".
  6. **Rédaction Académique** : Rédaction itérative des chapitres basée sur les preuves validées.
- **Interface Professionnelle (v2.1)** :
  * Dashboard moderne avec Sidebar.
  * Suivi en temps réel des metrics système (CPU/GPU).
  * Journal d'exécution détaillé pour chaque tâche.
  * Éditeur de sommaire visuel avant lancement de la rédaction.
- **Multi-Serveurs & Multi-Modèles** :
  * Gérez plusieurs serveurs Ollama distants.
  * Choisissez dynamiquement vos modèles pour chaque rôle (Planner, Writer, Judge).
  * Bibliothèque complète de modèles open-source avec fonction "Pull" intégrée.
- **Exports Multi-Formats** :
  * **Markdown** : Pour une édition rapide.
  * **LaTeX** : Pour une mise en page de type thèse/livre.
  * **PDF** : Génération automatique du rapport final via TeX Live.

## 🛠 Architecture

- **Backend** : FastAPI (Python 3.11+)
- **Frontend** : React + Vite + TypeScript
- **Recherche** : DuckDuckGo API, SearxNG, MediaWiki (Wikipedia)
- **Moteur PDF** : TeX Live / pdflatex

## 📦 Installation

### Preréquis
- Python 3.11+
- Node.js & npm
- Ollama
- TeX Live (pour l'export PDF) : `apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra texlive-lang-french`

### Setup Backend
```bash
cd ollama-ensemble-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Setup Frontend
```bash
cd web-ui
npm install
npm run build
```

## 🚀 Lancement

1. Configurez vos modèles dans `ollama-ensemble-proxy/ensemble-proxy.env`.
2. Démarrez le serveur :
```bash
cd ollama-ensemble-proxy
./.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001 --env-file ensemble-proxy.env
```
3. Accédez à l'interface sur `http://localhost:8001`.

## 🛡 Sécurité & Confidentialité
AIDocGen est conçu pour fonctionner **100% localement** (hors recherche web). Vos documents et vos logs de réflexion ne quittent jamais vos serveurs.

---
Développé avec ❤️ pour la génération de connaissances structurées.
