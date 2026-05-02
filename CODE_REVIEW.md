# AIDocGen - Revue de Code Complète

**Date**: 15 Février 2026  
**Version analysée**: État actuel du dépôt

---

## 📊 Vue d'ensemble

| Fichier | Lignes | Complexité | Priorité |
|---------|--------|------------|----------|
| `dossier_engine.py` | 5720 | 🔴 Très élevée | Haute |
| `app.py` | 253 | 🟡 Moyenne | Moyenne |
| `App.tsx` | 418 | 🟡 Moyenne | Moyenne |
| `core/writer.py` | 191 | 🟢 Faible | Basse |
| `core/research.py` | 227 | 🟢 Faible | Basse |
| `core/config.py` | 121 | 🟢 Faible | Basse |
| `core/llm.py` | 106 | 🟢 Faible | Basse |
| `core/auth.py` | 81 | 🟢 Faible | Basse |

---

## 🔴 Critique - À corriger immédiatement

### 1. Sécurité: Secret JWT Hardcodé
**Fichier**: `core/auth.py:12`
```python
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "73f1e8a9c2b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0")
```
**Problème**: Clé secrète par défaut exposée dans le code source.  
**Impact**: Quiconque a accès au code peut forger des tokens JWT valides.  
**Recommandation**: Lever une erreur si `JWT_SECRET_KEY` n'est pas défini en production.

---

### 2. Sécurité: Mot de passe par défaut "admin/admin"
**Fichier**: `core/auth.py:24-30`
```python
admin_user = {
    "username": "admin",
    "hashed_password": pwd_context.hash("admin"),
    "role": "admin",
    "id": "u-admin"
}
```
**Problème**: Identifiants par défaut triviaux.  
**Impact**: Vulnérabilité en production si non changé.  
**Recommandation**: Exiger un changement de mot de passe au premier démarrage ou générer un mot de passe aléatoire.

---

### 3. Gestion d'erreurs silencieuse
**Fichier**: `core/research.py:28-30`
```python
except Exception:
    pass
```
**Problème**: Les exceptions sont ignorées sans logging.  
**Impact**: Impossible de diagnostiquer les problèmes de recherche web.  
**Recommandation**: Logger les erreurs avec `logging.exception()`.

---

### 4. Injection de dépendances manquante
**Fichier**: `app.py:50-63`
```python
def _get_engine(ollama_url: str = None, models: dict = None):
    config = DossierConfig.from_env()
    # ... mutation directe de la config
```
**Problème**: La fonction modifie la configuration globale pour chaque requête.  
**Impact**: Risque de race conditions en cas de requêtes concurrentes.  
**Recommandation**: Créer une nouvelle instance de config par requête.

---

## 🟠 Important - À corriger à court terme

### 5. Memory Leak potentiel
**Fichier**: `app.py:31`
```python
_DOSSIER_TASKS: dict[str, asyncio.Task] = {}
```
**Problème**: Les tâches terminées ne sont jamais nettoyées du dictionnaire.  
**Impact**: Fuite mémoire sur le long terme.  
**Recommandation**: Implémenter un nettoyage périodique des tâches terminées.

---

### 6. Variables TypeScript non typées
**Fichier**: `App.tsx` (multiple)
```typescript
const [planner, setPlanner] = useState<any>(null);
const [debug, setDebug] = useState<any>(null);
```
**Problème**: Utilisation excessive de `any`.  
**Impact**: Perte des avantages de TypeScript, bugs potentiels.  
**Recommandation**: Définir des interfaces explicites pour `Planner`, `DebugInfo`, etc.

---

### 7. Requêtes API sans gestion d'erreur
**Fichier**: `App.tsx` (multiple)
```typescript
axios.post(`/v1/dossier/runs/${run.run_id}/reset`)
```
**Problème**: Pas de `.catch()` ni de feedback utilisateur en cas d'échec.  
**Impact**: L'utilisateur ne sait pas si l'action a réussi ou échoué.  
**Recommandation**: Ajouter une gestion d'erreur avec notification.

---

### 8. Regex fragile pour parsing Markdown
**Fichier**: `core/writer.py:26-27`
```python
party_regex = r'^#{1,2}\s*(?:Partie|Part)\s*([IVX\d]+)[\s\:\-–—]*\s*(.*?)(?:\s*#|$)'
chap_regex = r'^#{1,4}\s*(?:Chapitre|Chapter)\s*[\d\.]*\s*[\:\-\.\s—]*\s*(.*?)(?:\s*#|$)'
```
**Problème**: Ne gère pas tous les formats (numéros romains sans espace, variantes typographiques).  
**Impact**: Échec du parsing sur certains plans.  
**Recommandation**: Étendre les regex ou utiliser un parser Markdown dédié.

---

### 9. Timeout non configurable pour les requêtes HTTP
**Fichier**: `core/research.py:188`
```python
async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
```
**Problème**: Le timeout est configurable mais pas les retry.  
**Impact**: Échec définitif sur une erreur réseau temporaire.  
**Recommandation**: Implémenter une stratégie de retry avec backoff exponentiel.

---

### 10. CORS ouvert à tous les origins
**Fichier**: `app.py:22`
```python
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)
```
**Problème**: CORS trop permissif.  
**Impact**: Vulnérabilité CSRF potentielle.  
**Recommandation**: Restreindre aux domaines connus en production.

---

## 🟡 Moyenne - Améliorations suggérées

### 11. Fichier monolithique `dossier_engine.py`
**Fichier**: `dossier_engine.py` (5720 lignes)  
**Problème**: Trop grand, trop de responsabilités.  
**Recommandation**: Diviser en modules séparés:
- `planner.py`
- `searcher.py`
- `writer.py`
- `verifier.py`
- `assembler.py`

---

### 12. Manque de tests unitaires
**Problème**: Aucun test unitaire dans le projet.  
**Impact**: Régressions difficiles à détecter.  
**Recommandation**: Ajouter des tests pour:
- `parse_markdown_outline()`
- `normalize_outline()`
- Authentification JWT
- Endpoints API critiques

---

### 13. Logging insuffisant
**Problème**: Utilisation de `print()` au lieu de logging structuré.  
**Exemples**:
- `dossier_engine.py`: `print(f"Warning: ...")`
- `core/llm.py`: `print(f"DEBUG: ...")`  
**Recommandation**: Configurer le module `logging` avec niveaux appropriés.

---

### 14. Gestion des états frontend
**Fichier**: `App.tsx`
```typescript
const [token, setToken] = useState(localStorage.getItem('token'));
// ...
if (token) axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
```
**Problème**: Mutation directe de `axios.defaults` dans le render.  
**Recommandation**: Utiliser un `useEffect` ou un intercepteur Axios.

---

### 15. Duplication de code dans App.tsx
**Fichier**: `App.tsx`
- `VisualPlanEditor` et `RunDetailPanel` ont du code similaire pour l'édition du plan.  
**Recommandation**: Extraire un composant `PlanEditor` réutilisable.

---

## 🟢 Mineur - Suggestions cosmétiques

### 16. Noms de variables non descriptifs
**Fichier**: `App.tsx`
```typescript
const [u, setU] = useState('');
const [p, setP] = useState('');
```
**Recommandation**: Utiliser `username`, `password`.

---

### 17. Styles inline dans React
**Fichier**: `App.tsx` (nombreux)  
**Problème**: Difficile à maintenir.  
**Recommandation**: Extraire vers `App.css` ou utiliser styled-components.

---

### 18. Commentaires manquants
**Problème**: Peu de documentation inline.  
**Recommandation**: Ajouter des docstrings pour les fonctions complexes.

---

### 19. Fichiers de test orphelins
**Fichier**: `test_*.py`, `bench_*.py` à la racine  
**Problème**: Fichiers de test mélangés avec le code principal.  
**Recommandation**: Déplacer vers un dossier `tests/`.

---

### 20. Configuration éparpillée
**Problème**: Configuration dans `ensemble-proxy.env`, `config.py`, et hardcodée.  
**Recommandation**: Centraliser et utiliser Pydantic Settings.

---

## 📈 Métriques de qualité

| Métrique | Valeur | Cible |
|----------|--------|-------|
| Couverture de tests | 0% | >80% |
| Lignes par fichier (max) | 5720 | <500 |
| Complexité cyclomatique | Élevée | <10/fonction |
| Dépendances vulnérables | À vérifier | 0 |
| Documentation | Minimale | Complète |

---

## ✅ Points positifs

1. **Architecture modulaire** : Séparation claire backend/frontend
2. **Type hints Python** : Utilisation de `dict[str, Any]` et `list[dict]`
3. **Async/await** : Utilisation correcte de l'asynchrone
4. **FastAPI** : Framework moderne et performant
5. **React + TypeScript** : Stack frontend solide
6. **Authentification JWT** : Implémentation correcte (à part le secret)
7. **Export multi-format** : Markdown, LaTeX, PDF
8. **Pipeline de recherche** : Fallback DDG → SearxNG → Wikipedia

---

## 🎯 Priorisation des corrections

| Priorité | Issue | Effort |
|----------|-------|--------|
| P0 | Secret JWT hardcodé | 1h |
| P0 | Mot de passe par défaut | 1h |
| P1 | Memory leak _DOSSIER_TASKS | 2h |
| P1 | Gestion d'erreurs silencieuse | 2h |
| P2 | Types TypeScript | 4h |
| P2 | Tests unitaires de base | 8h |
| P3 | Refactoring dossier_engine.py | 16h |

---

## 🔧 Recommandations d'architecture

1. **Ajouter une couche de validation** : Pydantic pour toutes les entrées API
2. **Implémenter un système de logging centralisé** : Structured logging avec rotation
3. **Ajouter des health checks** : `/health` et `/ready` endpoints
4. **Rate limiting** : Protéger l'API contre les abus
5. **Monitoring** : Intégrer Prometheus/Grafana ou similaire
6. **CI/CD** : Pipeline de tests automatiques avant merge

---

*Fin de la revue*
