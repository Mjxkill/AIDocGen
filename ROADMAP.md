# AIDocGen — Plan d'évolution

État au **2026-05-02**. À cocher au fur et à mesure.

Légende effort : `S` < 4h · `M` 0,5–2 jours · `L` > 2 jours · `XL` > 1 semaine.

---

## 🔴 Robustesse — à faire en premier

| # | Item | Effort | Statut | Notes |
|---|------|--------|--------|-------|
| 1 | **Job queue persistante** (Celery / Dramatiq + Redis) — survit aux redémarrages du proxy, scaling horizontal | L | ☐ | Migrer `_DOSSIER_TASKS / _PDF_TASKS / _VIDEO_TASKS / _AUDIO_TASKS / _WEB_TASKS / _AUDIOBOOK_TASKS` de asyncio in-process vers une vraie queue. Modèle Beat pour les schedules. |
| 2 | **Tests automatisés + CI** (pytest + GitHub Actions) | M | ☐ | Démarrer par : split markdown, _has_prose, parse_tags, ownership filtering. Couverture cible : pipeline critique. |
| 3 | **Logging structuré** (structlog → JSON → Loki) | S | ☐ | Remplacer les `print()`. Champs: `run_id, user, stage, duration_ms`. |
| 4 | **Healthchecks + métriques Prometheus** (`/health` détaillé + `/metrics`) + dashboard Grafana | M | ☐ | Surveiller: GPU/CPU, queue size, erreurs, durée par stage, coût cumulatif. |
| 5 | **Backups automatiques** (restic ou rclone vers S3/Backblaze, cron quotidien) | S | ☐ | Cibles: `data/dossiers`, `data/audiobook_jobs`, `data/video_jobs`, `data/audio_jobs`, `data/pdf_jobs`, `data/web_jobs`, `data/users.json`, `data/servers.json`. |
| 6 | **Quotas par utilisateur** (storage, compute, tokens cloud) | M | ☐ | Limites configurables: GB stockage, runs/jour, tokens DeepSeek/mois, requêtes Firecrawl. |

## 🟡 Sécurité

| # | Item | Effort | Statut | Notes |
|---|------|--------|--------|-------|
| 7 | **Secrets manager** (sops + age, ou Docker secrets, ou Vault) | M | ☐ | `OLLAMA_API_KEY`, `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `FIRECRAWL_API_KEY`, `JWT_SECRET_KEY`, `DIFY_*`. |
| 8 | **Forcer changement mdp admin/admin à la 1ère connexion** | S | ☐ | Flag `must_change_password` dans `users.json`. |
| 9 | **Rate limiting** par user (slowapi) | S | ☐ | Endpoints sensibles: `/dossier/runs`, `/web/agent`, `/audiobook/*`, `/auth/login`. |
| 10 | **2FA TOTP** (au moins admin) | M | ☐ | pyotp + QR code dans le profil. |
| 11 | **Audit log** (login, run launched, doc deleted, etc.) | S | ☐ | Append-only `data/audit.log`. Vue admin. |

## 🟢 UX / fonctionnalités utilisateur

| # | Item | Effort | Statut | Notes |
|---|------|--------|--------|-------|
| 12 | **Notifications email + webhook** sur fin de job | M | ☐ | SMTP via env. Préférence par user (immédiat / digest / off). Webhook URL custom par user. |
| 13 | **Recherche full-text** dans dossiers terminés | M | ☐ | Meilisearch local ou Postgres FTS sur `report.md`. |
| 14 | **Diff entre deux runs** | M | ☐ | UI side-by-side ou wikidiff sur les sections. |
| 15 | **Cost tracker par job + agrégat user/mois** | M | ☐ | Tracker calls: Firecrawl pages, DeepSeek tokens, OpenAI TTS chars, etc. Tarifs en config. |
| 16 | **Reset password par email + invitation lien email** | S | ☐ | Token signé HMAC à durée limitée. |
| 17 | **Code-splitting frontend** (lazy-load Tools/Wiki/Users) | S | ☐ | Vite + React.lazy. |

## 🔵 Pro-grade (ambitieux, grand impact)

| # | Item | Effort | Statut | Notes |
|---|------|--------|--------|-------|
| 18 | **Espaces / projets** + partage entre users | L | ☐ | Group dossiers, audiobooks, web jobs par projet. ACL. |
| 19 | **API publique documentée** + clés API par user | M | ☐ | OpenAPI déjà auto-généré ; à exposer + générer/révoquer des clés. |
| 20 | **Plugins / hooks** (script custom à la fin d'un job) | L | ☐ | Webhook + sandbox d'exécution Python. |
| 21 | **RGPD : export + suppression user** | M | ☐ | Bouton « télécharger mes données » (ZIP), bouton « supprimer mon compte ». Email de confirmation. |
| 22 | **SSO** (Google/GitHub/Microsoft) | M | ☐ | OAuth2 + auto-provisioning user. |
| 23 | **Persistance multi-machines** : Postgres + S3/MinIO | XL | ☐ | Migration `users.json` → table users, `data/dossiers` → S3. Permet load balancer. |
| 24 | **Versioning prompts** (planner/writer/audiobook) | M | ☐ | Schema dans Postgres ou git. A/B test entre versions. |

## 💎 Polish

| # | Item | Effort | Statut | Notes |
|---|------|--------|--------|-------|
| 25 | **Skeletons de chargement** + animations | S | ☐ | Composants `<Skeleton />` pour run-grid, jobs lists, modals. |
| 26 | **Thème clair / sombre** | S | ☐ | Toggle dans la sidebar. CSS vars déjà presque OK. |
| 27 | **Audit accessibilité** (ARIA, contraste, nav clavier) | M | ☐ | Tester avec axe-core ou Lighthouse. |
| 28 | **Empty states soignés** sur chaque vue | S | ☐ | « Aucun dossier — voici comment commencer ». Mini-tutoriel. |

---

## Suivi

- Date de création : 2026-05-02
- Dernière revue :
- Cibles d'achèvement par phase :
  - 🔴 Robustesse : 
  - 🟡 Sécurité : 
  - 🟢 UX : 
  - 🔵 Pro-grade : 
  - 💎 Polish : 

Cocher en remplaçant `☐` par `☑` dans la colonne Statut quand fait.
