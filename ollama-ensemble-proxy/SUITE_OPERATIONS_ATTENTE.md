# Suite Des Operations (En Attente)

Date: 2026-02-07
Statut global: `EN ATTENTE`

## 1) Protection et exposition des services

- [ ] Mettre en place `nginx` comme reverse proxy unique.
- [ ] Exposer uniquement `80/443` publiquement.
- [ ] Bloquer l'exposition directe des ports techniques (`11434`, `8001`, `4000`, `3000` si inutile).
- [ ] Activer HTTPS (Let's Encrypt) pour:
  - `chat.electrosens.fr` (Open-WebUI)
  - `ollama.electrosens.fr` (acces modeles, verrouille)
  - `worddev.electrosens.fr` (WordPress dev)
- [ ] Durcir SSH:
  - desactiver root login distant
  - cles SSH uniquement
  - fail2ban
- [ ] Mettre en place une sauvegarde serveur sur `BorgBase` (systeme + donnees + bases) avec tests de restauration periodiques.
- [ ] Supprimer Perplexica du serveur (container, image, configuration associee) apres verification qu'il n'est plus requis.

## 2) Installation WordPress de dev

- [ ] Installer un WordPress de dev sur `worddev.electrosens.fr`.
- [ ] Aligner versions PHP/MySQL et config serveur.
- [ ] Installer les memes plugins/themes qu'en prod.
- [ ] Configurer sauvegarde locale (DB + fichiers) sur le dev.

## 3) Import du site prod vers dev

- [ ] Exporter prod (DB + `wp-content`).
- [ ] Importer sur dev.
- [ ] Faire search/replace des URLs et verifier permaliens.
- [ ] Verifier plugins, medias, formulaires, cache.
- [ ] Valider l'environnement dev comme miroir de prod.

## 4) Pipeline IA code + recherche (mode identique)

- [ ] Definir un profil "code" identique/stable pour les travaux de dev.
- [ ] Verifier la possibilite de parametrer les plugins/fonctions dans Open-WebUI.
- [ ] Documenter la methode:
  - generation dossier/recherche
  - generation/patch code
  - verification avant publication

## 5) Bascule Ollama vers RunPod (H100)

- [ ] Choisir un RunPod cible (nombre de H100).
- [ ] Etablir l'acces SSH RunPod.
- [ ] Installer Ollama sur RunPod.
- [ ] Installer les modeles choisis sur RunPod.
- [ ] Pointer le proxy local vers l'Ollama RunPod (URL securisee).
- [ ] Tester:
  - recherches longues
  - generation code
  - stabilite/timeouts/retries

## 6) Gestion batch et file d'attente

- [ ] Definir workflow batch (lancer, suspendre, reprendre, annuler).
- [ ] Mettre en place suivi des runs et resolution des batches.
- [ ] Standardiser l'audit/resultat pour chaque batch.

## 7) Console de gestion des batches

- [ ] Creer une console de supervision des batches.
- [ ] Afficher 3 vues de liste:
- [ ] `En cours`
- [ ] `Realises`
- [ ] `En attente`
- [ ] Afficher pour chaque batch:
- [ ] `run_id`
- [ ] type/profil
- [ ] date de creation
- [ ] etape courante
- [ ] progression globale
- [ ] Afficher le detail des etapes avec code couleur:
- [ ] `Vert` = etape terminee
- [ ] `Orange` = etape en cours
- [ ] `Gris` = etape a faire
- [ ] Pour l'etape en cours, afficher une progression chiffrable:
- [ ] exemple `40/120`
- [ ] + message de progression (ex: `Claim extraction 40/120 sources`)
- [ ] Ajouter un rafraichissement auto (polling) configurable.
- [ ] Ajouter actions de pilotage:
- [ ] relancer/reprendre
- [ ] annuler
- [ ] ouvrir rapport
- [ ] ouvrir audit JSON

## 8) Constitution du site web dev par lots

- [ ] Alimenter WordPress dev avec contenus valides issus du pipeline.
- [ ] Relecture humaine et validation editoriale.
- [ ] Ajuster structure (pages/CPT/champs meta/sources).
- [ ] Preparer procedure de promotion dev -> prod.

## Notes

- Ce document sert de backlog "mis en attente".
- Aucune execution automatique n'est lancee depuis ce fichier.
