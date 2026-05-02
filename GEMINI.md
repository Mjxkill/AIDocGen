# Ollama Ensemble Proxy - Spécifications Techniques et Exigences (Mandat Ingénieur)

## 1. Directives de Qualité Logicielle (Anti-Régression)
- **Stabilité de l'Interface** : Toute modification du Frontend doit préserver le menu latéral, la barre de progression, et le gestionnaire de serveurs.
- **Rigueur des Types** : Utilisation systématique de garde-fous pour éviter les crashs de rendu (Page Blanche).
- **Zéro Hardcoding de Modèles** : Interdiction formelle d'écrire des noms de modèles en dur. La liste doit être 100% dynamique via l'API.
- **Test de Transformation Unitaire (TTU)** : Toute modification d'un moteur de cristallisation doit être validée par un script de test utilisant des données réelles.

## 2. Exigences de l'Interface Utilisateur (UI/UX)
- **Transparence Totale** : Affichage du message d'événement le plus récent et onglet Debug (Thinking/Draft).
- **Édition du Plan** : Les titres des sous-sections doivent être modifiables via des champs `input`.
- **Indicateurs Système** : Affichage temps réel du CPU et des GPU dans le header.

## 3. Architecture du Pipeline
- **Shadow Drafting** : Planification en Markdown riche (Thinking activé) suivie d'une cristallisation JSON robuste.
- **Validation de Densité** : Rejet automatique de tout JSON ayant moins de 3 chapitres.

## 4. Checklist de Validation (V-Plan)
- [ ] Vérifier que les modèles sont chargés dynamiquement depuis le serveur sélectionné.
- [ ] Vérifier que la barre de progression affiche un message textuel précis.
- [ ] Vérifier que l'onglet Debug contient bien le texte "Thinking".
