import json
from pathlib import Path

run_dir = Path("ollama-ensemble-proxy/data/dossiers/run-1770764739-19a9c99d58")
planner_path = run_dir / "planner.json"

with open(planner_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 1. Ajouter les nouvelles sous-questions
new_sqs = [
    {
        "id": "SQ_NEW_1",
        "question": "Quelles sont les obligations vaccinales actuelles pour les enfants (Calendrier, 11 vaccins en France/Europe) ?",
        "proof_criteria": ["Calendriers officiels de santé publique", "Législation en vigueur"],
        "search_queries": ["calendrier vaccinal 2024 2025", "obligation vaccinale enfants 11 vaccins", "vaccination mandatory schedule children"]
    },
    {
        "id": "SQ_NEW_2",
        "question": "Comment la qualité et la sécurité des vaccins modernes se comparent-elles à celles du passé ?",
        "proof_criteria": ["Rapports de pharmacovigilance comparatifs", "Évolution des processus de fabrication"],
        "search_queries": ["sécurité vaccins modernes vs anciens", "qualité fabrication vaccins évolution", "vaccine safety evolution"]
    },
    {
        "id": "SQ_NEW_3",
        "question": "Quel est le bilan spécifique des vaccins contre la Grippe et le Covid-19 (Efficacité, Technologie ARNm) ?",
        "proof_criteria": ["Données épidémiologiques récentes", "Études sur l'efficacité ARNm"],
        "search_queries": ["efficacité vaccin grippe vs covid", "technologie ARNm avantages inconvénients", "bilan vaccination covid-19"]
    }
]
data["sub_questions"].extend(new_sqs)

# 2. Créer le nouveau chapitre
new_chapter = {
    "id": "CH_NEW",
    "title": "La Vaccination Aujourd'hui : Obligations et Réalités",
    "goal": "Analyser le paysage vaccinal actuel, des obligations légales aux spécificités des vaccins modernes (Grippe, Covid).",
    "linked_sub_questions": ["SQ_NEW_1", "SQ_NEW_2", "SQ_NEW_3"],
    "status": "planned",
    "reason": "Ajouté manuellement pour couvrir l'actualité et les obligations légales.",
    "sub_sections": [
        "Obligations vaccinales et calendrier de l'enfant",
        "Qualité et sécurité : Comparaison Hier/Aujourd'hui",
        "Focus : Grippe et Covid-19 (ARNm)"
    ]
}

# 3. Insérer le chapitre en position 2 (index 2, donc 3ème chapitre)
data["master_outline"].insert(2, new_chapter)

# 4. Sauvegarder
with open(planner_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Planner updated successfully.")
