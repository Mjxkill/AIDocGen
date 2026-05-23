"""Ré-applique inject_illustrations sur des dossiers déjà terminés.

Réutilise les vision_captions déjà calculés dans corpus.json, applique le
seuil ENSEMBLE_VISION_SCORE_THRESHOLD courant, ré-écrit sections.json et
régénère le PDF.

Usage:
    python scripts/reinject_images.py [run_id1 run_id2 ...]

Sans arguments: traite les 3 derniers runs terminés.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# load env first
ENV_PATH = Path(__file__).resolve().parent.parent / "ensemble-proxy.env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DossierConfig
from core.llm import LLMClient
from core.writer import Writer


async def reinject(run_id: str, cfg: DossierConfig) -> dict:
    base = Path(cfg.data_dir) / run_id
    if not (base / "sections.json").exists() or not (base / "corpus.json").exists():
        return {"run": run_id, "skipped": "missing files"}

    sections = json.loads((base / "sections.json").read_text(encoding="utf-8"))
    corpus = json.loads((base / "corpus.json").read_text(encoding="utf-8"))
    claims_path = base / "claims.json"
    claims = []
    if claims_path.exists():
        claims = json.loads(claims_path.read_text(encoding="utf-8")).get("claims", [])

    # Strip existing image refs so inject_illustrations doesn't see "already
    # illustrated" sections and skip them.
    import re
    img_block_re = re.compile(
        r"\n*!\[[^\]]*\]\([^)]+\)(?:\{[^}]*\})?(?:[ \t]*\n+[ \t]*\*[^*\n][^*]*\*)?\n*",
    )
    for sec in sections.get("sections", []) or []:
        sec["content"] = img_block_re.sub("\n\n", sec.get("content") or "")

    llm = LLMClient(cfg)
    writer = Writer(cfg, llm)
    out = await writer.inject_illustrations(
        sections, claims, corpus, base,
        generate_ai=False, language="fr",
    )
    # Count results
    n_with = sum(
        1 for s in out.get("sections", [])
        if re.search(r"!\[[^\]]*\]\([^)]+\)", s.get("content", ""))
    )
    n_total = len(out.get("sections", []))
    return {"run": run_id, "sections": n_total, "with_image": n_with,
            "threshold": cfg.vision_score_threshold}


async def main(run_ids: list[str]):
    cfg = DossierConfig.from_env()
    print(f"Seuil utilisé: {cfg.vision_score_threshold}")
    for rid in run_ids:
        res = await reinject(rid, cfg)
        print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    runs = sys.argv[1:]
    if not runs:
        # default: 3 derniers terminés
        runs = [
            "run-1779489079-97e60adde0",
            "run-1779488912-f3e6b77af3",
            "run-1779466493-11944b573e",
        ]
    asyncio.run(main(runs))
