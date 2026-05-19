"""Vision-LLM image captioning for corpus images.

Pour chaque image candidate, on demande à un modèle vision (qwen3-vl par défaut)
de produire un caption factuel court qui sera ensuite utilisé pour le scoring
de pertinence dans inject_illustrations (writer.py).

Sans ce module, le scoring se base uniquement sur l'alt-text trouvé sur la page
source — souvent vide, générique, ou hors-sujet.
"""
import asyncio
import base64
import re
import time
from typing import Any

import httpx

from .config import DossierConfig
from .logging_config import get_logger

log = get_logger("aidocgen.vision")


_VISION_PROMPT = (
    "Décris cette image en UNE phrase en français, en mettant l'accent sur "
    "son SUJET technique ou informatif principal (ex: schéma de circuit, "
    "graphe de données, photo de matériel, capture d'écran, équation, "
    "diagramme de flux). Sois concis et factuel — pas d'adjectifs subjectifs. "
    "Si l'image est purement décorative (logo, photo générique d'illustration "
    "sans contenu informatif), réponds uniquement par le mot DECORATIVE."
)

# alt-text patterns suggesting a useless image — skip before paying for vision
_DECO_HINT = re.compile(
    r"\b(logo|icon|favicon|sprite|banner|placeholder|avatar|profile[-_ ]photo)\b",
    re.IGNORECASE,
)


async def _download_image(url: str, client: httpx.AsyncClient) -> bytes | None:
    try:
        r = await client.get(
            url, timeout=15.0,
            headers={"User-Agent": "AIDocGen/1.0", "Accept": "image/*"},
        )
        if r.status_code != 200:
            return None
        ct = r.headers.get("Content-Type", "").lower()
        if "image" not in ct:
            return None
        data = r.content
        # ignore tiny (likely tracking/icons) and overly huge (cost) images
        if len(data) < 1024 or len(data) > 8_000_000:
            return None
        return data
    except Exception:
        return None


async def _caption_one(
    img_url: str,
    model: str,
    base_url: str,
    api_key: str | None,
    sem: asyncio.Semaphore,
    dl_client: httpx.AsyncClient,
) -> str | None:
    async with sem:
        data = await _download_image(img_url, dl_client)
        if not data:
            return None
        b64 = base64.b64encode(data).decode("ascii")
        url = f"{base_url.rstrip('/')}/api/chat"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": _VISION_PROMPT,
                "images": [b64],
            }],
            "stream": False,
        }
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code != 200:
                    log.debug("vision http error",
                              extra={"status": resp.status_code, "url": img_url[:120]})
                    return None
                content = resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            log.debug("vision call failed",
                      extra={"error": str(e), "url": img_url[:120]})
            return None
        if not content:
            return None
        # "DECORATIVE" sentinel => caller treats as no-content
        if content.upper().startswith("DECORATIVE"):
            return ""
        # bound length to keep corpus.json reasonable
        return content[:400]


async def caption_corpus_images(
    sources: list[dict[str, Any]],
    config: DossierConfig,
) -> dict[str, int]:
    """Mutate ``sources`` in place: add ``vision_caption`` to selected images.

    - Skips images with decoration hints in alt-text.
    - Caps to ``config.vision_max_per_source`` candidates per source.
    - Re-uses cached caption if already present (resume-safe).
    - Empty caption = DECORATIVE (image will be deprioritised by scorer).

    Returns counters for logging.
    """
    if not config.vision_enabled or not config.vision_model:
        return {"total": 0, "captioned": 0, "decorative": 0, "skipped_alt": 0,
                "skipped_cap": 0, "cached": 0}

    model = config.vision_model
    cap_per_source = max(1, config.vision_max_per_source)

    tasks: list[tuple[dict, str]] = []
    cached = 0
    skipped_alt = 0
    skipped_cap = 0
    for src in sources:
        imgs = src.get("images") or []
        chosen = 0
        for img in imgs:
            url = (img.get("url") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            if chosen >= cap_per_source:
                skipped_cap += 1
                continue
            if _DECO_HINT.search(img.get("alt") or ""):
                skipped_alt += 1
                continue
            # already captioned (resume) — count as used quota
            if "vision_caption" in img:
                cached += 1
                chosen += 1
                continue
            tasks.append((img, url))
            chosen += 1

    if not tasks:
        log.info("vision: nothing to caption",
                 extra={"cached": cached, "skipped_alt": skipped_alt,
                        "skipped_cap": skipped_cap})
        return {"total": 0, "captioned": 0, "decorative": 0,
                "skipped_alt": skipped_alt, "skipped_cap": skipped_cap,
                "cached": cached}

    log.info("vision caption batch starting",
             extra={"count": len(tasks), "cached": cached, "model": model,
                    "parallel": config.vision_max_parallel})

    sem = asyncio.Semaphore(max(1, config.vision_max_parallel))
    start_ts = time.time()
    async with httpx.AsyncClient() as dl_client:
        results = await asyncio.gather(
            *(_caption_one(url, model, config.ollama_base_url,
                           config.ollama_api_key, sem, dl_client)
              for _, url in tasks),
            return_exceptions=False,
        )

    captioned = 0
    decorative = 0
    for (img, _url), result in zip(tasks, results):
        if result is None:
            # download or vision call failed — do NOT mark, retry on resume
            continue
        img["vision_caption"] = result
        if result == "":
            decorative += 1
        else:
            captioned += 1

    log.info("vision caption batch done",
             extra={"total": len(tasks), "captioned": captioned,
                    "decorative": decorative,
                    "duration_s": round(time.time() - start_ts, 1)})
    return {"total": len(tasks), "captioned": captioned, "decorative": decorative,
            "skipped_alt": skipped_alt, "skipped_cap": skipped_cap,
            "cached": cached}
