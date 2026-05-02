"""
Export AIDocGen corpus to Dify Knowledge Base.

Two modes:
  1. export_zip(run_id) -> ZIP file with clean texts + metadata (manual import)
  2. push_to_dify(run_id) -> Auto-push via Dify Console API (requires credentials)
"""

import json
import zipfile
import io
import time
import httpx
from pathlib import Path
from typing import Optional


def export_zip(run_id: str, data_dir: str = "data/dossiers") -> Path:
    """Create a ZIP with all clean source files + metadata for Dify import."""
    run_dir = Path(data_dir) / run_id
    clean_dir = run_dir / "clean"
    if not clean_dir.exists():
        raise FileNotFoundError(f"No clean directory for {run_id}")

    # Load metadata
    corpus = {}
    corpus_path = run_dir / "corpus.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text())

    claims = {}
    claims_path = run_dir / "claims.json"
    if claims_path.exists():
        claims = json.loads(claims_path.read_text())

    verdicts = {}
    verdicts_path = run_dir / "verdicts.json"
    if verdicts_path.exists():
        vdata = json.loads(verdicts_path.read_text())
        for v in vdata.get("verdicts", []):
            verdicts[v.get("claim_id")] = v

    shortlist = {}
    shortlist_path = run_dir / "shortlist.json"
    if shortlist_path.exists():
        sdata = json.loads(shortlist_path.read_text())
        for s in sdata.get("shortlist", sdata.get("ranked", [])):
            shortlist[s.get("source_id")] = s

    # Build source index
    source_meta = {}
    for src in corpus.get("sources", []):
        sid = src.get("source_id", "")
        source_meta[sid] = {
            "source_id": sid,
            "url": src.get("url", ""),
            "domain": src.get("domain", ""),
            "title": src.get("title", "").strip(),
            "relevance_score": shortlist.get(sid, {}).get("score", 0),
        }

    # Build claims per source
    claims_per_source = {}
    for c in claims.get("claims", []):
        sid = c.get("source_id", "")
        if sid not in claims_per_source:
            claims_per_source[sid] = []
        v = verdicts.get(c.get("claim_id"), {})
        claims_per_source[sid].append({
            "claim_id": c.get("claim_id"),
            "text": c.get("claim_text", ""),
            "type": c.get("claim_type", ""),
            "status": v.get("status", "UNKNOWN"),
        })

    # Create ZIP
    zip_path = run_dir / f"dify_export_{run_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write each clean file with metadata header
        for txt_file in sorted(clean_dir.glob("*.txt")):
            sid = txt_file.stem
            meta = source_meta.get(sid, {})
            content = txt_file.read_text(encoding="utf-8", errors="replace")

            # Prepend metadata header
            header = f"# Source: {meta.get('title', sid)}\n"
            header += f"# URL: {meta.get('url', '')}\n"
            header += f"# Domain: {meta.get('domain', '')}\n"
            header += f"# Relevance Score: {meta.get('relevance_score', 0)}\n"
            header += f"# Source ID: {sid}\n"

            src_claims = claims_per_source.get(sid, [])
            accepted = sum(1 for c in src_claims if c["status"] == "ACCEPTED")
            rejected = sum(1 for c in src_claims if c["status"] == "REJECTED")
            header += f"# Claims: {len(src_claims)} (accepted: {accepted}, rejected: {rejected})\n"
            header += "---\n\n"

            zf.writestr(f"sources/{txt_file.name}", header + content)

        # Write full metadata index
        index = {
            "run_id": run_id,
            "total_sources": len(source_meta),
            "total_claims": len(claims.get("claims", [])),
            "sources": list(source_meta.values()),
            "export_timestamp": int(time.time()),
        }
        zf.writestr("metadata.json", json.dumps(index, ensure_ascii=False, indent=2))

        # Write claims summary
        zf.writestr("claims.json", json.dumps(claims.get("claims", []), ensure_ascii=False, indent=2))

    return zip_path


async def push_to_dify(
    run_id: str,
    dify_url: str,
    email: str,
    password: str,
    data_dir: str = "data/dossiers",
    dataset_name: Optional[str] = None,
) -> dict:
    """Push corpus to a Dify Knowledge Base via Console API."""
    run_dir = Path(data_dir) / run_id
    clean_dir = run_dir / "clean"
    if not clean_dir.exists():
        raise FileNotFoundError(f"No clean directory for {run_id}")

    # Load corpus metadata
    corpus = {}
    corpus_path = run_dir / "corpus.json"
    if corpus_path.exists():
        corpus = json.loads(corpus_path.read_text())

    source_meta = {}
    for src in corpus.get("sources", []):
        sid = src.get("source_id", "")
        source_meta[sid] = src

    # Step 1: Login to Dify Console (password must be base64 encoded)
    import base64
    b64_password = base64.b64encode(password.encode("utf-8")).decode("ascii")

    # Dify 1.13+: cookie-based auth with CSRF token
    async with httpx.AsyncClient(base_url=dify_url.rstrip("/"), timeout=60) as client:
        login_resp = await client.post("/console/api/login", json={
            "email": email,
            "password": b64_password,
        })
        login_resp.raise_for_status()

        cookies = dict(login_resp.cookies)
        access_token = cookies.get("__Host-access_token", "")
        csrf_token = cookies.get("__Host-csrf_token", "")

        if not access_token:
            raise RuntimeError("Dify login succeeded but no access token in cookies")

        auth_cookies = {
            "__Host-access_token": access_token,
            "__Host-csrf_token": csrf_token,
        }
        headers = {"X-CSRF-Token": csrf_token}

        # Step 2: Create Dataset
        ds_name = (dataset_name or f"AIDocGen - {run_id}")[:40]
        status_data = {}
        status_path = run_dir / "status.json"
        if status_path.exists():
            status_data = json.loads(status_path.read_text())

        ds_resp = await client.post("/console/api/datasets", json={
            "name": ds_name,
            "description": f"Corpus AIDocGen: {status_data.get('question', run_id)} ({corpus.get('count', 0)} sources)",
            "permission": "all_team_members",
            "indexing_technique": "high_quality",
        }, headers=headers, cookies=auth_cookies)

        if ds_resp.status_code == 409:
            # Dataset name already exists -- find it and reuse
            list_resp = await client.get("/console/api/datasets", headers=headers, cookies=auth_cookies)
            list_resp.raise_for_status()
            dataset_id = None
            for ds in list_resp.json().get("data", []):
                if ds.get("name") == ds_name:
                    dataset_id = ds["id"]
                    break
            if not dataset_id:
                raise RuntimeError(f"Dataset '{ds_name}' conflict but not found in list")
        else:
            ds_resp.raise_for_status()
            dataset_id = ds_resp.json().get("id")

        # Step 3: Upload documents (upload file -> create document)
        uploaded = 0
        errors = []
        txt_files = sorted(clean_dir.glob("*.txt"))

        for txt_file in txt_files:
            sid = txt_file.stem
            meta = source_meta.get(sid, {})
            content = txt_file.read_bytes()

            if len(content.strip()) < 50:
                continue

            raw_name = meta.get("title", "").strip() or sid
            # Sanitize filename: keep only alphanumeric, spaces, hyphens, underscores
            import re as _re
            doc_name = _re.sub(r'[^\w\s\-.]', '', raw_name)[:80].strip() or sid
            doc_name = doc_name + ".txt"

            try:
                # Step 3a: Upload file
                upload_resp = await client.post(
                    "/console/api/files/upload",
                    files={"file": (doc_name, content, "text/plain")},
                    data={"source": "datasets"},
                    headers=headers,
                    cookies=auth_cookies,
                    timeout=30,
                )
                if upload_resp.status_code not in (200, 201):
                    errors.append({"source_id": sid, "step": "upload", "status": upload_resp.status_code, "detail": upload_resp.text[:200]})
                    continue

                file_id = upload_resp.json().get("id")

                # Step 3b: Create document from uploaded file
                create_resp = await client.post(
                    f"/console/api/datasets/{dataset_id}/documents",
                    json={
                        "indexing_technique": "high_quality",
                        "data_source": {
                            "info_list": {
                                "data_source_type": "upload_file",
                                "file_info_list": {"file_ids": [file_id]},
                            }
                        },
                        "process_rule": {"mode": "automatic"},
                        "doc_form": "text_model",
                        "doc_language": "French",
                    },
                    headers=headers,
                    cookies=auth_cookies,
                    timeout=30,
                )
                if create_resp.status_code < 300:
                    uploaded += 1
                else:
                    errors.append({"source_id": sid, "step": "create", "status": create_resp.status_code, "detail": create_resp.text[:200]})
            except Exception as e:
                errors.append({"source_id": sid, "error": str(e)[:200]})

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "dataset_name": ds_name,
            "uploaded": uploaded,
            "total": len(txt_files),
            "errors": errors[:10],
        }


async def _dify_login(client: httpx.AsyncClient, email: str, password: str) -> tuple[dict, dict]:
    """Login to Dify and return (headers, cookies) for authenticated requests."""
    import base64
    b64_password = base64.b64encode(password.encode("utf-8")).decode("ascii")
    login_resp = await client.post("/console/api/login", json={
        "email": email,
        "password": b64_password,
    })
    login_resp.raise_for_status()
    cookies = dict(login_resp.cookies)
    access_token = cookies.get("__Host-access_token", "")
    csrf_token = cookies.get("__Host-csrf_token", "")
    if not access_token:
        raise RuntimeError("Dify login failed: no access token")
    auth_cookies = {"__Host-access_token": access_token, "__Host-csrf_token": csrf_token}
    headers = {"X-CSRF-Token": csrf_token}
    return headers, auth_cookies


async def _dify_get_or_create_dataset(client: httpx.AsyncClient, name: str, description: str, headers: dict, cookies: dict) -> str:
    """Create a Dify dataset or return existing one if name conflicts."""
    ds_resp = await client.post("/console/api/datasets", json={
        "name": name,
        "description": description,
        "permission": "all_team_members",
        "indexing_technique": "high_quality",
    }, headers=headers, cookies=cookies)
    if ds_resp.status_code == 409:
        list_resp = await client.get("/console/api/datasets", headers=headers, cookies=cookies)
        list_resp.raise_for_status()
        for ds in list_resp.json().get("data", []):
            if ds.get("name") == name:
                return ds["id"]
        raise RuntimeError(f"Dataset '{name}' conflict but not found")
    ds_resp.raise_for_status()
    return ds_resp.json().get("id")


async def _dify_upload_text(client: httpx.AsyncClient, dataset_id: str, filename: str, content: bytes, headers: dict, cookies: dict) -> bool:
    """Upload a file and create a document in Dify. Returns True on success."""
    import re as _re
    safe_name = _re.sub(r'[^\w\s\-.]', '', filename)[:80].strip() or "doc"
    if not safe_name.endswith(".txt"):
        safe_name += ".txt"

    upload_resp = await client.post(
        "/console/api/files/upload",
        files={"file": (safe_name, content, "text/plain")},
        data={"source": "datasets"},
        headers=headers, cookies=cookies, timeout=30,
    )
    if upload_resp.status_code not in (200, 201):
        return False
    file_id = upload_resp.json().get("id")
    create_resp = await client.post(
        f"/console/api/datasets/{dataset_id}/documents",
        json={
            "indexing_technique": "high_quality",
            "data_source": {"info_list": {"data_source_type": "upload_file", "file_info_list": {"file_ids": [file_id]}}},
            "process_rule": {"mode": "automatic"},
            "doc_form": "text_model",
            "doc_language": "French",
        },
        headers=headers, cookies=cookies, timeout=30,
    )
    return create_resp.status_code < 300


async def push_verified_to_dify(
    run_id: str,
    dify_url: str,
    email: str,
    password: str,
    data_dir: str = "data/dossiers",
    dataset_name: Optional[str] = None,
) -> dict:
    """Push only VERIFIED content to Dify: report.md + accepted claims."""
    run_dir = Path(data_dir) / run_id

    # Load status for question
    status_data = {}
    status_path = run_dir / "status.json"
    if status_path.exists():
        status_data = json.loads(status_path.read_text())

    ds_name = (dataset_name or f"AIDocGen - {run_id}")[:40]

    async with httpx.AsyncClient(base_url=dify_url.rstrip("/"), timeout=60) as client:
        headers, auth_cookies = await _dify_login(client, email, password)
        dataset_id = await _dify_get_or_create_dataset(
            client, ds_name,
            f"Contenu verifie AIDocGen: {status_data.get('question', run_id)}",
            headers, auth_cookies,
        )

        uploaded = 0
        errors = []

        # 1. Upload report.md (the main verified document)
        report_path = run_dir / "report.md"
        if report_path.exists():
            content = report_path.read_bytes()
            # Split into chapters for better chunking in Dify
            report_text = report_path.read_text(encoding="utf-8")
            chapters = _split_report_into_chapters(report_text)

            for i, (title, chapter_content) in enumerate(chapters, 1):
                safe_title = title[:60] or f"Chapitre {i}"
                ok = await _dify_upload_text(
                    client, dataset_id, f"{i:02d} - {safe_title}",
                    chapter_content.encode("utf-8"),
                    headers, auth_cookies,
                )
                if ok:
                    uploaded += 1
                else:
                    errors.append({"file": safe_title, "step": "upload"})

        # 2. Upload accepted claims as a structured reference document
        claims_path = run_dir / "claims.json"
        verdicts_path = run_dir / "verdicts.json"
        if claims_path.exists() and verdicts_path.exists():
            claims = json.loads(claims_path.read_text())
            verdicts = json.loads(verdicts_path.read_text())
            verdict_map = {v.get("claim_id"): v for v in verdicts.get("verdicts", [])}

            accepted_lines = ["# Donnees Factuelles Validees\n"]
            accepted_lines.append(f"Run: {run_id}\n")
            accepted_lines.append(f"Question: {status_data.get('question', '')}\n\n")

            for c in claims.get("claims", []):
                v = verdict_map.get(c.get("claim_id"), {})
                if v.get("status", "").upper() == "ACCEPTED":
                    accepted_lines.append(f"- [{c.get('claim_id')}] {c.get('claim_text', '')}\n")
                    accepted_lines.append(f"  Source: {c.get('source_id', '')}\n\n")

            if len(accepted_lines) > 3:
                ok = await _dify_upload_text(
                    client, dataset_id, "Claims validees",
                    "\n".join(accepted_lines).encode("utf-8"),
                    headers, auth_cookies,
                )
                if ok:
                    uploaded += 1

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "dataset_name": ds_name,
            "uploaded": uploaded,
            "errors": errors[:10],
        }


def _split_report_into_chapters(report_md: str) -> list[tuple[str, str]]:
    """Split a markdown report into chapters (## headings)."""
    import re
    chapters = []
    parts = re.split(r'^(## .+)$', report_md, flags=re.MULTILINE)

    # parts[0] is content before first ##, then alternating heading/content
    if parts[0].strip():
        chapters.append(("Introduction", parts[0]))

    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("# ").strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        if content.strip():
            chapters.append((heading, parts[i] + content))

    # If no chapters found, return whole document
    if not chapters:
        chapters.append(("Document complet", report_md))

    return chapters


async def push_via_dataset_api(
    run_id: str,
    api_url: str,
    api_key: str,
    data_dir: str = "data/dossiers",
    dataset_name: Optional[str] = None,
) -> dict:
    """Push verified content to Dify via the Dataset API (API key auth).

    This is the preferred method over Console API login.
    Uses POST /datasets/{id}/document/create-by-text for each chapter.
    """
    run_dir = Path(data_dir) / run_id

    # Load status
    status_data = {}
    status_path = run_dir / "status.json"
    if status_path.exists():
        status_data = json.loads(status_path.read_text())

    question = status_data.get("question", run_id)
    ds_name = (dataset_name or f"AIDocGen - {question}")[:40]

    base = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60) as client:
        # Step 1: Create or find dataset
        dataset_id = None

        # Check if dataset already exists
        list_resp = await client.get(f"{base}/datasets?page=1&limit=100", headers=headers)
        if list_resp.status_code == 200:
            for ds in list_resp.json().get("data", []):
                if ds.get("name") == ds_name:
                    dataset_id = ds["id"]
                    break

        if not dataset_id:
            create_resp = await client.post(f"{base}/datasets", headers=headers, json={
                "name": ds_name,
                "permission": "all_team_members",
                "indexing_technique": "high_quality",
            })
            create_resp.raise_for_status()
            dataset_id = create_resp.json().get("id")

        uploaded = 0
        errors = []

        # Step 2: Upload report.md split into chapters
        report_path = run_dir / "report.md"
        if report_path.exists():
            report_text = report_path.read_text(encoding="utf-8")
            chapters = _split_report_into_chapters(report_text)

            for i, (title, chapter_content) in enumerate(chapters, 1):
                safe_title = title[:60] or f"Chapitre {i}"
                try:
                    resp = await client.post(
                        f"{base}/datasets/{dataset_id}/document/create-by-text",
                        headers=headers,
                        json={
                            "name": f"{i:02d} - {safe_title}",
                            "text": chapter_content,
                            "indexing_technique": "high_quality",
                            "process_rule": {"mode": "automatic"},
                        },
                        timeout=30,
                    )
                    if resp.status_code < 300:
                        uploaded += 1
                    else:
                        errors.append({"file": safe_title, "status": resp.status_code, "detail": resp.text[:200]})
                except Exception as e:
                    errors.append({"file": safe_title, "error": str(e)[:200]})

        # Step 3: Upload accepted claims
        claims_path = run_dir / "claims.json"
        verdicts_path = run_dir / "verdicts.json"
        if claims_path.exists() and verdicts_path.exists():
            claims = json.loads(claims_path.read_text())
            verdicts = json.loads(verdicts_path.read_text())
            verdict_map = {v.get("claim_id"): v for v in verdicts.get("verdicts", [])}

            accepted_lines = [f"# Donnees Factuelles Validees\n\nRun: {run_id}\nQuestion: {question}\n\n"]
            for c in claims.get("claims", []):
                v = verdict_map.get(c.get("claim_id"), {})
                if v.get("status", "").upper() == "ACCEPTED":
                    accepted_lines.append(f"- [{c.get('claim_id')}] {c.get('claim_text', '')}\n  Source: {c.get('source_id', '')}\n\n")

            if len(accepted_lines) > 1:
                try:
                    resp = await client.post(
                        f"{base}/datasets/{dataset_id}/document/create-by-text",
                        headers=headers,
                        json={
                            "name": "Claims validees",
                            "text": "".join(accepted_lines),
                            "indexing_technique": "high_quality",
                            "process_rule": {"mode": "automatic"},
                        },
                        timeout=30,
                    )
                    if resp.status_code < 300:
                        uploaded += 1
                except Exception as e:
                    errors.append({"file": "claims", "error": str(e)[:200]})

        # Step 4: Upload raw sources
        clean_dir = run_dir / "clean"
        if clean_dir.exists():
            corpus = {}
            corpus_path = run_dir / "corpus.json"
            if corpus_path.exists():
                corpus = json.loads(corpus_path.read_text())
            source_meta = {s.get("source_id"): s for s in corpus.get("sources", [])}

            for txt_file in sorted(clean_dir.glob("*.txt")):
                sid = txt_file.stem
                meta = source_meta.get(sid, {})
                content = txt_file.read_text(encoding="utf-8", errors="replace")
                if len(content.strip()) < 50:
                    continue

                doc_title = (meta.get("title", "") or sid)[:80]
                header = f"Source: {meta.get('url', '')}\nDomain: {meta.get('domain', '')}\n\n"
                try:
                    resp = await client.post(
                        f"{base}/datasets/{dataset_id}/document/create-by-text",
                        headers=headers,
                        json={
                            "name": doc_title,
                            "text": header + content,
                            "indexing_technique": "high_quality",
                            "process_rule": {"mode": "automatic"},
                        },
                        timeout=30,
                    )
                    if resp.status_code < 300:
                        uploaded += 1
                    else:
                        errors.append({"source_id": sid, "status": resp.status_code, "detail": resp.text[:200]})
                except Exception as e:
                    errors.append({"source_id": sid, "error": str(e)[:200]})

        return {
            "status": "ok",
            "dataset_id": dataset_id,
            "dataset_name": ds_name,
            "uploaded": uploaded,
            "errors": errors[:10],
        }
