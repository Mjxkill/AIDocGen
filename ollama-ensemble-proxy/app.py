import os
import re
import time
import asyncio
import uuid
import psutil
import json
import shutil
import subprocess
import zipfile
import httpx
import mimetypes
from pathlib import Path
from typing import Any, Optional
from datetime import timedelta
from collections import Counter
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Depends, status, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, ConfigDict

from core.config import DossierConfig
from core.engine import DossierEngine
from core.auth import get_current_user, get_admin_user, create_access_token, verify_password, get_password_hash, _get_users, _save_users, ACCESS_TOKEN_EXPIRE_MINUTES

# --- MIMETYPES SETUP ---
mimetypes.init()
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/css', '.css')

AIDOCGEN_VERSION = "1.0.0"
app = FastAPI(title="AIDocGen Proxy")

# --- ASSETS PRIORITY (For MIME stability) ---
app.mount("/ui/assets", StaticFiles(directory="../web-ui/dist/assets"), name="assets")
app.mount("/assets", StaticFiles(directory="../web-ui/dist/assets"), name="assets_legacy")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "dossiers"
PDF_JOBS_DIR = BASE_DIR / "data" / "pdf_jobs"
VIDEO_JOBS_DIR = BASE_DIR / "data" / "video_jobs"
AUDIO_JOBS_DIR = BASE_DIR / "data" / "audio_jobs"
WEB_JOBS_DIR = BASE_DIR / "data" / "web_jobs"
AUDIOBOOK_JOBS_DIR = BASE_DIR / "data" / "audiobook_jobs"
SCRIPTS_DIR = BASE_DIR.parent / "scripts"
_DOSSIER_TASKS: dict[str, asyncio.Task] = {}
_PDF_TASKS: dict[str, asyncio.Task] = {}
_VIDEO_TASKS: dict[str, asyncio.Task] = {}
_AUDIO_TASKS: dict[str, asyncio.Task] = {}
_WEB_TASKS: dict[str, asyncio.Task] = {}
_AUDIOBOOK_TASKS: dict[str, asyncio.Task] = {}
SERVERS_FILE = BASE_DIR / "data" / "servers.json"

# --- INTERNAL UTILS ---
def _get_servers():
    if not SERVERS_FILE.exists():
        initial = [{"name": "Localhost", "url": "http://127.0.0.1:11434"}]
        SERVERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERVERS_FILE.write_text(json.dumps(initial))
    return json.loads(SERVERS_FILE.read_text())

def _get_engine(ollama_url: str = None, models: dict = None):
    config = DossierConfig.from_env()
    config.data_dir = str(DATA_DIR)
    if models and ("744b" in str(models.get("planner_model", "")) or "675b" in str(models.get("writer_model", ""))):
        config.ollama_base_url = "https://ollama.com"
    elif ollama_url:
        config.ollama_base_url = ollama_url
    if models:
        if models.get("planner_model"): config.planner_model = models["planner_model"]
        if models.get("writer_model"): config.writer_model = models["writer_model"]
        if models.get("judge_model"): config.judge_model = models["judge_model"]
        if models.get("coder_model"): config.planner_book_model_4_json = models["coder_model"]
        if models.get("extract_model"): config.extract_model = models["extract_model"]
    return DossierEngine(config)

def _get_gpu_usage():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], encoding="utf-8")
        gpus = []
        for line in output.strip().split("\n"):
            util, used, total = map(int, line.split(","))
            gpus.append({"util": util, "mem_used": used, "mem_total": total})
        return gpus
    except: return []

def _is_admin(user: dict) -> bool:
    return user.get("role") == "admin"

def _check_run_access(run_id: str, user: dict) -> Path:
    """Return the run dir if user is admin or owner, else raise."""
    run_dir = DATA_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404)
    if _is_admin(user):
        return run_dir
    s_path = run_dir / "status.json"
    if s_path.exists():
        try:
            data = json.loads(s_path.read_text())
            if data.get("owner_id") == user["id"]:
                return run_dir
        except Exception:
            pass
    raise HTTPException(403, "Access denied")

def _check_pdf_access(job_id: str, user: dict) -> Path:
    job_dir = PDF_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404)
    if _is_admin(user):
        return job_dir
    data = _read_pdf_status(job_id) or {}
    if data.get("owner_id") == user["id"]:
        return job_dir
    raise HTTPException(403, "Access denied")

def _check_video_access(job_id: str, user: dict) -> Path:
    job_dir = VIDEO_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404)
    if _is_admin(user):
        return job_dir
    data = _read_video_status(job_id) or {}
    if data.get("owner_id") == user["id"]:
        return job_dir
    raise HTTPException(403, "Access denied")

def _check_audio_access(job_id: str, user: dict) -> Path:
    job_dir = AUDIO_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404)
    if _is_admin(user):
        return job_dir
    data = _read_audio_status(job_id) or {}
    if data.get("owner_id") == user["id"]:
        return job_dir
    raise HTTPException(403, "Access denied")

def _check_web_access(job_id: str, user: dict) -> Path:
    job_dir = WEB_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404)
    if _is_admin(user):
        return job_dir
    data = _read_web_status(job_id) or {}
    if data.get("owner_id") == user["id"]:
        return job_dir
    raise HTTPException(403, "Access denied")

def _check_audiobook_access(job_id: str, user: dict) -> Path:
    job_dir = AUDIOBOOK_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404)
    if _is_admin(user):
        return job_dir
    data = _read_audiobook_status(job_id) or {}
    if data.get("owner_id") == user["id"]:
        return job_dir
    raise HTTPException(403, "Access denied")

def parse_tags(tags_str: str) -> list[str]:
    """Parse tags string into list. Accepts comma-separated or space-separated."""
    if not tags_str:
        return []
    # Split by comma or space
    tags = re.split(r'[,\s]+', tags_str.strip())
    # Clean and filter
    tags = [t.strip().lstrip('#').upper() for t in tags if t.strip()]
    return tags

class RunRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    question: str
    prompt_type: Optional[str] = "generic"
    detail_level: Optional[str] = "medium"
    language: Optional[str] = "fr"
    tags: Optional[str] = None  # Comma or space separated tags, e.g. "IMX8MP, DSP, HiFi4"
    ollama_url: Optional[str] = None
    planner_model: Optional[str] = None
    writer_model: Optional[str] = None
    judge_model: Optional[str] = None
    coder_model: Optional[str] = None
    extract_model: Optional[str] = None
    include_images: Optional[bool] = True
    generate_ai_images: Optional[bool] = False

# --- AUTH API ---
@app.post("/v1/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    users = _get_users()
    user = next((u for u in users if u["username"] == form_data.username), None)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Bad credentials")
    return {
        "access_token": create_access_token(data={"sub": user["username"]}),
        "token_type": "bearer",
        "role": user.get("role", "user"),
        "must_change_password": bool(user.get("must_change_password", False)),
    }

@app.get("/v1/auth/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {
        "username": current_user["username"],
        "role": current_user["role"],
        "id": current_user["id"],
        "must_change_password": bool(current_user.get("must_change_password", False)),
    }

class SelfPasswordChange(BaseModel):
    current_password: str
    new_password: str

@app.post("/v1/auth/change-password")
async def change_own_password(payload: SelfPasswordChange,
                              current_user: dict = Depends(get_current_user)):
    """Self-service password change. Used by the forced-change-on-first-login flow
    AND by any user who wants to update their own password."""
    if not payload.new_password or len(payload.new_password) < 4:
        raise HTTPException(400, "new password must be at least 4 characters")
    users = _get_users()
    user = next((u for u in users if u["id"] == current_user["id"]), None)
    if not user:
        raise HTTPException(404, "user not found")
    if not verify_password(payload.current_password, user["hashed_password"]):
        raise HTTPException(400, "current password is incorrect")
    user["hashed_password"] = get_password_hash(payload.new_password)
    user["must_change_password"] = False
    _save_users(users)
    return {"status": "ok"}

@app.get("/v1/auth/preferences")
async def get_preferences(current_user: dict = Depends(get_current_user)):
    users = _get_users()
    u = next((x for x in users if x["id"] == current_user["id"]), None)
    return (u or {}).get("preferences", {})

@app.put("/v1/auth/preferences")
async def save_preferences(prefs: dict, current_user: dict = Depends(get_current_user)):
    users = _get_users()
    for u in users:
        if u["id"] == current_user["id"]:
            existing = u.get("preferences", {})
            existing.update(prefs)
            u["preferences"] = existing
            _save_users(users)
            return {"status": "ok"}
    raise HTTPException(404)

# --- USER MANAGEMENT (admin only) ---
class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"

class PasswordChange(BaseModel):
    password: str

@app.get("/v1/users")
async def list_users(admin: dict = Depends(get_admin_user)):
    users = _get_users()
    return {"data": [{"id": u["id"], "username": u["username"], "role": u.get("role", "user")} for u in users]}

@app.post("/v1/users")
async def create_user(payload: UserCreate, admin: dict = Depends(get_admin_user)):
    if not payload.username.strip() or not payload.password:
        raise HTTPException(400, "username and password required")
    if payload.role not in ("admin", "user"):
        raise HTTPException(400, "role must be 'admin' or 'user'")
    users = _get_users()
    if any(u["username"] == payload.username for u in users):
        raise HTTPException(409, "username already exists")
    new_user = {
        "id": f"u-{uuid.uuid4().hex[:10]}",
        "username": payload.username.strip(),
        "hashed_password": get_password_hash(payload.password),
        "role": payload.role,
        "must_change_password": True,
    }
    users.append(new_user)
    _save_users(users)
    return {"id": new_user["id"], "username": new_user["username"], "role": new_user["role"]}

@app.post("/v1/users/{username}/password")
async def change_user_password(username: str, payload: PasswordChange, admin: dict = Depends(get_admin_user)):
    users = _get_users()
    user = next((u for u in users if u["username"] == username), None)
    if not user:
        raise HTTPException(404, "user not found")
    user["hashed_password"] = get_password_hash(payload.password)
    user["must_change_password"] = True  # force the user to pick their own
    _save_users(users)
    return {"status": "ok"}

@app.delete("/v1/users/{username}")
async def delete_user(username: str, admin: dict = Depends(get_admin_user)):
    if username == admin["username"]:
        raise HTTPException(400, "cannot delete yourself")
    users = _get_users()
    if not any(u["username"] == username for u in users):
        raise HTTPException(404, "user not found")
    users = [u for u in users if u["username"] != username]
    _save_users(users)
    return {"status": "ok"}

# --- DOSSIER API ---
@app.get("/v1/dossier/prompts")
async def list_prompts(user: dict = Depends(get_current_user)):
    p_dir = BASE_DIR / "prompts"
    return {"prompts": [f.stem.replace("planner_", "") for f in p_dir.glob("planner_*.txt")]}

@app.get("/v1/dossier/defaults")
async def dossier_defaults(user: dict = Depends(get_current_user)):
    """Server-defined default models, used when a user has no saved preferences."""
    cfg = DossierConfig.from_env()
    return {
        "planner_model": cfg.planner_model,
        "writer_model": cfg.writer_model,
        "judge_model": cfg.judge_model,
        "extract_model": cfg.extract_model,
        "coder_model": cfg.planner_book_model_4_json,
    }

@app.get("/v1/dossier/runs")
async def list_runs(limit: int = 20, user: dict = Depends(get_current_user)):
    runs = []
    is_admin = _is_admin(user)
    if DATA_DIR.exists():
        dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs:
            s_path = d / "status.json"
            if not s_path.exists(): continue
            try:
                data = json.loads(s_path.read_text())
            except: continue
            if not is_admin and data.get("owner_id") != user["id"]:
                continue
            if data.get("state") == "running" and data.get("run_id") not in _DOSSIER_TASKS:
                data["state"] = "interrupted"
            runs.append(data)
            if len(runs) >= limit:
                break
    return {"data": runs}

@app.post("/v1/dossier/runs")
async def start_run(req: RunRequest, user: dict = Depends(get_current_user)):
    import re
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    engine = _get_engine(req.ollama_url, req.dict())
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse tags
    tags = parse_tags(req.tags) if req.tags else []
    
    status_data = {
        "run_id": run_id, "question": req.question, "state": "running", "stage": "init",
        "prompt_type": req.prompt_type, "detail_level": req.detail_level, "language": req.language,
        "tags": tags,  # Store tags in status
        "ollama_url": req.ollama_url, "planner_model": req.planner_model, "writer_model": req.writer_model,
        "coder_model": req.coder_model, "extract_model": req.extract_model,
        "include_images": bool(req.include_images),
        "generate_ai_images": bool(req.generate_ai_images),
        "owner_id": user["id"], "owner_name": user["username"],
        "events": [{"timestamp": int(time.time()), "stage": "init", "message": "Démarrage..."}],
        "updated_at": int(time.time())
    }
    (run_dir / "status.json").write_text(json.dumps(status_data))
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(
        run_id, req.question, req.prompt_type, req.detail_level,
        language=req.language, coder_model=req.coder_model, tags=tags,
        include_images=bool(req.include_images),
        generate_ai_images=bool(req.generate_ai_images),
    ))
    return {"run_id": run_id}

@app.get("/v1/dossier/runs/{run_id}/planner")
async def get_planner(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    path = DATA_DIR / run_id / "planner.json"
    if not path.exists(): raise HTTPException(404)
    return json.loads(path.read_text())

@app.post("/v1/dossier/runs/{run_id}/planner")
async def update_planner(run_id: str, payload: dict, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    path = DATA_DIR / run_id / "planner.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"status": "ok"}

@app.get("/v1/dossier/runs/{run_id}/planner/debug")
async def get_planner_debug(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    path = DATA_DIR / run_id / "planner_debug.json"
    if not path.exists(): return {"attempts": [], "planner_prompt": {"system": "", "user": ""}, "planner_response_raw": "N/A"}
    return json.loads(path.read_text())

@app.get("/v1/dossier/runs/{run_id}/details")
async def get_run_details(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    run_dir = DATA_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(404)

    def _load(name: str):
        p = run_dir / name
        if p.exists():
            try: return json.loads(p.read_text())
            except: pass
        return None

    status = _load("status.json") or {}
    config = DossierConfig.from_env()

    # --- Models (from status.json or fallback) ---
    models = status.get("models") or {
        "planner": status.get("planner_model") or config.planner_model,
        "writer": status.get("writer_model") or config.writer_model,
        "judge": config.judge_model,
        "verify": config.verify_model,
        "extract": config.extract_model,
        "coder": status.get("coder_model") or config.planner_book_model_4_json,
    }

    # --- Search stats ---
    search_data = _load("search_results.json")
    search = None
    if search_data and search_data.get("sub_questions"):
        total_queries = len(search_data["sub_questions"])
        total_results = 0
        engine_counter: Counter = Counter()
        domain_counter: Counter = Counter()
        for sq in search_data["sub_questions"]:
            for lnk in sq.get("links", []):
                total_results += 1
                if isinstance(lnk, dict):
                    eng = lnk.get("engine", "unknown")
                    engine_counter[eng] += 1
                    url = lnk.get("url", "")
                    try: domain_counter[urlparse(url).netloc.removeprefix("www.")] += 1
                    except: pass
        top_domains = [{"domain": d, "count": c} for d, c in domain_counter.most_common(10)]
        search = {
            "total_queries": total_queries, "total_results": total_results,
            "engines": dict(engine_counter), "top_domains": top_domains,
        }

    # --- Quality / verdicts ---
    corpus = _load("corpus.json")
    claims = _load("claims.json")
    verdicts = _load("verdicts.json")
    quality = None
    if claims and claims.get("claims"):
        sources_total = corpus.get("count", 0) if corpus else 0
        claims_total = len(claims["claims"])
        verdict_counter: Counter = Counter()
        if verdicts and verdicts.get("verdicts"):
            for v in verdicts["verdicts"]:
                verdict_counter[v.get("status", "UNKNOWN").upper()] += 1
        accepted = verdict_counter.get("ACCEPTED", 0)
        rejected = verdict_counter.get("REJECTED", 0)
        uncertain = verdict_counter.get("UNCERTAIN", 0)
        total_v = accepted + rejected + uncertain or 1
        quality = {
            "sources_total": sources_total, "claims_total": claims_total,
            "verdicts": {
                "accepted": accepted, "rejected": rejected, "uncertain": uncertain,
                "accepted_pct": round(accepted / total_v * 100, 1),
                "rejected_pct": round(rejected / total_v * 100, 1),
                "uncertain_pct": round(uncertain / total_v * 100, 1),
            }
        }

    # --- Errors ---
    errors = []
    for ev in status.get("events", []):
        msg = ev.get("message", "")
        if ev.get("stage") == "failed" or "error" in msg.lower() or "exception" in msg.lower():
            errors.append({"timestamp": ev.get("timestamp"), "stage": ev.get("stage", ""), "message": msg})
    if status.get("error"):
        errors.append({"timestamp": status.get("updated_at"), "stage": "failed", "message": status["error"]})

    # --- Sources list (all URLs visited) ---
    sources_list = []
    if corpus and corpus.get("sources"):
        for src in corpus["sources"]:
            sources_list.append({
                "source_id": src.get("source_id", ""),
                "url": src.get("url", ""),
                "title": src.get("title", "Sans titre"),
                "domain": src.get("domain", ""),
            })

    # --- Claims detail by status ---
    claims_detail = {"accepted": [], "rejected": [], "uncertain": []}
    verdict_map = {}
    if verdicts and verdicts.get("verdicts"):
        for v in verdicts["verdicts"]:
            verdict_map[v.get("claim_id")] = v
    if claims and claims.get("claims"):
        for c in claims["claims"]:
            cid = c.get("claim_id", "")
            v = verdict_map.get(cid, {})
            entry = {
                "claim_id": cid,
                "text": c.get("claim_text", ""),
                "type": c.get("claim_type", ""),
                "source_id": c.get("source_id", ""),
                "status": v.get("status", "UNCERTAIN"),
                "justification": v.get("justification", ""),
                "confidence": v.get("confidence", 0.5),
            }
            bucket = v.get("status", "UNCERTAIN").lower()
            if bucket in claims_detail:
                claims_detail[bucket].append(entry)
            else:
                claims_detail["uncertain"].append(entry)

    # --- Reliability index ---
    reliability = None
    if quality and quality.get("verdicts"):
        vd = quality["verdicts"]
        total_v = vd["accepted"] + vd["rejected"] + vd["uncertain"] or 1
        # Formula rationale (user feedback 2026-04):
        # - ACCEPTED = verified true, full credit
        # - REJECTED = verified false AND REMOVED = system works, full credit (the bad data is filtered out)
        # - UNCERTAIN = system couldn't decide = weak signal, only 30% credit
        # Score reflects "how much the pipeline made a confident decision (yes or no)"
        decisive_ratio = (vd["accepted"] + vd["rejected"]) / total_v
        uncertain_ratio = vd["uncertain"] / total_v
        score = round(decisive_ratio * 100 + uncertain_ratio * 30)
        score = max(0, min(100, score))

        if score >= 80: grade = "A"
        elif score >= 65: grade = "B"
        elif score >= 50: grade = "C"
        elif score >= 35: grade = "D"
        else: grade = "F"

        reliability = {
            "score": score,
            "grade": grade,
            "total_claims": total_v,
            "accepted_ratio": round(vd["accepted"] / total_v, 3),
            "rejected_ratio": round(vd["rejected"] / total_v, 3),
            "uncertain_ratio": round(vd["uncertain"] / total_v, 3),
        }

    # --- Pipeline timeline ---
    timeline = []
    events = status.get("events", [])
    seen_stages = set()
    for ev in events:
        stage = ev.get("stage", "")
        if stage and stage not in seen_stages:
            seen_stages.add(stage)
            timeline.append({
                "stage": stage,
                "timestamp": ev.get("timestamp", 0),
                "message": ev.get("message", ""),
            })

    # --- LLM usage stats ---
    llm_logs_data = _load("llm_logs.json")
    llm_stats = None
    if llm_logs_data and isinstance(llm_logs_data, list):
        total_calls = len(llm_logs_data)
        total_duration = sum(l.get("duration", 0) for l in llm_logs_data)
        total_input = sum(l.get("input_len", 0) for l in llm_logs_data)
        total_output = sum(l.get("output_len", 0) for l in llm_logs_data)
        by_stage: Counter = Counter()
        for l in llm_logs_data:
            by_stage[l.get("stage", "unknown")] += 1
        llm_stats = {
            "total_calls": total_calls,
            "total_duration_s": round(total_duration, 1),
            "total_input_chars": total_input,
            "total_output_chars": total_output,
            "calls_by_stage": dict(by_stage),
        }

    # --- Coherence report ---
    coherence = None
    coherence_path = run_dir / "coherence.json"
    if coherence_path.exists():
        coherence = _load("coherence.json")

    return {
        "models": models, "search": search, "quality": quality,
        "errors": errors, "version": AIDOCGEN_VERSION,
        "sources": sources_list,
        "claims_detail": claims_detail,
        "reliability": reliability,
        "timeline": timeline,
        "llm_stats": llm_stats,
        "tags": status.get("tags", []),
        "coherence": coherence,
    }

@app.post("/v1/dossier/runs/{run_id}/reset")
async def reset_run(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    run_dir = DATA_DIR / run_id
    if run_id in _DOSSIER_TASKS: _DOSSIER_TASKS[run_id].cancel()
    for f in ["presearch.json", "planner.json", "report.md", "report.pdf", "planner_debug.json"]:
        if (run_dir / f).exists(): (run_dir / f).unlink()
    d = json.loads((run_dir / "status.json").read_text())
    d.update({"state": "running", "stage": "init", "events": [], "error": None})
    (run_dir / "status.json").write_text(json.dumps(d))
    engine = _get_engine(d.get("ollama_url"), d)
    tags = d.get("tags", [])
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, d["question"], resume=False, language=d.get("language", "fr"), coder_model=d.get("coder_model"), tags=tags, include_images=bool(d.get("include_images", True)), generate_ai_images=bool(d.get("generate_ai_images", False))))
    return {"status": "ok"}

@app.post("/v1/dossier/runs/{run_id}/cancel")
async def cancel_run(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    task = _DOSSIER_TASKS.pop(run_id, None)
    if task and not task.done(): task.cancel()
    run_dir = DATA_DIR / run_id
    if (run_dir / "status.json").exists():
        d = json.loads((run_dir / "status.json").read_text())
        d["state"] = "interrupted"
        (run_dir / "status.json").write_text(json.dumps(d))
    return {"status": "ok"}

@app.post("/v1/dossier/runs/{run_id}/resume")
async def resume_run(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    run_dir = DATA_DIR / run_id
    if not run_dir.exists(): raise HTTPException(404)
    status_path = run_dir / "status.json"
    d = json.loads(status_path.read_text())
    d.update({"state": "running", "error": None})
    status_path.write_text(json.dumps(d))
    engine = _get_engine(d.get("ollama_url"), d)
    tags = d.get("tags", [])
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, d["question"], resume=True, language=d.get("language", "fr"), coder_model=d.get("coder_model"), tags=tags, include_images=bool(d.get("include_images", True)), generate_ai_images=bool(d.get("generate_ai_images", False))))
    return {"status": "ok"}

@app.post("/v1/dossier/runs/{run_id}/approve")
async def approve_run(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    run_dir = DATA_DIR / run_id
    (run_dir / "validated.txt").write_text("ok")
    d = json.loads((run_dir / "status.json").read_text())
    engine = _get_engine(d.get("ollama_url"), d)
    tags = d.get("tags", [])
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, d["question"], resume=True, language=d.get("language", "fr"), coder_model=d.get("coder_model"), tags=tags, include_images=bool(d.get("include_images", True)), generate_ai_images=bool(d.get("generate_ai_images", False))))
    return {"status": "ok"}

@app.get("/v1/dossier/runs/{run_id}/report/pdf")
async def download_pdf(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    path = DATA_DIR / run_id / "report.pdf"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.pdf")
    raise HTTPException(404)

@app.get("/v1/dossier/runs/{run_id}/report/download")
async def download_md(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    path = DATA_DIR / run_id / "report.md"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.md")
    raise HTTPException(404)

@app.post("/v1/dossier/runs/{run_id}/publish/wordpress")
async def publish_wordpress(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    run_dir = DATA_DIR / run_id
    if not (run_dir / "report.md").exists():
        raise HTTPException(404, "report.md not found")
    try:
        from export_wordpress import publish_dossier
        result = publish_dossier(str(run_dir), status="draft")
        return result
    except Exception as e:
        raise HTTPException(500, str(e))

# --- DIFY EXPORT ---
@app.get("/v1/dossier/runs/{run_id}/export/dify-zip")
async def export_dify_zip(run_id: str, user: dict = Depends(get_current_user)):
    """Download a ZIP with all clean sources + metadata for Dify import."""
    _check_run_access(run_id, user)
    try:
        from export_dify import export_zip
        zip_path = export_zip(run_id, data_dir=str(DATA_DIR))
        return FileResponse(zip_path, filename=f"dify_{run_id}.zip", media_type="application/zip")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

class DifyPushRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    dify_url: str = "http://localhost:5001"
    email: str
    password: str
    dataset_name: str | None = None

@app.post("/v1/dossier/runs/{run_id}/export/dify")
async def export_dify_push(run_id: str, req: DifyPushRequest, user: dict = Depends(get_current_user)):
    """Push raw corpus to Dify Knowledge Base via Console API."""
    _check_run_access(run_id, user)
    try:
        from export_dify import push_to_dify
        result = await push_to_dify(
            run_id=run_id,
            dify_url=req.dify_url,
            email=req.email,
            password=req.password,
            data_dir=str(DATA_DIR),
            dataset_name=req.dataset_name,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/v1/dossier/runs/{run_id}/export/dify-verified")
async def export_dify_verified(run_id: str, req: DifyPushRequest | None = None, user: dict = Depends(get_current_user)):
    """Push VERIFIED content only (report.md + accepted claims) to Dify."""
    _check_run_access(run_id, user)
    dify_url = (req and req.dify_url) or os.getenv("DIFY_API_URL", "http://localhost:5001")
    email = (req and req.email) or os.getenv("DIFY_EMAIL", "")
    password = (req and req.password) or os.getenv("DIFY_PASSWORD", "")
    ds_name = (req and req.dataset_name) or None
    if not email or not password:
        raise HTTPException(400, "Dify credentials not configured. Set DIFY_EMAIL and DIFY_PASSWORD in env.")
    try:
        from export_dify import push_verified_to_dify
        result = await push_verified_to_dify(
            run_id=run_id, dify_url=dify_url, email=email, password=password,
            data_dir=str(DATA_DIR), dataset_name=ds_name,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/v1/dossier/runs/{run_id}/export/dify-auto")
async def export_dify_auto(run_id: str, user: dict = Depends(get_current_user)):
    """One-click push verified content to Dify using env credentials."""
    _check_run_access(run_id, user)
    dify_url = os.getenv("DIFY_API_URL", "http://localhost:5001")
    email = os.getenv("DIFY_EMAIL", "")
    password = os.getenv("DIFY_PASSWORD", "")
    if not email or not password:
        raise HTTPException(400, "DIFY_EMAIL / DIFY_PASSWORD not set in env")
    # Get question for dataset name
    run_dir = DATA_DIR / run_id
    status_data = {}
    if (run_dir / "status.json").exists():
        status_data = json.loads((run_dir / "status.json").read_text())
    ds_name = status_data.get("question", run_id)[:40]
    try:
        from export_dify import push_verified_to_dify
        result = await push_verified_to_dify(
            run_id=run_id, dify_url=dify_url, email=email, password=password,
            data_dir=str(DATA_DIR), dataset_name=ds_name,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/v1/dossier/runs/{run_id}/export/dify-dataset")
async def export_dify_dataset_api(run_id: str, user: dict = Depends(get_current_user)):
    """Push to Dify Knowledge Base via Dataset API key (preferred method)."""
    _check_run_access(run_id, user)
    api_url = os.getenv("DIFY_DATASET_API_URL", "")
    api_key = os.getenv("DIFY_DATASET_API_KEY", "")
    if not api_url or not api_key:
        raise HTTPException(400, "DIFY_DATASET_API_URL / DIFY_DATASET_API_KEY not set in env")
    run_dir = DATA_DIR / run_id
    status_data = {}
    if (run_dir / "status.json").exists():
        status_data = json.loads((run_dir / "status.json").read_text())
    ds_name = status_data.get("question", run_id)[:40]
    try:
        from export_dify import push_via_dataset_api
        result = await push_via_dataset_api(
            run_id=run_id, api_url=api_url, api_key=api_key,
            data_dir=str(DATA_DIR), dataset_name=ds_name,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))

# --- PDF / DOCUMENT → MARKDOWN CONVERSION ---
PDF_MODES = {
    "simple": {"label": "PDF — Texte court (1 appel LLM)", "default_model": "glm-5.1:cloud"},
    "huge":   {"label": "PDF — Texte volumineux (chunks parallèles)", "default_model": "glm-5.1:cloud"},
    "vision": {"label": "PDF — Scans / screenshots (vision)", "default_model": "qwen3-vl:235b-cloud"},
    "parse":  {"label": "Cloud Parse (Office, HTML — Firecrawl)", "default_model": ""},
}
PARSE_EXTS = {".pdf", ".docx", ".doc", ".odt", ".rtf", ".xlsx", ".xls", ".html", ".htm"}

def _pdf_status_path(job_id: str) -> Path:
    return PDF_JOBS_DIR / job_id / "status.json"

def _pdf_log_path(job_id: str) -> Path:
    return PDF_JOBS_DIR / job_id / "run.log"

def _read_pdf_status(job_id: str) -> dict | None:
    p = _pdf_status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _write_pdf_status(job_id: str, data: dict):
    _pdf_status_path(job_id).write_text(json.dumps(data))

async def _run_pdf_job(job_id: str, mode: str, model: str, pdf_path: Path, out_path: Path):
    job_dir = PDF_JOBS_DIR / job_id
    log_path = _pdf_log_path(job_id)
    status = _read_pdf_status(job_id) or {}

    async def _update(**fields):
        cur = _read_pdf_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_pdf_status(job_id, cur)

    try:
        # CLOUD PARSE — sync call to Firecrawl /v2/parse, no subprocess.
        if mode == "parse":
            api_key = os.getenv("FIRECRAWL_API_KEY", "")
            if not api_key:
                await _update(state="failed", stage="failed", error="FIRECRAWL_API_KEY not set")
                return
            await _update(state="running", stage="parsing")
            with log_path.open("w") as logf:
                logf.write(f"firecrawl parse on {pdf_path.name}\n")
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    with pdf_path.open("rb") as fh:
                        files = {"file": (pdf_path.name, fh, "application/octet-stream")}
                        resp = await client.post(
                            "https://api.firecrawl.dev/v2/parse",
                            headers={"Authorization": f"Bearer {api_key}"},
                            files=files,
                        )
                if resp.status_code >= 300:
                    await _update(state="failed", stage="failed",
                                  error=f"firecrawl {resp.status_code}: {resp.text[:300]}")
                    return
                data = resp.json()
                if not data.get("success"):
                    await _update(state="failed", stage="failed",
                                  error=data.get("error") or "parse failed")
                    return
                payload = data.get("data") or {}
                md = payload.get("markdown", "")
                meta = payload.get("metadata") or {}
                title = meta.get("title") or pdf_path.stem
                out_path.write_text(f"# {title}\n\n{md}\n", encoding="utf-8")
                await _update(state="completed", stage="completed",
                              output_size=out_path.stat().st_size)
            except Exception as e:
                await _update(state="failed", stage="failed", error=str(e))
            return

        if mode == "simple":
            raw_path = job_dir / "raw.txt"
            proc = await asyncio.create_subprocess_exec(
                "pdftotext", "-layout", str(pdf_path), str(raw_path),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            out, _ = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"pdftotext failed: {out.decode('utf-8', errors='replace')[:500]}")
            cmd = ["python3", "-u", str(SCRIPTS_DIR / "convert_pdf_md.py"),
                   str(raw_path), str(out_path), model]
        elif mode == "huge":
            cmd = ["python3", "-u", str(SCRIPTS_DIR / "process_huge.py"),
                   str(pdf_path), str(out_path), model]
        elif mode == "vision":
            cmd = ["python3", "-u", str(SCRIPTS_DIR / "process_vision.py"),
                   str(pdf_path), str(out_path), model]
        else:
            raise RuntimeError(f"unknown mode {mode!r}")

        await _update(state="running", stage="converting")
        with log_path.open("w") as logf:
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT,
            )
            rc = await proc.wait()

        if rc != 0:
            tail = log_path.read_text()[-2000:] if log_path.exists() else ""
            await _update(state="failed", stage="failed", error=f"exit={rc}", log_tail=tail)
            return
        if not out_path.exists():
            await _update(state="failed", stage="failed", error="output not produced")
            return
        await _update(
            state="completed", stage="completed",
            output_size=out_path.stat().st_size,
        )
    except asyncio.CancelledError:
        await _update(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _update(state="failed", stage="failed", error=str(e))
    finally:
        _PDF_TASKS.pop(job_id, None)

@app.get("/v1/pdf/modes")
async def pdf_modes(user: dict = Depends(get_current_user)):
    return {"modes": [{"key": k, **v} for k, v in PDF_MODES.items()]}

@app.get("/v1/pdf/jobs")
async def pdf_list_jobs(limit: int = 30, user: dict = Depends(get_current_user)):
    jobs = []
    is_admin = _is_admin(user)
    if PDF_JOBS_DIR.exists():
        dirs = sorted([d for d in PDF_JOBS_DIR.iterdir() if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs:
            data = _read_pdf_status(d.name)
            if not data: continue
            if not is_admin and data.get("owner_id") != user["id"]:
                continue
            if data.get("state") == "running" and d.name not in _PDF_TASKS:
                data["state"] = "interrupted"
            jobs.append(data)
            if len(jobs) >= limit:
                break
    return {"data": jobs}

@app.post("/v1/pdf/convert")
async def pdf_convert(
    file: UploadFile = File(...),
    mode: str = Form("simple"),
    model: str = Form(""),
    user: dict = Depends(get_current_user),
):
    if mode not in PDF_MODES:
        raise HTTPException(400, f"unknown mode: {mode}")
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if mode == "parse":
        if ext not in PARSE_EXTS:
            raise HTTPException(400, f"parse mode supports: {sorted(PARSE_EXTS)}")
    else:
        if ext != ".pdf":
            raise HTTPException(400, f"mode {mode!r} only accepts .pdf — use 'parse' for {ext}")
    chosen_model = model.strip() or PDF_MODES[mode]["default_model"]

    job_id = f"pdf-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = PDF_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = job_dir / f"input{ext or '.pdf'}"
    out_path = job_dir / (Path(safe_name).stem + ".md")

    with pdf_path.open("wb") as f:
        while chunk := await file.read(1024 * 1024):
            f.write(chunk)

    _write_pdf_status(job_id, {
        "job_id": job_id, "filename": safe_name,
        "mode": mode, "model": chosen_model,
        "state": "queued", "stage": "queued",
        "size": pdf_path.stat().st_size,
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
        "output_name": out_path.name,
    })
    _PDF_TASKS[job_id] = asyncio.create_task(
        _run_pdf_job(job_id, mode, chosen_model, pdf_path, out_path)
    )
    return {"job_id": job_id}

@app.get("/v1/pdf/jobs/{job_id}")
async def pdf_job_status(job_id: str, user: dict = Depends(get_current_user)):
    _check_pdf_access(job_id, user)
    data = _read_pdf_status(job_id)
    if not data:
        raise HTTPException(404)
    log_path = _pdf_log_path(job_id)
    if log_path.exists():
        try:
            tail = log_path.read_text()[-4000:]
        except Exception:
            tail = ""
        data["log_tail"] = tail
    return data

@app.get("/v1/pdf/jobs/{job_id}/download")
async def pdf_job_download(job_id: str, user: dict = Depends(get_current_user)):
    _check_pdf_access(job_id, user)
    data = _read_pdf_status(job_id)
    if not data:
        raise HTTPException(404)
    md_path = PDF_JOBS_DIR / job_id / data.get("output_name", "output.md")
    if not md_path.exists():
        raise HTTPException(404, "output not ready")
    return FileResponse(md_path, filename=md_path.name, media_type="text/markdown")

@app.post("/v1/pdf/jobs/{job_id}/cancel")
async def pdf_job_cancel(job_id: str, user: dict = Depends(get_current_user)):
    _check_pdf_access(job_id, user)
    task = _PDF_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    data = _read_pdf_status(job_id) or {}
    data.update({"state": "cancelled", "stage": "cancelled", "updated_at": int(time.time())})
    _write_pdf_status(job_id, data)
    return {"status": "ok"}

@app.delete("/v1/pdf/jobs/{job_id}")
async def pdf_job_delete(job_id: str, user: dict = Depends(get_current_user)):
    _check_pdf_access(job_id, user)
    task = _PDF_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    job_dir = PDF_JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return {"status": "ok"}

# --- VIDEO → MARKDOWN CONVERSION ---
VIDEO_MODES = {
    "vision": {"label": "Vision seule (frames décrites)", "default_model": "nemotron3:33b"},
    "audio":  {"label": "Audio seul (transcription)",      "default_model": "nemotron3:33b"},
    "full":   {"label": "Audio + Vision (recommandé)",     "default_model": "nemotron3:33b"},
}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv"}
DEFAULT_SYNTHESIS_MODEL = "deepseek-v4-pro"

VIDEO_SYNTHESIS_PROMPT = (
    "You are given the chronological transcript of a video, segmented into chunks "
    "with [HH:MM:SS] timestamps. Each chunk contains a verbatim transcript of the "
    "speech and may include a short scene description prefixed with '> Scene:'.\n\n"
    "Your task: produce a clean, coherent technical synthesis in Markdown.\n\n"
    "Rules:\n"
    "- Identify the main topics in order and use them as `## ` section headers.\n"
    "- Keep `[HH:MM:SS]` timestamps as inline references when introducing a topic.\n"
    "- Preserve verbatim: proper names, quoted code, commands, file paths, numbers, "
    "technical terms, acronyms.\n"
    "- Remove filler words (\"euh\", \"donc voilà\", \"en fait\"), repetitions, false starts.\n"
    "- Stay in the source language (do NOT translate).\n"
    "- Do NOT add information that is not present in the source.\n"
    "- Do NOT summarize away technical detail — the goal is a clean, coherent rewrite, "
    "not a short summary. The output should be roughly the same length as the source.\n"
    "- Output ONLY valid Markdown, no preamble, no closing remarks.\n"
)

def _video_status_path(job_id: str) -> Path:
    return VIDEO_JOBS_DIR / job_id / "status.json"

def _video_log_path(job_id: str) -> Path:
    return VIDEO_JOBS_DIR / job_id / "run.log"

def _read_video_status(job_id: str) -> dict | None:
    p = _video_status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _write_video_status(job_id: str, data: dict):
    _video_status_path(job_id).write_text(json.dumps(data))

def _video_synthesis_path(job_id: str) -> Path:
    data = _read_video_status(job_id) or {}
    out_name = data.get("output_name") or "output.md"
    stem = Path(out_name).stem
    return VIDEO_JOBS_DIR / job_id / f"{stem}_synthesis.md"

async def _synthesize_video_md(job_id: str, model: str) -> dict:
    """Send the assembled raw .md to Ollama Cloud, save synthesis next to it."""
    job_dir = VIDEO_JOBS_DIR / job_id
    data = _read_video_status(job_id) or {}
    raw_path = job_dir / (data.get("output_name") or "output.md")
    if not raw_path.exists():
        raise HTTPException(409, "raw .md not ready yet")
    syn_path = _video_synthesis_path(job_id)

    raw_text = raw_path.read_text(encoding="utf-8")
    cloud_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "OLLAMA_API_KEY not set in env")

    async def _update_synth(state: str, **fields):
        cur = _read_video_status(job_id) or {}
        cur["synthesis_state"] = state
        cur["synthesis_model"] = model
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_video_status(job_id, cur)

    await _update_synth("running", synthesis_error=None)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VIDEO_SYNTHESIS_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=1800) as client:
            r = await client.post(f"{cloud_url}/api/chat", json=payload, headers=headers)
            r.raise_for_status()
            d = r.json()
        content = (d.get("message") or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("empty synthesis response")
        syn_path.write_text(content, encoding="utf-8")
        await _update_synth("completed", synthesis_size=syn_path.stat().st_size,
                            synthesis_name=syn_path.name)
        return {"status": "ok", "size": syn_path.stat().st_size}
    except Exception as e:
        await _update_synth("failed", synthesis_error=str(e))
        raise HTTPException(500, f"synthesis failed: {e}")

async def _run_video_job(job_id: str, mode: str, model: str, chunk: float,
                         parallel: int, overlap: float, video_path: Path, out_path: Path,
                         resumed: bool = False, synthesis_model: str | None = None):
    log_path = _video_log_path(job_id)
    state_dir = VIDEO_JOBS_DIR / job_id / "chunks"

    async def _update(**fields):
        cur = _read_video_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_video_status(job_id, cur)

    try:
        cmd = [
            "python3", "-u", str(SCRIPTS_DIR / "process_video.py"),
            str(video_path), str(out_path),
            "--mode", mode, "--model", model,
            "--chunk", str(chunk), "--parallel", str(parallel),
            "--overlap", str(overlap),
            "--state-dir", str(state_dir),
        ]
        await _update(state="running", stage="converting", error=None)
        # Append on resume so we keep the previous log; truncate on first run.
        with log_path.open("a" if resumed else "w") as logf:
            if resumed:
                logf.write(f"\n--- RESUME @ {int(time.time())} ---\n")
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT,
            )
            rc = await proc.wait()

        if rc != 0:
            tail = log_path.read_text()[-2000:] if log_path.exists() else ""
            await _update(state="failed", stage="failed", error=f"exit={rc}", log_tail=tail)
            return
        if not out_path.exists():
            await _update(state="failed", stage="failed", error="output not produced")
            return
        await _update(state="completed", stage="completed", output_size=out_path.stat().st_size)
        # Auto-trigger synthesis if a model was requested at submit time
        if synthesis_model:
            try:
                await _synthesize_video_md(job_id, synthesis_model)
            except Exception as e:
                # Synthesis is best-effort — don't fail the whole job if it errors.
                cur = _read_video_status(job_id) or {}
                cur["synthesis_state"] = "failed"
                cur["synthesis_error"] = str(e)
                _write_video_status(job_id, cur)
    except asyncio.CancelledError:
        await _update(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _update(state="failed", stage="failed", error=str(e))
    finally:
        _VIDEO_TASKS.pop(job_id, None)

@app.get("/v1/video/modes")
async def video_modes(user: dict = Depends(get_current_user)):
    return {"modes": [{"key": k, **v} for k, v in VIDEO_MODES.items()]}

@app.get("/v1/video/jobs")
async def video_list_jobs(limit: int = 30, user: dict = Depends(get_current_user)):
    jobs = []
    is_admin = _is_admin(user)
    if VIDEO_JOBS_DIR.exists():
        dirs = sorted([d for d in VIDEO_JOBS_DIR.iterdir() if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs:
            data = _read_video_status(d.name)
            if not data: continue
            if not is_admin and data.get("owner_id") != user["id"]:
                continue
            if data.get("state") == "running" and d.name not in _VIDEO_TASKS:
                data["state"] = "interrupted"
            _attach_video_progress(d.name, data)
            jobs.append(data)
            if len(jobs) >= limit:
                break
    return {"data": jobs}

@app.post("/v1/video/convert")
async def video_convert(
    file: UploadFile = File(...),
    mode: str = Form("full"),
    model: str = Form(""),
    chunk: float = Form(30.0),
    parallel: int = Form(2),
    overlap: float = Form(1.5),
    synthesis: bool = Form(False),
    synthesis_model: str = Form(""),
    user: dict = Depends(get_current_user),
):
    if mode not in VIDEO_MODES:
        raise HTTPException(400, f"unknown mode: {mode}")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"unsupported extension {ext!r}; supported: {sorted(VIDEO_EXTS)}")
    if chunk < 5 or chunk > 600:
        raise HTTPException(400, "chunk must be between 5 and 600 seconds")
    if parallel < 1 or parallel > 8:
        raise HTTPException(400, "parallel must be between 1 and 8")
    if overlap < 0 or overlap > 30:
        raise HTTPException(400, "overlap must be between 0 and 30 seconds")

    chosen_model = model.strip() or VIDEO_MODES[mode]["default_model"]
    chosen_synth = (synthesis_model.strip() or DEFAULT_SYNTHESIS_MODEL) if synthesis else None
    job_id = f"vid-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = VIDEO_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    video_path = job_dir / f"input{ext}"
    out_path = job_dir / (Path(safe_name).stem + ".md")

    with video_path.open("wb") as f:
        while data := await file.read(1024 * 1024):
            f.write(data)

    _write_video_status(job_id, {
        "job_id": job_id, "filename": safe_name,
        "mode": mode, "model": chosen_model,
        "chunk": chunk, "parallel": parallel, "overlap": overlap,
        "synthesis_requested": bool(chosen_synth),
        "synthesis_model": chosen_synth,
        "state": "queued", "stage": "queued",
        "size": video_path.stat().st_size,
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
        "output_name": out_path.name,
    })
    _VIDEO_TASKS[job_id] = asyncio.create_task(
        _run_video_job(job_id, mode, chosen_model, chunk, parallel, overlap, video_path, out_path,
                       synthesis_model=chosen_synth)
    )
    return {"job_id": job_id}

def _attach_video_progress(job_id: str, data: dict) -> dict:
    """Merge progress.json (written by the script) into the status payload."""
    p = VIDEO_JOBS_DIR / job_id / "progress.json"
    if p.exists():
        try:
            data["progress"] = json.loads(p.read_text())
        except Exception:
            pass
    return data

@app.get("/v1/video/jobs/{job_id}")
async def video_job_status(job_id: str, user: dict = Depends(get_current_user)):
    _check_video_access(job_id, user)
    data = _read_video_status(job_id)
    if not data:
        raise HTTPException(404)
    _attach_video_progress(job_id, data)
    log_path = _video_log_path(job_id)
    if log_path.exists():
        try:
            data["log_tail"] = log_path.read_text()[-4000:]
        except Exception:
            data["log_tail"] = ""
    return data

@app.post("/v1/video/jobs/{job_id}/resume")
async def video_job_resume(job_id: str, user: dict = Depends(get_current_user)):
    _check_video_access(job_id, user)
    if job_id in _VIDEO_TASKS and not _VIDEO_TASKS[job_id].done():
        raise HTTPException(409, "job already running")
    data = _read_video_status(job_id) or {}
    job_dir = VIDEO_JOBS_DIR / job_id
    video_path = next((p for p in job_dir.glob("input.*") if p.is_file()), None)
    if not video_path or not video_path.exists():
        raise HTTPException(404, "input video missing")
    out_name = data.get("output_name") or "output.md"
    out_path = job_dir / out_name
    _VIDEO_TASKS[job_id] = asyncio.create_task(
        _run_video_job(
            job_id,
            data.get("mode", "full"),
            data.get("model", "nemotron3:33b"),
            float(data.get("chunk", 30)),
            int(data.get("parallel", 2)),
            float(data.get("overlap", 1.5)),
            video_path, out_path, resumed=True,
        )
    )
    return {"status": "ok"}

@app.get("/v1/video/jobs/{job_id}/download")
async def video_job_download(job_id: str, type: str = "raw",
                             user: dict = Depends(get_current_user)):
    _check_video_access(job_id, user)
    data = _read_video_status(job_id) or {}
    if type == "synthesis":
        md_path = _video_synthesis_path(job_id)
        if not md_path.exists():
            raise HTTPException(404, "synthesis not ready")
    else:
        md_path = VIDEO_JOBS_DIR / job_id / data.get("output_name", "output.md")
        if not md_path.exists():
            raise HTTPException(404, "output not ready")
    return FileResponse(md_path, filename=md_path.name, media_type="text/markdown")

class SynthRequest(BaseModel):
    model: str | None = None

@app.post("/v1/video/jobs/{job_id}/synthesize")
async def video_job_synthesize(job_id: str, req: SynthRequest | None = None,
                               user: dict = Depends(get_current_user)):
    _check_video_access(job_id, user)
    data = _read_video_status(job_id) or {}
    if data.get("state") != "completed":
        raise HTTPException(409, "raw conversion not completed yet")
    model = (req and req.model and req.model.strip()) or data.get("synthesis_model") or DEFAULT_SYNTHESIS_MODEL
    return await _synthesize_video_md(job_id, model)

@app.post("/v1/video/jobs/{job_id}/cancel")
async def video_job_cancel(job_id: str, user: dict = Depends(get_current_user)):
    _check_video_access(job_id, user)
    task = _VIDEO_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    data = _read_video_status(job_id) or {}
    data.update({"state": "cancelled", "stage": "cancelled", "updated_at": int(time.time())})
    _write_video_status(job_id, data)
    return {"status": "ok"}

@app.delete("/v1/video/jobs/{job_id}")
async def video_job_delete(job_id: str, user: dict = Depends(get_current_user)):
    _check_video_access(job_id, user)
    task = _VIDEO_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    job_dir = VIDEO_JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return {"status": "ok"}

# --- AUDIO → MARKDOWN CONVERSION ---
AUDIO_DEFAULT_MODEL = "nemotron3:33b"
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac", ".wma", ".webm"}

def _audio_status_path(job_id: str) -> Path:
    return AUDIO_JOBS_DIR / job_id / "status.json"

def _audio_log_path(job_id: str) -> Path:
    return AUDIO_JOBS_DIR / job_id / "run.log"

def _read_audio_status(job_id: str) -> dict | None:
    p = _audio_status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _write_audio_status(job_id: str, data: dict):
    _audio_status_path(job_id).write_text(json.dumps(data))

def _audio_synthesis_path(job_id: str) -> Path:
    data = _read_audio_status(job_id) or {}
    out_name = data.get("output_name") or "output.md"
    stem = Path(out_name).stem
    return AUDIO_JOBS_DIR / job_id / f"{stem}_synthesis.md"

async def _synthesize_audio_md(job_id: str, model: str) -> dict:
    job_dir = AUDIO_JOBS_DIR / job_id
    data = _read_audio_status(job_id) or {}
    raw_path = job_dir / (data.get("output_name") or "output.md")
    if not raw_path.exists():
        raise HTTPException(409, "raw .md not ready yet")
    syn_path = _audio_synthesis_path(job_id)

    raw_text = raw_path.read_text(encoding="utf-8")
    cloud_url = os.getenv("OLLAMA_BASE_URL", "https://ollama.com").rstrip("/")
    api_key = os.getenv("OLLAMA_API_KEY", "")
    if not api_key:
        raise HTTPException(500, "OLLAMA_API_KEY not set in env")

    async def _u(state: str, **fields):
        cur = _read_audio_status(job_id) or {}
        cur["synthesis_state"] = state
        cur["synthesis_model"] = model
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_audio_status(job_id, cur)

    await _u("running", synthesis_error=None)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": VIDEO_SYNTHESIS_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=1800) as client:
            r = await client.post(f"{cloud_url}/api/chat", json=payload, headers=headers)
            r.raise_for_status()
            d = r.json()
        content = (d.get("message") or {}).get("content", "").strip()
        if not content:
            raise RuntimeError("empty synthesis response")
        syn_path.write_text(content, encoding="utf-8")
        await _u("completed", synthesis_size=syn_path.stat().st_size,
                 synthesis_name=syn_path.name)
        return {"status": "ok", "size": syn_path.stat().st_size}
    except Exception as e:
        await _u("failed", synthesis_error=str(e))
        raise HTTPException(500, f"synthesis failed: {e}")

async def _run_audio_job(job_id: str, model: str, chunk: float, parallel: int,
                         overlap: float, audio_path: Path, out_path: Path,
                         resumed: bool = False, synthesis_model: str | None = None):
    log_path = _audio_log_path(job_id)
    state_dir = AUDIO_JOBS_DIR / job_id / "chunks"

    async def _update(**fields):
        cur = _read_audio_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_audio_status(job_id, cur)

    try:
        cmd = [
            "python3", "-u", str(SCRIPTS_DIR / "process_video.py"),
            str(audio_path), str(out_path),
            "--mode", "audio", "--model", model,
            "--chunk", str(chunk), "--parallel", str(parallel),
            "--overlap", str(overlap),
            "--state-dir", str(state_dir),
        ]
        await _update(state="running", stage="converting", error=None)
        with log_path.open("a" if resumed else "w") as logf:
            if resumed:
                logf.write(f"\n--- RESUME @ {int(time.time())} ---\n")
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT,
            )
            rc = await proc.wait()

        if rc != 0:
            tail = log_path.read_text()[-2000:] if log_path.exists() else ""
            await _update(state="failed", stage="failed", error=f"exit={rc}", log_tail=tail)
            return
        if not out_path.exists():
            await _update(state="failed", stage="failed", error="output not produced")
            return
        await _update(state="completed", stage="completed", output_size=out_path.stat().st_size)
        if synthesis_model:
            try:
                await _synthesize_audio_md(job_id, synthesis_model)
            except Exception as e:
                cur = _read_audio_status(job_id) or {}
                cur["synthesis_state"] = "failed"
                cur["synthesis_error"] = str(e)
                _write_audio_status(job_id, cur)
    except asyncio.CancelledError:
        await _update(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _update(state="failed", stage="failed", error=str(e))
    finally:
        _AUDIO_TASKS.pop(job_id, None)

def _attach_audio_progress(job_id: str, data: dict) -> dict:
    p = AUDIO_JOBS_DIR / job_id / "progress.json"
    if p.exists():
        try:
            data["progress"] = json.loads(p.read_text())
        except Exception:
            pass
    return data

@app.get("/v1/audio/jobs")
async def audio_list_jobs(limit: int = 30, user: dict = Depends(get_current_user)):
    jobs = []
    is_admin = _is_admin(user)
    if AUDIO_JOBS_DIR.exists():
        dirs = sorted([d for d in AUDIO_JOBS_DIR.iterdir() if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs:
            data = _read_audio_status(d.name)
            if not data: continue
            if not is_admin and data.get("owner_id") != user["id"]:
                continue
            if data.get("state") == "running" and d.name not in _AUDIO_TASKS:
                data["state"] = "interrupted"
            _attach_audio_progress(d.name, data)
            jobs.append(data)
            if len(jobs) >= limit:
                break
    return {"data": jobs}

@app.post("/v1/audio/convert")
async def audio_convert(
    file: UploadFile = File(...),
    model: str = Form(""),
    chunk: float = Form(60.0),
    parallel: int = Form(2),
    overlap: float = Form(1.5),
    synthesis: bool = Form(False),
    synthesis_model: str = Form(""),
    user: dict = Depends(get_current_user),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in AUDIO_EXTS:
        raise HTTPException(400, f"unsupported extension {ext!r}; supported: {sorted(AUDIO_EXTS)}")
    if chunk < 5 or chunk > 600:
        raise HTTPException(400, "chunk must be between 5 and 600 seconds")
    if parallel < 1 or parallel > 8:
        raise HTTPException(400, "parallel must be between 1 and 8")
    if overlap < 0 or overlap > 30:
        raise HTTPException(400, "overlap must be between 0 and 30 seconds")

    chosen_model = model.strip() or AUDIO_DEFAULT_MODEL
    chosen_synth = (synthesis_model.strip() or DEFAULT_SYNTHESIS_MODEL) if synthesis else None
    job_id = f"aud-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = AUDIO_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    audio_path = job_dir / f"input{ext}"
    out_path = job_dir / (Path(safe_name).stem + ".md")

    with audio_path.open("wb") as f:
        while data := await file.read(1024 * 1024):
            f.write(data)

    _write_audio_status(job_id, {
        "job_id": job_id, "filename": safe_name,
        "model": chosen_model,
        "chunk": chunk, "parallel": parallel, "overlap": overlap,
        "synthesis_requested": bool(chosen_synth),
        "synthesis_model": chosen_synth,
        "state": "queued", "stage": "queued",
        "size": audio_path.stat().st_size,
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
        "output_name": out_path.name,
    })
    _AUDIO_TASKS[job_id] = asyncio.create_task(
        _run_audio_job(job_id, chosen_model, chunk, parallel, overlap, audio_path, out_path,
                       synthesis_model=chosen_synth)
    )
    return {"job_id": job_id}

@app.get("/v1/audio/jobs/{job_id}")
async def audio_job_status(job_id: str, user: dict = Depends(get_current_user)):
    _check_audio_access(job_id, user)
    data = _read_audio_status(job_id)
    if not data:
        raise HTTPException(404)
    _attach_audio_progress(job_id, data)
    log_path = _audio_log_path(job_id)
    if log_path.exists():
        try:
            data["log_tail"] = log_path.read_text()[-4000:]
        except Exception:
            data["log_tail"] = ""
    return data

@app.post("/v1/audio/jobs/{job_id}/resume")
async def audio_job_resume(job_id: str, user: dict = Depends(get_current_user)):
    _check_audio_access(job_id, user)
    if job_id in _AUDIO_TASKS and not _AUDIO_TASKS[job_id].done():
        raise HTTPException(409, "job already running")
    data = _read_audio_status(job_id) or {}
    job_dir = AUDIO_JOBS_DIR / job_id
    audio_path = next((p for p in job_dir.glob("input.*") if p.is_file()), None)
    if not audio_path or not audio_path.exists():
        raise HTTPException(404, "input audio missing")
    out_path = job_dir / (data.get("output_name") or "output.md")
    _AUDIO_TASKS[job_id] = asyncio.create_task(
        _run_audio_job(
            job_id,
            data.get("model", AUDIO_DEFAULT_MODEL),
            float(data.get("chunk", 60)),
            int(data.get("parallel", 2)),
            float(data.get("overlap", 1.5)),
            audio_path, out_path, resumed=True,
            synthesis_model=data.get("synthesis_model") if data.get("synthesis_requested") else None,
        )
    )
    return {"status": "ok"}

@app.get("/v1/audio/jobs/{job_id}/download")
async def audio_job_download(job_id: str, type: str = "raw",
                             user: dict = Depends(get_current_user)):
    _check_audio_access(job_id, user)
    data = _read_audio_status(job_id) or {}
    if type == "synthesis":
        md_path = _audio_synthesis_path(job_id)
        if not md_path.exists():
            raise HTTPException(404, "synthesis not ready")
    else:
        md_path = AUDIO_JOBS_DIR / job_id / data.get("output_name", "output.md")
        if not md_path.exists():
            raise HTTPException(404, "output not ready")
    return FileResponse(md_path, filename=md_path.name, media_type="text/markdown")

@app.post("/v1/audio/jobs/{job_id}/synthesize")
async def audio_job_synthesize(job_id: str, req: SynthRequest | None = None,
                               user: dict = Depends(get_current_user)):
    _check_audio_access(job_id, user)
    data = _read_audio_status(job_id) or {}
    if data.get("state") != "completed":
        raise HTTPException(409, "raw conversion not completed yet")
    model = (req and req.model and req.model.strip()) or data.get("synthesis_model") or DEFAULT_SYNTHESIS_MODEL
    return await _synthesize_audio_md(job_id, model)

@app.post("/v1/audio/jobs/{job_id}/cancel")
async def audio_job_cancel(job_id: str, user: dict = Depends(get_current_user)):
    _check_audio_access(job_id, user)
    task = _AUDIO_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    data = _read_audio_status(job_id) or {}
    data.update({"state": "cancelled", "stage": "cancelled", "updated_at": int(time.time())})
    _write_audio_status(job_id, data)
    return {"status": "ok"}

@app.delete("/v1/audio/jobs/{job_id}")
async def audio_job_delete(job_id: str, user: dict = Depends(get_current_user)):
    _check_audio_access(job_id, user)
    task = _AUDIO_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    job_dir = AUDIO_JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return {"status": "ok"}

# --- WEB → MARKDOWN (Firecrawl) ---
def _web_status_path(job_id: str) -> Path:
    return WEB_JOBS_DIR / job_id / "status.json"

def _read_web_status(job_id: str) -> dict | None:
    p = _web_status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _write_web_status(job_id: str, data: dict):
    _web_status_path(job_id).write_text(json.dumps(data))

def _slug_from_url(u: str) -> str:
    try:
        p = urlparse(u)
        s = (p.netloc + p.path).strip("/").replace("/", "_")
        s = re.sub(r"[^A-Za-z0-9._-]", "_", s)[:120]
        return s or "page"
    except Exception:
        return "page"

async def _firecrawl_scrape_one(url: str) -> dict:
    api_key = os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key:
        raise RuntimeError("FIRECRAWL_API_KEY not set")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"url": url, "formats": ["markdown"]}
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post("https://api.firecrawl.dev/v1/scrape", json=payload, headers=headers)
        r.raise_for_status()
        d = r.json()
    if not d.get("success"):
        raise RuntimeError(d.get("error") or "scrape failed")
    data = d.get("data", {}) or {}
    return {
        "markdown": data.get("markdown", ""),
        "title": (data.get("metadata") or {}).get("title", ""),
        "url": (data.get("metadata") or {}).get("sourceURL", url),
    }

async def _run_web_scrape(job_id: str, url: str):
    job_dir = WEB_JOBS_DIR / job_id

    async def _u(**fields):
        cur = _read_web_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_web_status(job_id, cur)

    try:
        await _u(state="running", stage="scraping", error=None)
        page = await _firecrawl_scrape_one(url)
        slug = _slug_from_url(url)
        out = job_dir / f"{slug}.md"
        body = page["markdown"] or ""
        title = page.get("title") or url
        out.write_text(f"# {title}\n\n_Source: {url}_\n\n{body}\n", encoding="utf-8")
        await _u(state="completed", stage="completed",
                 output_name=out.name, output_size=out.stat().st_size,
                 page_title=title, page_url=page.get("url", url))
    except asyncio.CancelledError:
        await _u(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _u(state="failed", stage="failed", error=str(e))
    finally:
        _WEB_TASKS.pop(job_id, None)

async def _run_web_crawl(job_id: str, root_url: str, max_pages: int):
    job_dir = WEB_JOBS_DIR / job_id
    pages_dir = job_dir / "pages"
    pages_dir.mkdir(exist_ok=True)
    api_key = os.getenv("FIRECRAWL_API_KEY", "")

    async def _u(**fields):
        cur = _read_web_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_web_status(job_id, cur)

    try:
        if not api_key:
            raise RuntimeError("FIRECRAWL_API_KEY not set")
        await _u(state="running", stage="submitting", error=None)
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {"url": root_url, "limit": max_pages,
                   "scrapeOptions": {"formats": ["markdown"]}}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post("https://api.firecrawl.dev/v1/crawl", json=payload, headers=headers)
            r.raise_for_status()
            d = r.json()
        if not d.get("success") or not d.get("id"):
            raise RuntimeError(d.get("error") or "crawl submission failed")
        crawl_id = d["id"]
        await _u(stage="crawling", crawl_id=crawl_id)

        poll_url = f"https://api.firecrawl.dev/v1/crawl/{crawl_id}"
        # Up to ~30 min total: 360 polls × 5s
        for _ in range(360):
            await asyncio.sleep(5)
            async with httpx.AsyncClient(timeout=30.0) as client:
                pr = await client.get(poll_url, headers=headers)
            if pr.status_code != 200:
                continue
            res = pr.json()
            status = res.get("status", "")
            done = res.get("completed", 0) or 0
            total = res.get("total", 0) or max_pages
            await _u(progress={"done": done, "total": total,
                               "pct": round((done / total * 100), 1) if total else 0})
            if status == "completed":
                pages = res.get("data", []) or []
                stem = _slug_from_url(root_url)
                wrote = 0
                index_lines = [f"# {stem}\n", f"_Source root: {root_url} · {len(pages)} pages_\n", "## Pages\n"]
                for item in pages:
                    md = item.get("markdown") or ""
                    meta = item.get("metadata") or {}
                    page_url = meta.get("sourceURL") or root_url
                    title = meta.get("title") or page_url
                    name = _slug_from_url(page_url) + ".md"
                    (pages_dir / name).write_text(
                        f"# {title}\n\n_Source: {page_url}_\n\n{md}\n",
                        encoding="utf-8",
                    )
                    index_lines.append(f"- [{title}](pages/{name}) — {page_url}")
                    wrote += 1
                (job_dir / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

                # Bundle into ZIP
                zip_path = job_dir / f"{stem}.zip"
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(job_dir / "INDEX.md", arcname="INDEX.md")
                    for f in pages_dir.iterdir():
                        z.write(f, arcname=f"pages/{f.name}")
                await _u(state="completed", stage="completed",
                         output_name=zip_path.name, output_size=zip_path.stat().st_size,
                         pages_count=wrote)
                return
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"firecrawl status: {status}")
        raise RuntimeError("crawl timed out after 30 minutes")
    except asyncio.CancelledError:
        await _u(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _u(state="failed", stage="failed", error=str(e))
    finally:
        _WEB_TASKS.pop(job_id, None)

@app.get("/v1/web/jobs")
async def web_list_jobs(limit: int = 30, user: dict = Depends(get_current_user)):
    jobs = []
    is_admin = _is_admin(user)
    if WEB_JOBS_DIR.exists():
        dirs = sorted([d for d in WEB_JOBS_DIR.iterdir() if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs:
            data = _read_web_status(d.name)
            if not data: continue
            if not is_admin and data.get("owner_id") != user["id"]:
                continue
            if data.get("state") == "running" and d.name not in _WEB_TASKS:
                data["state"] = "interrupted"
            jobs.append(data)
            if len(jobs) >= limit:
                break
    return {"data": jobs}

class ScrapeRequest(BaseModel):
    url: str

class CrawlRequest(BaseModel):
    url: str
    max_pages: int = 25

@app.post("/v1/web/scrape")
async def web_scrape(req: ScrapeRequest, user: dict = Depends(get_current_user)):
    if not req.url.strip().startswith(("http://", "https://")):
        raise HTTPException(400, "url must start with http:// or https://")
    job_id = f"web-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = WEB_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_web_status(job_id, {
        "job_id": job_id, "kind": "scrape", "url": req.url.strip(),
        "state": "queued", "stage": "queued",
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
    })
    _WEB_TASKS[job_id] = asyncio.create_task(_run_web_scrape(job_id, req.url.strip()))
    return {"job_id": job_id}

@app.post("/v1/web/crawl")
async def web_crawl(req: CrawlRequest, user: dict = Depends(get_current_user)):
    if not req.url.strip().startswith(("http://", "https://")):
        raise HTTPException(400, "url must start with http:// or https://")
    if req.max_pages < 1 or req.max_pages > 500:
        raise HTTPException(400, "max_pages must be between 1 and 500")
    job_id = f"web-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = WEB_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_web_status(job_id, {
        "job_id": job_id, "kind": "crawl",
        "url": req.url.strip(), "max_pages": req.max_pages,
        "state": "queued", "stage": "queued",
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
    })
    _WEB_TASKS[job_id] = asyncio.create_task(_run_web_crawl(job_id, req.url.strip(), req.max_pages))
    return {"job_id": job_id}

@app.get("/v1/web/jobs/{job_id}")
async def web_job_status(job_id: str, user: dict = Depends(get_current_user)):
    _check_web_access(job_id, user)
    data = _read_web_status(job_id)
    if not data:
        raise HTTPException(404)
    return data

@app.get("/v1/web/jobs/{job_id}/download")
async def web_job_download(job_id: str, user: dict = Depends(get_current_user)):
    _check_web_access(job_id, user)
    data = _read_web_status(job_id) or {}
    out_name = data.get("output_name")
    if not out_name:
        raise HTTPException(404, "output not ready")
    path = WEB_JOBS_DIR / job_id / out_name
    if not path.exists():
        raise HTTPException(404, "output not ready")
    media = "application/zip" if path.suffix.lower() == ".zip" else "text/markdown"
    return FileResponse(path, filename=path.name, media_type=media)

@app.post("/v1/web/jobs/{job_id}/cancel")
async def web_job_cancel(job_id: str, user: dict = Depends(get_current_user)):
    _check_web_access(job_id, user)
    task = _WEB_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    data = _read_web_status(job_id) or {}
    data.update({"state": "cancelled", "stage": "cancelled", "updated_at": int(time.time())})
    _write_web_status(job_id, data)
    return {"status": "ok"}

@app.delete("/v1/web/jobs/{job_id}")
async def web_job_delete(job_id: str, user: dict = Depends(get_current_user)):
    _check_web_access(job_id, user)
    task = _WEB_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    job_dir = WEB_JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return {"status": "ok"}

async def _run_web_agent(job_id: str, prompt: str):
    job_dir = WEB_JOBS_DIR / job_id

    async def _u(**fields):
        cur = _read_web_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_web_status(job_id, cur)

    api_key = os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key:
        await _u(state="failed", stage="failed", error="FIRECRAWL_API_KEY not set")
        return
    try:
        await _u(state="running", stage="submitting", error=None)
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post("https://api.firecrawl.dev/v2/agent",
                                  headers=headers, json={"prompt": prompt})
            r.raise_for_status()
            d = r.json()
        if not d.get("success") or not d.get("id"):
            raise RuntimeError(d.get("error") or "agent submission failed")
        agent_id = d["id"]
        await _u(stage="agent_running", agent_id=agent_id)

        poll_url = f"https://api.firecrawl.dev/v2/agent/{agent_id}"
        # Up to ~30 min (180 polls × 10s)
        for _ in range(180):
            await asyncio.sleep(10)
            async with httpx.AsyncClient(timeout=30.0) as client:
                pr = await client.get(poll_url, headers=headers)
            if pr.status_code != 200:
                continue
            res = pr.json()
            status = res.get("status", "")
            if status == "completed":
                data = res.get("data")
                # Render the data as Markdown — could be string, dict, or list.
                if isinstance(data, str):
                    body = data
                else:
                    body = "```json\n" + json.dumps(data, indent=2, ensure_ascii=False) + "\n```"
                out = job_dir / "agent.md"
                out.write_text(
                    f"# Agent result\n\n_Prompt: {prompt}_\n\n_Model: {res.get('model','')} · "
                    f"Credits used: {res.get('creditsUsed', 0)}_\n\n{body}\n",
                    encoding="utf-8",
                )
                await _u(state="completed", stage="completed",
                         output_name=out.name, output_size=out.stat().st_size,
                         credits_used=res.get("creditsUsed", 0),
                         model=res.get("model", ""))
                return
            if status in ("failed", "cancelled"):
                raise RuntimeError(f"agent status: {status} — {res.get('error', '')}")
            # Surface intermediate progress if Firecrawl returns any
            await _u(stage=f"agent_{status}")
        raise RuntimeError("agent timed out after 30 minutes")
    except asyncio.CancelledError:
        await _u(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _u(state="failed", stage="failed", error=str(e))
    finally:
        _WEB_TASKS.pop(job_id, None)

class AgentRequest(BaseModel):
    prompt: str

@app.post("/v1/web/agent")
async def web_agent(req: AgentRequest, user: dict = Depends(get_current_user)):
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is required")
    job_id = f"web-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = WEB_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_web_status(job_id, {
        "job_id": job_id, "kind": "agent",
        "url": "", "prompt": req.prompt.strip()[:500],
        "state": "queued", "stage": "queued",
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
    })
    _WEB_TASKS[job_id] = asyncio.create_task(_run_web_agent(job_id, req.prompt.strip()))
    return {"job_id": job_id}

async def _push_md_files_to_dify(files: list[tuple[str, str]], dataset_name: str) -> dict:
    """Upload a list of (name, markdown_text) pairs into a Dify dataset.
    Creates the dataset if it doesn't exist. Returns {dataset_id, uploaded, errors}."""
    api_url = os.getenv("DIFY_DATASET_API_URL", "")
    api_key = os.getenv("DIFY_DATASET_API_KEY", "")
    if not api_url or not api_key:
        raise HTTPException(400, "DIFY_DATASET_API_URL / DIFY_DATASET_API_KEY not set in env")
    base = api_url.rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    ds_name = dataset_name[:40]
    uploaded = 0
    errors: list[dict] = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Find or create dataset
        dataset_id = None
        list_resp = await client.get(f"{base}/datasets?page=1&limit=100", headers=headers)
        if list_resp.status_code == 200:
            for ds in list_resp.json().get("data", []):
                if ds.get("name") == ds_name:
                    dataset_id = ds["id"]
                    break
        if not dataset_id:
            cr = await client.post(f"{base}/datasets", headers=headers, json={
                "name": ds_name, "permission": "all_team_members",
                "indexing_technique": "high_quality",
            })
            cr.raise_for_status()
            dataset_id = cr.json().get("id")

        for name, text in files:
            if not text.strip():
                continue
            try:
                resp = await client.post(
                    f"{base}/datasets/{dataset_id}/document/create-by-text",
                    headers=headers,
                    json={
                        "name": name[:60],
                        "text": text,
                        "indexing_technique": "high_quality",
                        "process_rule": {"mode": "automatic"},
                    },
                    timeout=60,
                )
                if resp.status_code < 300:
                    uploaded += 1
                else:
                    errors.append({"file": name, "status": resp.status_code, "detail": resp.text[:200]})
            except Exception as e:
                errors.append({"file": name, "error": str(e)[:200]})
    return {"dataset_id": dataset_id, "dataset_name": ds_name, "uploaded": uploaded, "errors": errors}

@app.post("/v1/web/jobs/{job_id}/dify")
async def web_job_push_dify(job_id: str, user: dict = Depends(get_current_user)):
    """Push the scrape/crawl result(s) into a Dify dataset via the Dataset API."""
    _check_web_access(job_id, user)
    data = _read_web_status(job_id) or {}
    if data.get("state") != "completed":
        raise HTTPException(409, "job not completed yet")
    job_dir = WEB_JOBS_DIR / job_id
    files: list[tuple[str, str]] = []

    if data.get("kind") == "crawl":
        pages_dir = job_dir / "pages"
        if not pages_dir.exists():
            raise HTTPException(404, "no pages found")
        for f in sorted(pages_dir.iterdir()):
            if f.is_file() and f.suffix.lower() == ".md":
                files.append((f.stem, f.read_text(encoding="utf-8")))
    else:  # scrape
        out_name = data.get("output_name") or ""
        if not out_name:
            raise HTTPException(404, "no output found")
        p = job_dir / out_name
        if not p.exists():
            raise HTTPException(404, "no output found")
        files.append((p.stem, p.read_text(encoding="utf-8")))

    if not files:
        raise HTTPException(400, "nothing to upload")

    # Dataset name: derived from the URL hostname/path
    base_name = data.get("page_title") or data.get("url") or job_id
    try:
        host = urlparse(data.get("url", "")).netloc.removeprefix("www.")
        ds_name = f"web — {host}" if host else f"web — {base_name}"
    except Exception:
        ds_name = f"web — {base_name}"

    try:
        result = await _push_md_files_to_dify(files, ds_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    # Persist the last push outcome on the job for UI display
    cur = _read_web_status(job_id) or {}
    cur["dify_state"] = "completed" if not result.get("errors") else "partial"
    cur["dify_uploaded"] = result.get("uploaded", 0)
    cur["dify_dataset_name"] = result.get("dataset_name")
    cur["dify_errors_count"] = len(result.get("errors") or [])
    cur["updated_at"] = int(time.time())
    _write_web_status(job_id, cur)
    return result

# --- AUDIOBOOK (Markdown → MP3 via Kokoro-FastAPI) ---
XTTS_BASE_URL = os.getenv("XTTS_BASE_URL", "http://localhost:8021")
XTTS_DEFAULT_VOICE = os.getenv("XTTS_DEFAULT_VOICE", "Claribel Dervla")
XTTS_CONTAINER = os.getenv("XTTS_CONTAINER", "xtts-server")
XTTS_IMAGE = os.getenv("XTTS_IMAGE", "xtts-blackwell:6")
XTTS_MODEL_VOLUME = os.getenv("XTTS_MODEL_VOLUME", "/opt/xtts-models")

# Built-in voices shipped with XTTS-v2 — static list (no network call needed).
XTTS_VOICES = [
    "Aaron Dreschner", "Abrahan Mack", "Adde Michal", "Alexandra Hisakawa",
    "Alison Dietlinde", "Alma María", "Ana Florence", "Andrew Chipper",
    "Annmarie Nele", "Asya Anara", "Badr Odhiambo", "Baldur Sanjin",
    "Barbora MacLean", "Brenda Stern", "Camilla Holmström",
    "Chandra MacFarland", "Claribel Dervla", "Craig Gutsy", "Daisy Studious",
    "Damien Black", "Damjan Chapman", "Dionisio Schuyler", "Eugenio Mataracı",
    "Ferran Simen", "Filip Traverse", "Gilberto Mathias", "Gitta Nikolina",
    "Gracie Wise", "Henriette Usha", "Ige Behringer", "Ilkin Urbano",
    "Kazuhiko Atallah", "Kumar Dahl", "Lidiya Szekeres", "Lilya Stainthorpe",
    "Ludvig Milivoj", "Luis Moray", "Maja Ruoho", "Marcos Rudaski",
    "Narelle Moon", "Nova Hogarth", "Rosemary Okafor", "Royston Min",
    "Sofia Hellen", "Suad Qasim", "Szofi Granger", "Tammie Ema", "Tammy Grit",
    "Tanja Adelina", "Torcull Diarmuid", "Uta Obando", "Viktor Eka",
    "Viktor Menelaos", "Vjollca Johnnie", "Wulf Carlevaro", "Xavier Hayasaka",
    "Zacharie Aimilios", "Zofija Kendrick",
]

# ── XTTS lifecycle (load-on-demand) ──────────────────────────────────────────
_xtts_users = 0
_xtts_lock = asyncio.Lock()

async def _xtts_health() -> dict | None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{XTTS_BASE_URL.rstrip('/')}/health")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None

async def _xtts_start_and_wait(timeout: float = 180.0):
    """Make sure the XTTS container is running and the model is loaded."""
    h = await _xtts_health()
    if h is not None and h.get("loaded"):
        return  # already up and warm

    if h is None:
        # Container is not responding — try to (re)start it
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", "-f", "{{.State.Status}}", XTTS_CONTAINER,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        status = out.decode().strip() if proc.returncode == 0 else ""
        if status in ("exited", "created"):
            await (await asyncio.create_subprocess_exec(
                "docker", "start", XTTS_CONTAINER,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )).wait()
        else:
            await (await asyncio.create_subprocess_exec(
                "docker", "run", "-d", "--name", XTTS_CONTAINER,
                "--gpus", "all",
                "-p", f"127.0.0.1:8021:8021",
                "-v", f"{XTTS_MODEL_VOLUME}:/root/.local/share/tts",
                XTTS_IMAGE,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )).wait()

    # Wait for /health, then force model load via /v1/audio/voices
    deadline = time.time() + timeout
    while time.time() < deadline:
        await asyncio.sleep(2)
        h = await _xtts_health()
        if h is not None:
            break
    else:
        raise RuntimeError(f"XTTS server did not respond within {timeout}s")

    # Force model load (slow first time after container start)
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            await client.get(f"{XTTS_BASE_URL.rstrip('/')}/v1/audio/voices")
    except Exception as e:
        raise RuntimeError(f"XTTS model load failed: {e}")

async def _xtts_acquire():
    """Increment usage; start container if first user."""
    global _xtts_users
    async with _xtts_lock:
        if _xtts_users == 0:
            await _xtts_start_and_wait()
        _xtts_users += 1

async def _xtts_release():
    """Decrement usage; stop container if last user."""
    global _xtts_users
    async with _xtts_lock:
        _xtts_users = max(0, _xtts_users - 1)
        if _xtts_users == 0:
            await (await asyncio.create_subprocess_exec(
                "docker", "stop", XTTS_CONTAINER,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )).wait()

def _audiobook_status_path(job_id: str) -> Path:
    return AUDIOBOOK_JOBS_DIR / job_id / "status.json"

def _audiobook_log_path(job_id: str) -> Path:
    return AUDIOBOOK_JOBS_DIR / job_id / "run.log"

def _read_audiobook_status(job_id: str) -> dict | None:
    p = _audiobook_status_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def _write_audiobook_status(job_id: str, data: dict):
    _audiobook_status_path(job_id).write_text(json.dumps(data))

def _attach_audiobook_progress(job_id: str, data: dict) -> dict:
    p = AUDIOBOOK_JOBS_DIR / job_id / "progress.json"
    if p.exists():
        try:
            data["progress"] = json.loads(p.read_text())
        except Exception:
            pass
    return data

async def _run_audiobook_job(job_id: str, md_path: Path, out_dir: Path, basename: str,
                             voice: str, speed: float, engine: str, summarize: bool,
                             summary_model: str = "", summary_origin: str = "auto",
                             resumed: bool = False):
    state_dir = AUDIOBOOK_JOBS_DIR / job_id / "_state"
    log_path = _audiobook_log_path(job_id)

    async def _u(**fields):
        cur = _read_audiobook_status(job_id) or {}
        cur.update(fields)
        cur["updated_at"] = int(time.time())
        _write_audiobook_status(job_id, cur)

    try:
        cmd = [
            "python3", "-u", str(SCRIPTS_DIR / "build_audiobook.py"),
            str(md_path), str(out_dir),
            "--basename", basename,
            "--voice", voice, "--speed", str(speed),
            "--engine", engine,
            "--state-dir", str(state_dir),
            "--kokoro-host", XTTS_BASE_URL,  # legacy flag name; points to XTTS now
        ]
        if summarize:
            cmd.append("--summarize")
            chosen_summary_model = (summary_model or "").strip() or AUDIOBOOK_DEFAULT_LLM
            cmd += ["--summary-model", chosen_summary_model]
            cmd += ["--summary-origin", summary_origin or "auto"]
        await _u(state="running", stage="synthesizing", error=None)
        # XTTS is load-on-demand: bring it up before the script needs it.
        # (Started here, before the subprocess runs; released in finally.)
        if engine == "xtts":
            await _xtts_acquire()
        # Persist per-section adapter cache next to the job so Resume keeps progress.
        adapt_cache = AUDIOBOOK_JOBS_DIR / job_id / "adapt_cache"
        adapt_cache.mkdir(parents=True, exist_ok=True)
        progress_path = AUDIOBOOK_JOBS_DIR / job_id / "progress.json"
        sub_env = {
            **os.environ,
            "ADAPT_CACHE_DIR": str(adapt_cache),
            "ADAPT_PROGRESS_PATH": str(progress_path),
        }
        with log_path.open("a" if resumed else "w") as logf:
            if resumed:
                logf.write(f"\n--- RESUME @ {int(time.time())} ---\n")
            logf.write(f"$ {' '.join(cmd)}\n")
            logf.flush()
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=logf, stderr=asyncio.subprocess.STDOUT, env=sub_env,
            )
            rc = await proc.wait()
        if rc != 0:
            tail = log_path.read_text()[-2000:] if log_path.exists() else ""
            await _u(state="failed", stage="failed", error=f"exit={rc}", log_tail=tail)
            return
        m4b_path = out_dir / f"{basename}.m4b"
        zip_path = out_dir / f"{basename}.zip"
        if not m4b_path.exists():
            await _u(state="failed", stage="failed", error="m4b not produced")
            return
        # Compute total duration from M4B
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(m4b_path)],
            check=True, capture_output=True, text=True,
        )
        duration = float(r.stdout.strip() or 0)
        # Count chapters
        chapters_dir = out_dir / "chapters"
        n_chapters = sum(1 for _ in chapters_dir.glob("*.mp3")) if chapters_dir.exists() else 0
        await _u(state="completed", stage="completed",
                 m4b_name=m4b_path.name, m4b_size=m4b_path.stat().st_size,
                 zip_name=zip_path.name if zip_path.exists() else None,
                 zip_size=zip_path.stat().st_size if zip_path.exists() else None,
                 chapters_count=n_chapters,
                 duration_seconds=round(duration, 1))
    except asyncio.CancelledError:
        await _u(state="cancelled", stage="cancelled")
        raise
    except Exception as e:
        await _u(state="failed", stage="failed", error=str(e))
    finally:
        _AUDIOBOOK_TASKS.pop(job_id, None)
        if engine == "xtts":
            try:
                await _xtts_release()
            except Exception as e:
                print(f"[xtts] release error: {e}")

AUDIOBOOK_DEFAULT_LLM = "qwen3.6:35b"

@app.get("/v1/audiobook/llms")
async def audiobook_llms(user: dict = Depends(get_current_user)):
    """Return a unified list of LLMs available for audio adaptation:
    everything from the local Ollama daemon + everything from the Ollama Cloud
    catalog. Each entry is tagged with its origin (local / cloud)."""
    seen: dict[str, dict] = {}

    errors: list[str] = []

    # Local models (localhost) — includes any :cloud suffix-tagged remote ones
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get("http://localhost:11434/api/tags")
            r.raise_for_status()
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if not name:
                    continue
                origin = "cloud" if name.endswith(":cloud") or "-cloud" in name else "local"
                seen[name] = {"name": name, "origin": origin}
    except Exception as e:
        errors.append(f"local ollama: {e}")
        print(f"[llms] local ollama unreachable: {e}")

    # Ollama Cloud catalog
    api_key = os.getenv("OLLAMA_API_KEY", "")
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get("https://ollama.com/api/tags",
                                     headers={"Authorization": f"Bearer {api_key}"})
                r.raise_for_status()
                for m in r.json().get("models", []):
                    name = m.get("name") if isinstance(m, dict) else m
                    if not name or name in seen:
                        continue
                    seen[name] = {"name": name, "origin": "cloud"}
        except Exception as e:
            errors.append(f"ollama cloud: {e}")
            print(f"[llms] ollama cloud unreachable: {e}")
    else:
        errors.append("OLLAMA_API_KEY not set — Ollama Cloud catalog skipped")

    # DeepSeek API catalog (OpenAI-compatible /v1/models)
    ds_key = os.getenv("DEEPSEEK_API_KEY", "")
    ds_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    if ds_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{ds_base}/models",
                                     headers={"Authorization": f"Bearer {ds_key}"})
                r.raise_for_status()
                for m in r.json().get("data", []):
                    name = m.get("id") if isinstance(m, dict) else m
                    if not name:
                        continue
                    seen[name] = {"name": name, "origin": "deepseek"}
        except Exception as e:
            errors.append(f"deepseek: {e}")
            print(f"[llms] deepseek unreachable: {e}")
    else:
        errors.append("DEEPSEEK_API_KEY not set — DeepSeek catalog skipped")

    # Always expose the configured default, even if not yet pulled
    if AUDIOBOOK_DEFAULT_LLM not in seen:
        seen[AUDIOBOOK_DEFAULT_LLM] = {"name": AUDIOBOOK_DEFAULT_LLM, "origin": "local"}

    origin_order = {"local": 0, "deepseek": 1, "cloud": 2}
    items = sorted(seen.values(), key=lambda x: (origin_order.get(x["origin"], 9), x["name"]))
    return {"models": items, "default": AUDIOBOOK_DEFAULT_LLM, "errors": errors}

@app.get("/v1/audiobook/voices")
async def audiobook_voices(user: dict = Depends(get_current_user)):
    """Return XTTS-v2 built-in voice list (static, no network call)."""
    return {"voices": XTTS_VOICES, "default": XTTS_DEFAULT_VOICE}

@app.get("/v1/audiobook/jobs")
async def audiobook_list_jobs(limit: int = 30, user: dict = Depends(get_current_user)):
    jobs = []
    is_admin = _is_admin(user)
    if AUDIOBOOK_JOBS_DIR.exists():
        dirs = sorted([d for d in AUDIOBOOK_JOBS_DIR.iterdir() if d.is_dir()],
                      key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs:
            data = _read_audiobook_status(d.name)
            if not data: continue
            if not is_admin and data.get("owner_id") != user["id"]:
                continue
            if data.get("state") == "running" and d.name not in _AUDIOBOOK_TASKS:
                data["state"] = "interrupted"
            _attach_audiobook_progress(d.name, data)
            jobs.append(data)
            if len(jobs) >= limit:
                break
    return {"data": jobs}

def _safe_basename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\- ]", "_", s).strip()[:60].replace(" ", "_") or "audiobook"

class AudiobookFromRunRequest(BaseModel):
    run_id: str
    voice: str = ""
    speed: float = 1.0
    engine: str = "xtts"
    summarize: bool = False
    summary_model: str = ""
    summary_origin: str = "auto"

@app.post("/v1/audiobook/from-run")
async def audiobook_from_run(req: AudiobookFromRunRequest, user: dict = Depends(get_current_user)):
    _check_run_access(req.run_id, user)
    run_dir = DATA_DIR / req.run_id
    md_src = run_dir / "report.md"
    if not md_src.exists():
        raise HTTPException(404, "report.md not found for this run")
    voice = (req.voice or "").strip() or XTTS_DEFAULT_VOICE
    speed = max(0.5, min(2.0, float(req.speed)))
    engine = req.engine if req.engine in ("xtts", "openai") else "xtts"
    if engine == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not set in env")

    job_id = f"book-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = AUDIOBOOK_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    md_path = job_dir / "input.md"
    md_path.write_bytes(md_src.read_bytes())

    rd = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    title = (rd.get("question") or req.run_id)[:60]
    basename = _safe_basename(title)

    chosen_summary_model = (req.summary_model or "").strip() or AUDIOBOOK_DEFAULT_LLM
    chosen_summary_origin = (req.summary_origin or "auto").strip() or "auto"
    _write_audiobook_status(job_id, {
        "job_id": job_id, "source_kind": "run", "source_run_id": req.run_id,
        "title": title, "basename": basename,
        "voice": voice, "speed": speed, "engine": engine, "summarize": bool(req.summarize),
        "summary_model": chosen_summary_model,
        "summary_origin": chosen_summary_origin,
        "state": "queued", "stage": "queued",
        "input_size": md_path.stat().st_size,
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
    })
    _AUDIOBOOK_TASKS[job_id] = asyncio.create_task(
        _run_audiobook_job(job_id, md_path, job_dir, basename, voice, speed, engine,
                           bool(req.summarize), summary_model=chosen_summary_model,
                           summary_origin=chosen_summary_origin)
    )
    return {"job_id": job_id}

@app.post("/v1/audiobook/from-upload")
async def audiobook_from_upload(
    file: UploadFile = File(...),
    voice: str = Form(""),
    speed: float = Form(1.0),
    engine: str = Form("xtts"),
    summarize: bool = Form(False),
    summary_model: str = Form(""),
    summary_origin: str = Form("auto"),
    user: dict = Depends(get_current_user),
):
    safe_name = Path(file.filename or "").name
    ext = Path(safe_name).suffix.lower()
    if ext != ".md":
        raise HTTPException(400, "only .md files are accepted (convert your PDF first via Tools → PDF)")
    voice = (voice or "").strip() or XTTS_DEFAULT_VOICE
    speed = max(0.5, min(2.0, float(speed)))
    if engine not in ("xtts", "openai"):
        engine = "xtts"
    if engine == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "OPENAI_API_KEY not set in env")

    job_id = f"book-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job_dir = AUDIOBOOK_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    md_path = job_dir / "input.md"
    with md_path.open("wb") as f:
        while data := await file.read(1024 * 1024):
            f.write(data)
    basename = _safe_basename(Path(safe_name).stem)
    chosen_summary_model = (summary_model or "").strip() or AUDIOBOOK_DEFAULT_LLM
    chosen_summary_origin = (summary_origin or "auto").strip() or "auto"

    _write_audiobook_status(job_id, {
        "job_id": job_id, "source_kind": "upload",
        "title": safe_name, "basename": basename, "filename": safe_name,
        "voice": voice, "speed": speed, "engine": engine, "summarize": bool(summarize),
        "summary_model": chosen_summary_model,
        "summary_origin": chosen_summary_origin,
        "state": "queued", "stage": "queued",
        "input_size": md_path.stat().st_size,
        "owner_id": user["id"], "owner_name": user["username"],
        "created_at": int(time.time()), "updated_at": int(time.time()),
    })
    _AUDIOBOOK_TASKS[job_id] = asyncio.create_task(
        _run_audiobook_job(job_id, md_path, job_dir, basename, voice, speed, engine,
                           bool(summarize), summary_model=chosen_summary_model,
                           summary_origin=chosen_summary_origin)
    )
    return {"job_id": job_id}

class VoicePreviewRequest(BaseModel):
    voice: str
    speed: float = 1.0
    text: str = "Bonjour, ceci est un échantillon de ma voix. Hello, this is a voice sample."

@app.post("/v1/audiobook/preview")
async def audiobook_preview(req: VoicePreviewRequest, user: dict = Depends(get_current_user)):
    """Generate a 1-sentence MP3 preview for the chosen voice via XTTS-v2.
    Auto-starts the XTTS container if not running (~20-30s cold start)."""
    voice = (req.voice or "").strip() or XTTS_DEFAULT_VOICE
    speed = max(0.5, min(2.0, float(req.speed)))
    await _xtts_acquire()
    try:
        payload = {"voice": voice, "input": req.text[:300],
                   "response_format": "mp3", "speed": speed, "language": "fr"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(f"{XTTS_BASE_URL.rstrip('/')}/v1/audio/speech",
                                  json=payload)
            r.raise_for_status()
            mp3 = r.content
    finally:
        await _xtts_release()
    from fastapi.responses import Response
    return Response(content=mp3, media_type="audio/mpeg")

@app.get("/v1/audiobook/jobs/{job_id}")
async def audiobook_job_status(job_id: str, user: dict = Depends(get_current_user)):
    _check_audiobook_access(job_id, user)
    data = _read_audiobook_status(job_id)
    if not data:
        raise HTTPException(404)
    _attach_audiobook_progress(job_id, data)
    log_path = _audiobook_log_path(job_id)
    if log_path.exists():
        try:
            data["log_tail"] = log_path.read_text()[-4000:]
        except Exception:
            data["log_tail"] = ""
    return data

@app.post("/v1/audiobook/jobs/{job_id}/resume")
async def audiobook_job_resume(job_id: str, user: dict = Depends(get_current_user)):
    _check_audiobook_access(job_id, user)
    if job_id in _AUDIOBOOK_TASKS and not _AUDIOBOOK_TASKS[job_id].done():
        raise HTTPException(409, "job already running")
    data = _read_audiobook_status(job_id) or {}
    job_dir = AUDIOBOOK_JOBS_DIR / job_id
    md_path = job_dir / "input.md"
    if not md_path.exists():
        raise HTTPException(404, "input markdown missing")
    _AUDIOBOOK_TASKS[job_id] = asyncio.create_task(
        _run_audiobook_job(
            job_id, md_path, job_dir,
            data.get("basename", "audiobook"),
            data.get("voice", XTTS_DEFAULT_VOICE),
            float(data.get("speed", 1.0)),
            data.get("engine", "xtts"),
            bool(data.get("summarize", False)),
            summary_model=data.get("summary_model", ""),
            summary_origin=data.get("summary_origin", "auto"),
            resumed=True,
        )
    )
    return {"status": "ok"}

@app.get("/v1/audiobook/jobs/{job_id}/download")
async def audiobook_job_download(job_id: str, type: str = "m4b",
                                 user: dict = Depends(get_current_user)):
    _check_audiobook_access(job_id, user)
    data = _read_audiobook_status(job_id) or {}
    job_dir = AUDIOBOOK_JOBS_DIR / job_id
    if type == "zip":
        name = data.get("zip_name")
        if not name: raise HTTPException(404, "zip not ready")
        p = job_dir / name
        media = "application/zip"
    elif type == "summary":
        p = job_dir / "summary.md"
        if not p.exists():
            raise HTTPException(404, "summary not available")
        media = "text/markdown"
    else:  # m4b
        name = data.get("m4b_name")
        if not name: raise HTTPException(404, "m4b not ready")
        p = job_dir / name
        media = "audio/mp4"
    if not p.exists():
        raise HTTPException(404, "output not ready")
    return FileResponse(p, filename=p.name, media_type=media)

@app.post("/v1/audiobook/jobs/{job_id}/cancel")
async def audiobook_job_cancel(job_id: str, user: dict = Depends(get_current_user)):
    _check_audiobook_access(job_id, user)
    task = _AUDIOBOOK_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    data = _read_audiobook_status(job_id) or {}
    data.update({"state": "cancelled", "stage": "cancelled", "updated_at": int(time.time())})
    _write_audiobook_status(job_id, data)
    return {"status": "ok"}

@app.delete("/v1/audiobook/jobs/{job_id}")
async def audiobook_job_delete(job_id: str, user: dict = Depends(get_current_user)):
    _check_audiobook_access(job_id, user)
    task = _AUDIOBOOK_TASKS.pop(job_id, None)
    if task and not task.done():
        task.cancel()
    job_dir = AUDIOBOOK_JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    return {"status": "ok"}

# --- SYSTEM API ---
@app.get("/v1/servers")
async def list_servers(user: dict = Depends(get_current_user)): return _get_servers()

@app.delete("/v1/dossier/runs/{run_id}")
async def delete_run(run_id: str, user: dict = Depends(get_current_user)):
    _check_run_access(run_id, user)
    task = _DOSSIER_TASKS.pop(run_id, None)
    if task and not task.done(): task.cancel()
    run_dir = DATA_DIR / run_id
    if run_dir.exists(): shutil.rmtree(run_dir)
    return {"status": "ok"}

@app.post("/v1/servers")
async def add_server(server: dict, user: dict = Depends(get_current_user)):
    s = _get_servers(); s.append(server); SERVERS_FILE.write_text(json.dumps(s)); return {"status": "ok"}

@app.get("/ollama/models")
async def list_models(url: str = None, user: dict = Depends(get_current_user)):
    target = url or "http://127.0.0.1:11434"
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{target.rstrip('/')}/api/tags")
            return r.json()
    except: return {"models": []}

@app.get("/system/metrics")
async def metrics(user: dict = Depends(get_current_user)):
    return {"cpu_percent": psutil.cpu_percent(), "ram_percent": psutil.virtual_memory().percent, "gpus": _get_gpu_usage()}

# --- UI CATCH-ALL (MUST BE LAST) ---
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    if full_path.startswith(("v1/", "ollama/", "system/", "ui/assets/", "assets/")):
        raise HTTPException(status_code=404)
    potential_file = Path("../web-ui/dist") / full_path
    if full_path and potential_file.exists() and potential_file.is_file():
        return FileResponse(potential_file)
    return FileResponse("../web-ui/dist/index.html")
