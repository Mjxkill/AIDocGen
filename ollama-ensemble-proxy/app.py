import os
import re
import time
import asyncio
import uuid
import psutil
import json
import shutil
import subprocess
import httpx
import mimetypes
from pathlib import Path
from typing import Any, Optional
from datetime import timedelta

from fastapi import FastAPI, HTTPException, Depends, status, Request
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

app = FastAPI(title="AIDocGen Proxy")

# --- ASSETS PRIORITY (For MIME stability) ---
app.mount("/ui/assets", StaticFiles(directory="../web-ui/dist/assets"), name="assets")
app.mount("/assets", StaticFiles(directory="../web-ui/dist/assets"), name="assets_legacy")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "dossiers"
_DOSSIER_TASKS: dict[str, asyncio.Task] = {}
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

# --- AUTH API ---
@app.post("/v1/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    users = _get_users()
    user = next((u for u in users if u["username"] == form_data.username), None)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Bad credentials")
    return {"access_token": create_access_token(data={"sub": user["username"]}), "token_type": "bearer", "role": user.get("role", "user")}

@app.get("/v1/auth/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"], "role": current_user["role"], "id": current_user["id"]}

# --- DOSSIER API ---
@app.get("/v1/dossier/prompts")
async def list_prompts(user: dict = Depends(get_current_user)):
    p_dir = BASE_DIR / "prompts"
    return {"prompts": [f.stem.replace("planner_", "") for f in p_dir.glob("planner_*.txt")]}

@app.get("/v1/dossier/runs")
async def list_runs(limit: int = 20, user: dict = Depends(get_current_user)):
    runs = []
    if DATA_DIR.exists():
        dirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
        for d in dirs[:limit]:
            s_path = d / "status.json"
            if s_path.exists():
                try:
                    data = json.loads(s_path.read_text())
                    if data.get("state") == "running" and data.get("run_id") not in _DOSSIER_TASKS: data["state"] = "interrupted"
                    runs.append(data)
                except: pass
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
        "coder_model": req.coder_model, "owner_id": user["id"], "owner_name": user["username"],
        "events": [{"timestamp": int(time.time()), "stage": "init", "message": "Démarrage..."}], 
        "updated_at": int(time.time())
    }
    (run_dir / "status.json").write_text(json.dumps(status_data))
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, req.question, req.prompt_type, req.detail_level, language=req.language, coder_model=req.coder_model, tags=tags))
    return {"run_id": run_id}

@app.get("/v1/dossier/runs/{run_id}/planner")
async def get_planner(run_id: str, user: dict = Depends(get_current_user)):
    path = DATA_DIR / run_id / "planner.json"
    if not path.exists(): raise HTTPException(404)
    return json.loads(path.read_text())

@app.post("/v1/dossier/runs/{run_id}/planner")
async def update_planner(run_id: str, payload: dict, user: dict = Depends(get_current_user)):
    path = DATA_DIR / run_id / "planner.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"status": "ok"}

@app.get("/v1/dossier/runs/{run_id}/planner/debug")
async def get_planner_debug(run_id: str, user: dict = Depends(get_current_user)):
    path = DATA_DIR / run_id / "planner_debug.json"
    if not path.exists(): return {"attempts": [], "planner_prompt": {"system": "", "user": ""}, "planner_response_raw": "N/A"}
    return json.loads(path.read_text())

@app.post("/v1/dossier/runs/{run_id}/reset")
async def reset_run(run_id: str, user: dict = Depends(get_current_user)):
    run_dir = DATA_DIR / run_id
    if run_id in _DOSSIER_TASKS: _DOSSIER_TASKS[run_id].cancel()
    for f in ["presearch.json", "planner.json", "report.md", "report.pdf", "planner_debug.json"]:
        if (run_dir / f).exists(): (run_dir / f).unlink()
    d = json.loads((run_dir / "status.json").read_text())
    d.update({"state": "running", "stage": "init", "events": [], "error": None})
    (run_dir / "status.json").write_text(json.dumps(d))
    engine = _get_engine(d.get("ollama_url"), d)
    tags = d.get("tags", [])
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, d["question"], resume=False, language=d.get("language", "fr"), tags=tags))
    return {"status": "ok"}

@app.post("/v1/dossier/runs/{run_id}/cancel")
async def cancel_run(run_id: str, user: dict = Depends(get_current_user)):
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
    run_dir = DATA_DIR / run_id
    if not run_dir.exists(): raise HTTPException(404)
    status_path = run_dir / "status.json"
    d = json.loads(status_path.read_text())
    d.update({"state": "running", "error": None})
    status_path.write_text(json.dumps(d))
    engine = _get_engine(d.get("ollama_url"), d)
    tags = d.get("tags", [])
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, d["question"], resume=True, language=d.get("language", "fr"), tags=tags))
    return {"status": "ok"}

@app.post("/v1/dossier/runs/{run_id}/approve")
async def approve_run(run_id: str, user: dict = Depends(get_current_user)):
    run_dir = DATA_DIR / run_id
    (run_dir / "validated.txt").write_text("ok")
    d = json.loads((run_dir / "status.json").read_text())
    engine = _get_engine(d.get("ollama_url"))
    tags = d.get("tags", [])
    _DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(run_id, d["question"], resume=True, language=d.get("language", "fr"), tags=tags))
    return {"status": "ok"}

@app.get("/v1/dossier/runs/{run_id}/report/pdf")
async def download_pdf(run_id: str, user: dict = Depends(get_current_user)):
    path = DATA_DIR / run_id / "report.pdf"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.pdf")
    raise HTTPException(404)

@app.get("/v1/dossier/runs/{run_id}/report/download")
async def download_md(run_id: str, user: dict = Depends(get_current_user)):
    path = DATA_DIR / run_id / "report.md"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.md")
    raise HTTPException(404)

# --- SYSTEM API ---
@app.get("/v1/servers")
async def list_servers(user: dict = Depends(get_current_user)): return _get_servers()

@app.delete("/v1/dossier/runs/{run_id}")
async def delete_run(run_id: str, user: dict = Depends(get_current_user)):
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
