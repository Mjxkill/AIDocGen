import os
import time
import asyncio
import uuid
import psutil
import json
import shutil
import subprocess
import httpx
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.config import DossierConfig
from core.engine import DossierEngine

app = FastAPI(title="Ollama Ensemble Proxy")

# CORS
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Base paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data" / "dossiers"

# State
_DOSSIER_TASKS: dict[str, asyncio.Task] = {}
SERVERS_FILE = DATA_DIR.parent / "servers.json"

def _get_servers():
    if not SERVERS_FILE.exists():
        initial = [{"name": "Local Ollama", "url": os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")}]
        SERVERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SERVERS_FILE.write_text(json.dumps(initial))
    return json.loads(SERVERS_FILE.read_text())

def _save_servers(servers):
    SERVERS_FILE.write_text(json.dumps(servers))

def _get_engine(ollama_url: str = None, models: dict = None):
    config = DossierConfig.from_env()
    config.data_dir = str(DATA_DIR)
    if ollama_url: config.ollama_base_url = ollama_url
    if models:
        if models.get("planner"): config.planner_model = models["planner"]
        if models.get("writer"): config.writer_model = models["writer"]
        if models.get("judge"): config.judge_model = models["judge"]
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

# --- API ---

class RunRequest(BaseModel):
    question: str
    prompt_type: str = "generic"
    detail_level: str = "medium"
    ollama_url: str | None = None
    planner_model: str | None = None
    writer_model: str | None = None
    judge_model: str | None = None

@app.post("/v1/dossier/runs")
async def start_run(req: RunRequest):
    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    
    # Custom config for this run
    custom_models = {
        "planner": req.planner_model,
        "writer": req.writer_model,
        "judge": req.judge_model
    }
    engine = _get_engine(req.ollama_url, custom_models)
    
    run_dir = DATA_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "run_id": run_id, "question": req.question, "state": "running", "stage": "init",
        "prompt_type": req.prompt_type, "detail_level": req.detail_level,
        "ollama_url": req.ollama_url or os.getenv("OLLAMA_BASE_URL"),
        "planner_model": req.planner_model or os.getenv("ENSEMBLE_DOSSIER_PLANNER_MODEL"),
        "events": [], "updated_at": int(time.time())
    }
    (run_dir / "status.json").write_text(json.dumps(status))
    task = asyncio.create_task(engine.run(run_id, req.question, req.prompt_type, req.detail_level))
    _DOSSIER_TASKS[run_id] = task
    return {"run_id": run_id}

class ResumeRequest(BaseModel):
    background: bool = True

@app.post("/v1/dossier/runs/{run_id}/resume")
async def resume_run(run_id: str, body: ResumeRequest = None):
    if body is None: body = ResumeRequest()
    engine = _get_engine()
    
    # Load original params from status.json if available
    status_path = DATA_DIR / run_id / "status.json"
    question = ""
    p_type = "generic"
    d_level = "medium"
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
            question = data.get("question", "")
            p_type = data.get("prompt_type", "generic")
            d_level = data.get("detail_level", "medium")
        except: pass
        
    task = asyncio.create_task(engine.run(run_id, question, p_type, d_level, resume=True))
    _DOSSIER_TASKS[run_id] = task
    return {"status": "resumed"}

@app.post("/v1/dossier/runs/{run_id}/approve")
async def approve_run(run_id: str):
    run_dir = DATA_DIR / run_id
    (run_dir / "validated.txt").write_text("ok")
    return await resume_run(run_id)

@app.post("/v1/dossier/runs/{run_id}/cancel")
async def cancel_run(run_id: str):
    task = _DOSSIER_TASKS.pop(run_id, None)
    if task and not task.done():
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
        # Update disk status to interrupted
        status_path = DATA_DIR / run_id / "status.json"
        if status_path.exists():
            try:
                data = json.loads(status_path.read_text())
                data["state"] = "interrupted"
                status_path.write_text(json.dumps(data))
            except: pass
        return {"status": "cancelled"}
    return {"status": "not_running"}

@app.delete("/v1/dossier/runs/{run_id}")
async def delete_run(run_id: str):
    await cancel_run(run_id)
    run_dir = DATA_DIR / run_id
    if run_dir.exists():
        try: shutil.rmtree(run_dir)
        except: pass
    return {"status": "deleted"}

@app.get("/v1/dossier/runs")
async def list_runs(limit: int = 10):
    runs = []
    if DATA_DIR.exists():
        dirs = [d for d in DATA_DIR.iterdir() if d.is_dir()]
        for d in sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
            status_path = d / "status.json"
            if status_path.exists():
                try:
                    data = json.loads(status_path.read_text())
                    # SYNC LOGIC: If marked running but task is gone
                    if data.get("state") == "running" and data.get("run_id") not in _DOSSIER_TASKS:
                        data["state"] = "interrupted"
                    runs.append(data)
                except: pass
    return {"data": runs}

@app.get("/v1/dossier/runs/{run_id}/planner")
async def get_planner(run_id: str):
    path = DATA_DIR / run_id / "planner.json"
    if not path.exists(): raise HTTPException(404)
    return json.loads(path.read_text())

@app.get("/v1/dossier/runs/{run_id}/corpus")
async def get_run_corpus(run_id: str):
    path = DATA_DIR / run_id / "corpus.json"
    if not path.exists(): return {"sources": [], "count": 0}
    return json.loads(path.read_text())

@app.get("/v1/dossier/runs/{run_id}/sections")
async def get_run_sections(run_id: str):
    path = DATA_DIR / run_id / "sections.json"
    if not path.exists(): return {"sections": []}
    return json.loads(path.read_text())

@app.post("/v1/dossier/runs/{run_id}/planner")
async def update_planner(run_id: str, payload: dict):
    path = DATA_DIR / run_id / "planner.json"
    path.write_text(json.dumps(payload, indent=2))
    return {"status": "ok"}

@app.get("/v1/dossier/runs/{run_id}/report/download")
async def download_report(run_id: str):
    path = DATA_DIR / run_id / "report.md"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.md")
    raise HTTPException(404)

@app.get("/v1/dossier/runs/{run_id}/report/latex")
async def download_latex(run_id: str):
    path = DATA_DIR / run_id / "report.tex"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.tex")
    raise HTTPException(404)

@app.get("/v1/dossier/runs/{run_id}/report/pdf")
async def download_pdf(run_id: str):
    path = DATA_DIR / run_id / "report.pdf"
    if path.exists(): return FileResponse(path, filename=f"rapport_{run_id}.pdf")
    raise HTTPException(404)

@app.get("/v1/dossier/runs/{run_id}/annexes/download")
async def download_annexes(run_id: str):
    path = DATA_DIR / run_id / "annexes.md"
    if path.exists(): return FileResponse(path, filename=f"annexes_{run_id}.md")
    raise HTTPException(404)

@app.post("/v1/dossier/runs/{run_id}/restart")
async def restart_run(run_id: str):
    # 1. Get old data
    old_dir = DATA_DIR / run_id
    if not old_dir.exists(): raise HTTPException(404)
    
    status_old = json.loads((old_dir / "status.json").read_text())
    planner_path = old_dir / "planner.json"
    if not planner_path.exists(): raise HTTPException(400, "No planner found to restart with")

    # 2. Create new run
    new_run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    new_dir = DATA_DIR / new_run_id
    new_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Copy planner and prepare state
    shutil.copy(planner_path, new_dir / "planner.json")
    (new_dir / "validated.txt").write_text("ok") # Auto-approve the copied plan
    
    status_new = {
        "run_id": new_run_id,
        "question": status_old.get("question", ""),
        "prompt_type": status_old.get("prompt_type", "generic"),
        "detail_level": status_old.get("detail_level", "medium"),
        "state": "running",
        "stage": "init",
        "events": [{"timestamp": int(time.time()), "stage": "init", "message": f"Restarted from {run_id}"}],
        "updated_at": int(time.time())
    }
    (new_dir / "status.json").write_text(json.dumps(status_new))
    
    # 4. Start engine
    engine = _get_engine()
    task = asyncio.create_task(engine.run(
        new_run_id, 
        status_new["question"], 
        status_new["prompt_type"], 
        status_new["detail_level"], 
        resume=True
    ))
    _DOSSIER_TASKS[new_run_id] = task
    
    return {"new_run_id": new_run_id}

@app.get("/v1/dossier/prompts")
async def list_prompts():
    p_dir = BASE_DIR / "prompts"
    prompts = [f.stem.replace("planner_", "") for f in p_dir.glob("planner_*.txt")] if p_dir.exists() else []
    return {"prompts": prompts or ["generic"]}

@app.get("/config")
async def get_config():
    c = DossierConfig.from_env()
    return {
        "ollama_url": c.ollama_base_url, 
        "planner_model": c.planner_model, 
        "writer_model": c.writer_model, 
        "judge_model": c.judge_model,
        "search_engine": c.web_search_engine or "auto (DDG + SearxNG Fallback)"
    }

@app.get("/v1/servers")
async def list_servers():
    return _get_servers()

@app.post("/v1/servers")
async def add_server(server: dict):
    servers = _get_servers()
    servers.append(server)
    _save_servers(servers)
    return {"status": "ok"}

@app.delete("/v1/servers/{index}")
async def delete_server(index: int):
    servers = _get_servers()
    if 0 <= index < len(servers):
        servers.pop(index)
        _save_servers(servers)
    return {"status": "ok"}

@app.get("/ollama/models")
async def list_models(url: str = None):
    target_url = url or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{target_url.rstrip('/')}/api/tags")
            return resp.json()
    except: return {"models": []}

@app.post("/ollama/pull")
async def pull_model(payload: dict):
    target_url = payload.get("url") or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    async def _do_pull():
        async with httpx.AsyncClient(timeout=None) as client:
            await client.post(f"{target_url.rstrip('/')}/api/pull", json={"name": payload["name"]})
    asyncio.create_task(_do_pull())
    return {"status": "started"}

@app.get("/system/metrics")
async def metrics():
    return {"cpu_percent": psutil.cpu_percent(), "ram_percent": psutil.virtual_memory().percent, "gpus": _get_gpu_usage()}

app.mount("/ui", StaticFiles(directory="../web-ui/dist", html=True), name="ui")
app.mount("/", StaticFiles(directory="../web-ui/dist", html=True), name="root")
