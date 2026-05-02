# AGENTS.md - AIDocGen Codebase Guide

This document provides essential information for AI coding agents working in this repository.

## Build, Lint, Test Commands

### Frontend (web-ui/)
```bash
cd web-ui
npm install          # Install dependencies
npm run dev          # Development server
npm run build        # Production build (tsc -b && vite build)
npm run lint         # Run ESLint
npm run preview      # Preview production build
```

### Backend (ollama-ensemble-proxy/)
```bash
cd ollama-ensemble-proxy
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8001 --env-file ensemble-proxy.env
```

### Running Single Tests
No test framework is configured. Python test files (`test_*.py`) in root directory are standalone scripts:
```bash
cd ollama-ensemble-proxy && source .venv/bin/activate
python ../test_robustness.py    # Run standalone test
python ../test_cloud_direct.py  # Run cloud connectivity test
```

## Project Architecture

**Monorepo structure:**
- `web-ui/` - React + Vite + TypeScript frontend
- `ollama-ensemble-proxy/` - FastAPI backend with LLM orchestration

**Backend entry point:** `ollama-ensemble-proxy/app.py`
**Frontend entry point:** `web-ui/src/main.tsx` → `App.tsx`

## Code Style Guidelines

### TypeScript/React (Frontend)

**Imports:**
```typescript
import { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
```
- Group imports: external libraries first, then local files
- Use named imports from React

**Types:**
```typescript
interface Metrics { cpu_percent: number; ram_percent: number; gpus: any[]; }
interface RunStatus { run_id: string; question: string; state: string; ... }
```
- Define interfaces at the top of file (line ~5-12)
- Use `interface` over `type` for object shapes
- Use `any` sparingly; prefer explicit types

**Components:**
- Functional components only using arrow functions
- Component names: PascalCase (e.g., `RunDetailPanel`, `VisualPlanEditor`)
- Props defined inline: `{ onLogin }: { onLogin: () => void }`

**Naming conventions:**
- Variables: camelCase (`runId`, `planner`, `setPlanner`)
- State setters: `const [state, setState] = useState(...)`
- Files: PascalCase.tsx for components, camelCase.ts for utilities

**Event handling:**
```typescript
const handle = async (e: any) => { ... }  // Use 'any' for event types
onChange={e => setQ(e.target.value)}
onClick={() => setView('dashboard')}
```

**Async operations:**
```typescript
axios.get('/v1/servers').then(r => setSrv(r.data)).catch(()=>{});
```
- Use `.then/.catch` pattern (no async/await in event handlers)
- Empty catch blocks for non-critical errors

**Styling:**
- CSS classes defined in `App.css`
- Inline styles for dynamic values: `style={{width:'100%', marginBottom:'10px'}}`
- Use className composition: `className={\`nav-item ${view==='dashboard'?'active':''}\`}`

### Python (Backend)

**Imports:**
```python
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from core.config import DossierConfig
from core.engine import DossierEngine
```
- Standard library first (alphabetically)
- Third-party libraries second
- Local imports last with explicit module paths

**Function definitions:**
```python
def _get_servers():
async def list_runs(limit: int = 20, user: dict = Depends(get_current_user)):
```
- Private helpers prefix with `_`
- Type hints for all function parameters and return values
- Use async for FastAPI route handlers

**Models (Pydantic):**
```python
class RunRequest(BaseModel):
    model_config = ConfigDict(extra='allow')
    question: str
    prompt_type: Optional[str] = "generic"
```
- Use `model_config = ConfigDict(extra='allow')` for flexible models
- `Optional[T]` for nullable fields with defaults

**Error handling:**
```python
try:
    # operation
except: return {"models": []}  # Graceful fallback
raise HTTPException(status_code=404)  # Explicit error responses
```

**Naming conventions:**
- Variables: snake_case (`run_id`, `s_path`, `llm_logs`)
- Classes: PascalCase (`DossierEngine`, `LLMClient`)
- Constants: UPPER_SNAKE for true constants
- Private functions: prefix with underscore `_get_engine`

**Dataclass pattern:**
```python
@dataclass
class DossierConfig:
    ollama_base_url: str
    data_dir: str
    # ...

    @classmethod
    def from_env(cls) -> "DossierConfig":
        return cls(...)
```

**File structure:**
- Route handlers in `app.py`
- Core logic in `core/` directory (`engine.py`, `llm.py`, `config.py`, etc.)
- Data stored in `data/dossiers/`

## Important Patterns

### Frontend API calls
```typescript
axios.get('/v1/dossier/runs?limit=20').then(r => setRuns(r.data.data || []))
axios.post('/v1/dossier/runs', { question, ... }).then(() => ...)
```

### Backend async tasks
```typescript
_DOSSIER_TASKS[run_id] = asyncio.create_task(engine.run(...))
```

### Environment configuration
- Backend uses environment variables with defaults: `os.getenv("KEY", "default")`
- Config file: `ollama-ensemble-proxy/ensemble-proxy.env`

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | FastAPI routes, authentication, dossier management |
| `core/engine.py` | Main orchestration pipeline |
| `core/llm.py` | LLM API client wrapper |
| `core/config.py` | Configuration dataclass |
| `web-ui/src/App.tsx` | Single-page React application |

## Notes

- Frontend has no test framework configured
- Backend uses FastAPI with OAuth2 authentication
- LLM calls support both local Ollama and cloud endpoints
- Primary language for code comments/variables: English
- User-facing messages/labels in French