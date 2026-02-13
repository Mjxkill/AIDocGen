# Ollama Ensemble Proxy

## Project Overview

**Ollama Ensemble Proxy** is a specialized API service designed to orchestrate multiple Local LLMs (via Ollama) into a cohesive "ensemble" for complex tasks. Its primary feature is the **Deep Research Dossier** pipeline, which automates the creation of comprehensive, verified, and well-structured reports on arbitrary topics.

Instead of relying on a single model, this project assigns specific roles (Planner, Researcher, Writer, Reviewer) to different models (e.g., Qwen, Mistral, Llama, DeepSeek) to leverage their individual strengths.

## Key Features

-   **Deep Research Pipeline:** A multi-stage workflow including:
    1.  **Planning:** Decomposing questions into sub-questions and chapter outlines.
    2.  **Search:** Aggregating results from SearxNG or DuckDuckGo.
    3.  **Corpus Building:** Fetching and normalizing web content.
    4.  **Claim Extraction:** Extracting factual claims from sources.
    5.  **Verification:** Cross-checking claims against sources and other models.
    6.  **Writing:** Iterative drafting of long-form content.
-   **OpenAI Compatibility:** Exposes an API compatible with the OpenAI Chat Completions endpoint, allowing integration with tools like **Open-WebUI**.
-   **Resumability:** All run states are persisted to disk (`data/dossiers`), allowing long-running tasks to be paused, resumed, or inspected at any stage.
-   **Configurable Ensemble:** Model roles and pipeline parameters are fully configurable via environment variables.

## Architecture

*   **`app.py`**: The FastAPI application entry point. Handles HTTP requests, OpenAI compatibility layer, and routing.
*   **`dossier_engine.py`**: The core logic engine. Implements the state machine for the research pipeline, managing the transition between planning, searching, analyzing, and writing.
*   **`ensemble-proxy.env`**: Configuration file defining model assignments (e.g., `ENSEMBLE_DOSSIER_PLANNER_MODEL`), timeouts, and search settings.
*   **`data/dossiers/`**: Directory storing the artifacts of each run (JSON checkpoints, raw/clean text, final markdown reports).

## Setup & Installation

### Prerequisites

*   **Python 3.11+**
*   **Ollama** running locally (default: `http://127.0.0.1:11434`)
*   **(Optional) SearxNG** for better search results (default: `http://127.0.0.1:8080`)
*   **GPU(s)** capable of running the selected models.

### Installation

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    Copy or edit `ensemble-proxy.env` to match your available Ollama models and preferences.
    *   Ensure models defined in the `.env` file (e.g., `qwen2.5:32b`, `mistral-small3.2:24b`) are pulled in Ollama (`ollama pull <model>`).

## Running the Service

Start the server using `uvicorn`:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --env-file ensemble-proxy.env
```

The API will be available at `http://localhost:8001`.

## Usage

### 1. Via Open-WebUI (Interactive)

1.  Add a new **OpenAI-compatible connection** in Open-WebUI settings.
2.  **Base URL:** `http://localhost:8001/v1`
3.  **API Key:** (Any string, e.g., `sk-dummy`)
4.  Select the model **`ensemble-dossier-approfondi`** to trigger the deep research pipeline.

### 2. Via CLI / CURL (Batch Mode)

**Start a new run:**
```bash
curl -X POST http://127.0.0.1:8001/v1/dossier/runs \
  -H 'Content-Type: application/json' \
  -d '{"question": "Analyze the impact of quantum computing on cryptography", "background": true}'
```

**Check Status:**
```bash
curl http://127.0.0.1:8001/v1/dossier/runs/<run_id>
```

**Get Report:**
```bash
curl http://127.0.0.1:8001/v1/dossier/runs/<run_id>/report
```

## Development Conventions

*   **Type Hinting:** Codebase uses strict Python type hints.
*   **Async/Await:** The project is fully asynchronous to handle concurrent LLM requests and web fetching.
*   **Error Handling:** The `DossierEngine` is designed to be robust, with retries for LLM calls and state preservation on failure.
*   **No "Blind" Fallbacks:** The engine prefers explicit errors over silent hallucinations. If a model fails to produce valid JSON after retries, the step fails to alert the user.

## Current Status

See `SUITE_OPERATIONS_ATTENTE.md` for the current backlog and pending operations, including infrastructure improvements and future feature plans.
