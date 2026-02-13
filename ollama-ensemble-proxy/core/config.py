import os
from dataclasses import dataclass

def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

@dataclass
class DossierConfig:
    ollama_base_url: str
    data_dir: str
    planner_model: str
    planner_panel_models_csv: str
    planner_synth_model: str
    planner_book_model_1: str
    planner_book_model_2: str
    planner_book_model_3: str
    planner_book_model_4_json: str
    planner_book_web_links: int
    planner_book_page_chars: int
    outline_lenses_csv: str
    outline_refinement_rounds: int
    outline_min_subsections: int
    outline_max_subsections: int
    extract_model: str
    verify_model: str
    writer_model: str
    judge_model: str
    searxng_base_url: str
    web_max_sub_questions: int
    web_max_links_per_subquestion: int
    web_fetch_limit_per_subquestion: int
    web_shortlist_per_subquestion: int
    web_query_variants: int
    web_per_query_results: int
    presearch_query_variants: int
    presearch_per_query_results: int
    presearch_max_links: int
    web_timeout_seconds: float
    web_region: str
    web_safesearch: str
    chunk_size: int
    chunk_overlap: int
    max_claims_per_source: int
    llm_timeout_seconds: float
    planner_timeout_seconds: float
    writer_timeout_seconds: float
    context_window: int
    max_parallel_fetch: int
    max_parallel_llm: int
    llm_retry_attempts: int
    writer_iterations: int
    writer_batch_claims: int
    writer_min_words_per_section: int
    writer_target_words_per_section: int
    strict_no_fallback: bool
    web_search_engine: str
    web_request_delay: float

    @classmethod
    def from_env(cls) -> "DossierConfig":
        outline_min_subsections = max(
            0,
            int(os.getenv("ENSEMBLE_DOSSIER_OUTLINE_MIN_SUBSECTIONS", "0")),
        )
        outline_max_subsections = int(os.getenv("ENSEMBLE_DOSSIER_OUTLINE_MAX_SUBSECTIONS", "0"))
        if outline_max_subsections > 0 and outline_max_subsections < outline_min_subsections:
            outline_max_subsections = outline_min_subsections
        return cls(
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            data_dir=os.getenv(
                "ENSEMBLE_DOSSIER_DATA_DIR",
                "/root/codex/ollama-ensemble-proxy/data/dossiers",
            ),
            planner_model=os.getenv("ENSEMBLE_DOSSIER_PLANNER_MODEL", "qwen2.5:32b"),
            planner_panel_models_csv=os.getenv("ENSEMBLE_DOSSIER_PLANNER_PANEL_MODELS", ""),
            planner_synth_model=os.getenv(
                "ENSEMBLE_DOSSIER_PLANNER_SYNTH_MODEL",
                os.getenv("ENSEMBLE_DOSSIER_JUDGE_MODEL", "qwen2.5:32b"),
            ),
            planner_book_model_1=os.getenv("ENSEMBLE_DOSSIER_BOOK_MODEL_1", "gemma2:27b"),
            planner_book_model_2=os.getenv("ENSEMBLE_DOSSIER_BOOK_MODEL_2", "mistral-small3.2:24b"),
            planner_book_model_3=os.getenv("ENSEMBLE_DOSSIER_BOOK_MODEL_3", "qwen3:32b"),
            planner_book_model_4_json=os.getenv(
                "ENSEMBLE_DOSSIER_BOOK_MODEL_4_JSON",
                "qwen2.5-coder:32b",
            ),
            planner_book_web_links=max(8, int(os.getenv("ENSEMBLE_DOSSIER_BOOK_WEB_LINKS", "60"))),
            planner_book_page_chars=max(2200, int(os.getenv("ENSEMBLE_DOSSIER_BOOK_PAGE_CHARS", "7000"))),
            outline_lenses_csv=os.getenv(
                "ENSEMBLE_DOSSIER_OUTLINE_LENSES",
                (
                    "definitions et perimetre,faits etablis et mecanismes,benefices et opportunites,"
                    "limites et risques,cout et ressources,environnement,societe et politique,"
                    "aspects techniques ou operationnels,exemples d'application"
                ),
            ),
            outline_refinement_rounds=max(0, int(os.getenv("ENSEMBLE_DOSSIER_OUTLINE_REFINEMENT_ROUNDS", "3"))),
            outline_min_subsections=outline_min_subsections,
            outline_max_subsections=outline_max_subsections,
            extract_model=os.getenv("ENSEMBLE_DOSSIER_EXTRACT_MODEL", "mistral-small3.2:24b"),
            verify_model=os.getenv("ENSEMBLE_DOSSIER_VERIFY_MODEL", "qwen2.5:32b"),
            writer_model=os.getenv("ENSEMBLE_DOSSIER_WRITER_MODEL", "mistral-small3.2:24b"),
            judge_model=os.getenv("ENSEMBLE_DOSSIER_JUDGE_MODEL", "qwen2.5:32b"),
            searxng_base_url=os.getenv("ENSEMBLE_SEARXNG_BASE_URL", "http://127.0.0.1:8080"),
            web_max_sub_questions=max(0, int(os.getenv("ENSEMBLE_DOSSIER_MAX_SUBQUESTIONS", "0"))),
            web_max_links_per_subquestion=int(os.getenv("ENSEMBLE_DOSSIER_MAX_LINKS_PER_SUBQUESTION", "240")),
            web_fetch_limit_per_subquestion=int(os.getenv("ENSEMBLE_DOSSIER_FETCH_LIMIT_PER_SUBQUESTION", "70")),
            web_shortlist_per_subquestion=int(os.getenv("ENSEMBLE_DOSSIER_SHORTLIST_PER_SUBQUESTION", "18")),
            web_query_variants=int(os.getenv("ENSEMBLE_DOSSIER_QUERY_VARIANTS", "6")),
            web_per_query_results=int(os.getenv("ENSEMBLE_DOSSIER_PER_QUERY_RESULTS", "80")),
            presearch_query_variants=max(2, int(os.getenv("ENSEMBLE_DOSSIER_PRESEARCH_QUERY_VARIANTS", "5"))),
            presearch_per_query_results=max(8, int(os.getenv("ENSEMBLE_DOSSIER_PRESEARCH_PER_QUERY_RESULTS", "40"))),
            presearch_max_links=max(20, int(os.getenv("ENSEMBLE_DOSSIER_PRESEARCH_MAX_LINKS", "160"))),
            web_timeout_seconds=float(os.getenv("ENSEMBLE_DOSSIER_WEB_TIMEOUT_SECONDS", "25")),
            web_region=os.getenv("ENSEMBLE_DOSSIER_WEB_REGION", "wt-wt"),
            web_safesearch=os.getenv("ENSEMBLE_DOSSIER_WEB_SAFESEARCH", "moderate"),
            chunk_size=int(os.getenv("ENSEMBLE_DOSSIER_CHUNK_SIZE", "1400")),
            chunk_overlap=int(os.getenv("ENSEMBLE_DOSSIER_CHUNK_OVERLAP", "180")),
            max_claims_per_source=int(os.getenv("ENSEMBLE_DOSSIER_MAX_CLAIMS_PER_SOURCE", "6")),
            llm_timeout_seconds=float(os.getenv("ENSEMBLE_DOSSIER_LLM_TIMEOUT_SECONDS", "7200")),
            planner_timeout_seconds=float(os.getenv("ENSEMBLE_DOSSIER_PLANNER_TIMEOUT_SECONDS", "300")),
            writer_timeout_seconds=float(os.getenv("ENSEMBLE_DOSSIER_WRITER_TIMEOUT_SECONDS", "1200")),
            context_window=int(os.getenv("ENSEMBLE_DOSSIER_CONTEXT_WINDOW", "32000")),
            max_parallel_fetch=int(os.getenv("ENSEMBLE_DOSSIER_MAX_PARALLEL_FETCH", "8")),
            max_parallel_llm=int(os.getenv("ENSEMBLE_DOSSIER_MAX_PARALLEL_LLM", "2")),
            llm_retry_attempts=max(1, int(os.getenv("ENSEMBLE_DOSSIER_LLM_RETRY_ATTEMPTS", "3"))),
            writer_iterations=max(1, int(os.getenv("ENSEMBLE_DOSSIER_WRITER_ITERATIONS", "6"))),
            writer_batch_claims=max(6, int(os.getenv("ENSEMBLE_DOSSIER_WRITER_BATCH_CLAIMS", "24"))),
            writer_min_words_per_section=max(400, int(os.getenv("ENSEMBLE_DOSSIER_MIN_WORDS_PER_SECTION", "1400"))),
            writer_target_words_per_section=max(800, int(os.getenv("ENSEMBLE_DOSSIER_TARGET_WORDS_PER_SECTION", "2200"))),
            strict_no_fallback=env_bool("ENSEMBLE_DOSSIER_STRICT_NO_FALLBACK", False),
            web_search_engine=os.getenv("ENSEMBLE_WEB_SEARCH_ENGINE", "auto").strip().lower(),
            web_request_delay=float(os.getenv("ENSEMBLE_WEB_REQUEST_DELAY", "1.5")),
        )
