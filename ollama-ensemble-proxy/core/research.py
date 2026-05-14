import asyncio
import hashlib
import os
import time
import re
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
import httpx
from duckduckgo_search import DDGS

from .config import DossierConfig
from .utils import canonicalize_url, emit_progress
from .logging_config import get_logger
from . import cost as _cost

log = get_logger("aidocgen.research")


_BASE_DIR = Path(__file__).parent.parent.resolve()

# contextvars carry per-run identity (user, run_id) into the cost recorder
# without having to thread parameters through every function. Set by
# app.py around the `engine.run(...)` call.
import contextvars as _ctxvars
_ctx_user_id: _ctxvars.ContextVar[str | None] = _ctxvars.ContextVar("user_id", default=None)
_ctx_user_name: _ctxvars.ContextVar[str | None] = _ctxvars.ContextVar("user_name", default=None)
_ctx_run_id: _ctxvars.ContextVar[str | None] = _ctxvars.ContextVar("run_id", default=None)


def set_dossier_context(user_id: str | None, user_name: str | None,
                        run_id: str | None) -> None:
    _ctx_user_id.set(user_id or None)
    _ctx_user_name.set(user_name or None)
    _ctx_run_id.set(run_id or None)


def _record_cost(provider: str, model: str, kind: str, units: float,
                 user_id: str | None = None, user_name: str | None = None,
                 job_id: str | None = None) -> None:
    """Best-effort cost record for each external API call."""
    try:
        _cost.record(_BASE_DIR / "data",
                     provider=provider, model=model, kind=kind, units=units,
                     user_id=user_id or _ctx_user_id.get(),
                     user_name=user_name or _ctx_user_name.get(),
                     job_id=job_id or _ctx_run_id.get())
    except Exception as e:
        log.debug("cost record failed", extra={"error": str(e)})


_DATE_PATTERNS = [
    # ISO-ish: 2026-05-04, 2025-12-31
    re.compile(r"\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b"),
    # Natural: "May 2026", "mai 2026"
    re.compile(
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|"
        r"jan|fév|mar|avr|mai|juin|juil|aou|sep|oct|nov|déc)[a-zéû]*\s+(20\d{2})\b",
        re.IGNORECASE,
    ),
    # Bare year: 2026
    re.compile(r"\b(20\d{2})\b"),
]


def _parse_published_date(s) -> datetime | None:
    """Parse ISO 8601 datetime or 'YYYY-MM-DD' string. Returns None if not parseable.
    Tolerates list/tuple inputs (Firecrawl sometimes returns list-valued meta)
    by taking the first element."""
    if not s:
        return None
    if isinstance(s, (list, tuple)):
        s = next((x for x in s if x), None)
        if not s:
            return None
    if not isinstance(s, str):
        try:
            s = str(s)
        except Exception:
            return None
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S.%fZ"):
        try:
            dt = datetime.strptime(s.replace("Z", "+0000") if fmt.endswith("%z") else s, fmt)
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
        except ValueError:
            continue
    # Last resort: extract year
    m = re.search(r"\b(20\d{2})\b", s)
    if m:
        try:
            return datetime(int(m.group(1)), 6, 15, tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def _recency_score(published: datetime | None, recency_days: int) -> float:
    """Returns 1.0 if within recency_days, decays linearly to 0.0 at 4× recency_days."""
    if not published or recency_days <= 0:
        return 0.5
    age_days = (datetime.now(timezone.utc) - published).days
    if age_days < 0:
        return 1.0
    if age_days <= recency_days:
        return 1.0
    if age_days >= 4 * recency_days:
        return 0.0
    return max(0.0, 1.0 - (age_days - recency_days) / (3 * recency_days))

class WebResearcher:
    def __init__(self, config: DossierConfig):
        self.config = config

    def _filter_by_tags(self, text: str, tags: list[str]) -> bool:
        """Check if text contains at least one of the required tags."""
        if not tags:
            return True
        text_lower = text.lower()
        for tag in tags:
            tag_lower = tag.lower()
            tag_no_special = re.sub(r'[^a-z0-9]', '', tag_lower)
            if tag_lower in text_lower or tag_no_special in re.sub(r'[^a-z0-9]', '', text_lower):
                return True
        return False

    def _enhance_query_with_tags(self, query: str, tags: list[str]) -> str:
        """Add tags to search query if not already present."""
        if not tags:
            return query
        query_lower = query.lower()
        tags_to_add = []
        for tag in tags[:3]:
            tag_lower = tag.lower()
            if tag_lower not in query_lower:
                tags_to_add.append(tag)
        if tags_to_add:
            return f"{query} {' '.join(tags_to_add)}"
        return query

    def _add_year_to_query(self, query: str) -> str:
        """Add current year to query to favor recent results."""
        year = str(date.today().year)
        if year not in query:
            return f"{query} {year}"
        return query

    # ─────────────────────────────────────────────
    # FIRECRAWL ADVANCED: /map, /crawl, /extract
    # ─────────────────────────────────────────────

    async def _firecrawl_map(self, url: str, limit: int = 50) -> list[str]:
        """Discover all pages on a domain via Firecrawl /map API.
        Returns a list of URLs found on the site."""
        if not self.config.firecrawl_api_key:
            return []
        fc_url = "https://api.firecrawl.dev/v1/map"
        headers = {"Authorization": f"Bearer {self.config.firecrawl_api_key}"}
        payload = {"url": url, "limit": limit}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(fc_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    _record_cost("firecrawl", "map", "credit", 1)
                    data = resp.json()
                    return data.get("links", [])[:limit]
        except Exception as e:
            log.warning("Firecrawl /map error for", extra={"url": url, "error": str(e)})
        return []

    async def _firecrawl_crawl(self, url: str, max_pages: int = 10) -> list[dict[str, Any]]:
        """Crawl an entire site via Firecrawl /crawl API (async job).
        Returns list of {url, markdown, title} for each crawled page."""
        if not self.config.firecrawl_api_key:
            return []
        fc_url = "https://api.firecrawl.dev/v1/crawl"
        headers = {"Authorization": f"Bearer {self.config.firecrawl_api_key}"}
        payload = {
            "url": url,
            "limit": max_pages,
            "scrapeOptions": {"formats": ["markdown"]},
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(fc_url, json=payload, headers=headers)
                if resp.status_code not in (200, 201):
                    return []
                data = resp.json()
                if not data.get("success"):
                    return []
                crawl_id = data.get("id")
                if not crawl_id:
                    return []

            # Poll for completion (max 120s)
            poll_url = f"https://api.firecrawl.dev/v1/crawl/{crawl_id}"
            async with httpx.AsyncClient(timeout=15.0) as client:
                for _ in range(24):  # 24 * 5s = 120s
                    await asyncio.sleep(5)
                    resp = await client.get(poll_url, headers=headers)
                    if resp.status_code != 200:
                        continue
                    result = resp.json()
                    status = result.get("status", "")
                    if status == "completed":
                        pages = []
                        for item in result.get("data", []):
                            pages.append({
                                "url": item.get("metadata", {}).get("sourceURL", url),
                                "markdown": item.get("markdown", ""),
                                "title": item.get("metadata", {}).get("title", ""),
                            })
                        if pages:
                            _record_cost("firecrawl", "crawl", "credit", len(pages))
                        return pages
                    elif status in ("failed", "cancelled"):
                        return []
        except Exception as e:
            log.warning("Firecrawl /crawl error for", extra={"url": url, "error": str(e)})
        return []

    async def _firecrawl_extract(self, url: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        """Extract structured data from a page using Firecrawl /extract with LLM."""
        if not self.config.firecrawl_api_key or not self.config.firecrawl_extract_enabled:
            return None
        fc_url = "https://api.firecrawl.dev/v1/extract"
        headers = {"Authorization": f"Bearer {self.config.firecrawl_api_key}"}
        payload = {
            "urls": [url],
            "prompt": "Extract all key facts, data points, specifications, comparisons, and conclusions from this page.",
            "schema": schema,
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(fc_url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success"):
                        _record_cost("firecrawl", "extract", "credit", 5)
                        return data.get("data", {})
        except Exception as e:
            log.warning("Firecrawl /extract error for", extra={"url": url, "error": str(e)})
        return None

    # ─────────────────────────────────────────────
    # PRESEARCH
    # ─────────────────────────────────────────────

    async def presearch(self, question: str, tags: list[str] = None) -> list[dict[str, str]]:
        queries = [self._add_year_to_query(question)]
        if tags:
            queries.append(self._add_year_to_query(self._enhance_query_with_tags(question, tags)))

        results = []
        try:
            with DDGS() as ddgs:
                for q in queries[:2]:
                    for res in ddgs.text(q, max_results=5):
                        title = res.get("title", "")
                        snippet = res.get("body", "")
                        if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                            continue
                        results.append({
                            "title": title,
                            "url": res.get("href", ""),
                            "snippet": snippet,
                        })
        except Exception:
            pass
        return results

    # ─────────────────────────────────────────────
    # PARALLEL MULTI-ENGINE SEARCH
    # ─────────────────────────────────────────────

    async def _search_ddg(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        """DuckDuckGo search."""
        links = []
        try:
            await asyncio.sleep(self.config.web_request_delay)
            with DDGS() as ddgs:
                for res in ddgs.text(query, max_results=self.config.web_per_query_results):
                    title = res.get("title", "")
                    snippet = res.get("body", "")
                    if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                        continue
                    links.append({"title": title, "url": res.get("href", ""), "snippet": snippet, "engine": "ddg"})
        except Exception as e:
            log.warning("DDG Search error", extra={"error": str(e)})
        return links

    async def _search_searxng(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        url = f"{self.config.searxng_base_url.rstrip('/')}/search"
        params = {"q": query, "format": "json", "categories": "general", "language": "fr-FR"}
        links = []
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, params=params)
                if resp.status_code != 200:
                    return []
                # Defensive: a non-JSON response (eg. SearxNG behind login,
                # or wrong port serving a different app) would raise here.
                try:
                    data = resp.json()
                except Exception:
                    return []
                for res in data.get("results", [])[:self.config.web_per_query_results]:
                    title = res.get("title", "")
                    snippet = res.get("content", "")
                    if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                        continue
                    links.append({"title": title, "url": res.get("url"), "snippet": snippet, "engine": "searxng"})
        except Exception as e:
            log.warning("SearxNG API error", extra={"error": str(e)})
        return links

    async def _search_wikipedia(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        api_url = "https://fr.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 10}
        # Wikimedia requires a descriptive User-Agent since 2025-Q3, otherwise 403.
        headers = {"User-Agent": "AIDocGen/1.0 (https://electrosens.fr; noreply@office.electrosens.fr)"}
        links = []
        try:
            async with httpx.AsyncClient(timeout=10, headers=headers) as client:
                resp = await client.get(api_url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for res in data.get("query", {}).get("search", []):
                        title = res.get("title", "")
                        snippet = res.get("snippet", "")
                        if tags and len(tags) > 0 and len(tags[0]) > 4:
                            if not self._filter_by_tags(f"{title} {snippet}", tags):
                                continue
                        links.append({
                            "title": title,
                            "url": f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "snippet": snippet,
                            "engine": "wikipedia"
                        })
        except Exception:
            pass
        return links

    async def _search_firecrawl(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        """Firecrawl /v1/search — high-quality LLM-friendly search results.
        Honors recency_days as Google `tbs=qdr:d/w/m/y` filter."""
        if not self.config.firecrawl_api_key:
            return []
        rd = self.config.recency_days
        tbs = None
        if rd > 0:
            tbs = "qdr:y" if rd > 90 else "qdr:m" if rd > 7 else "qdr:w"
        # Firecrawl /search caps at 50 per query on most plans
        payload: dict[str, Any] = {
            "query": query,
            "limit": min(50, max(20, self.config.web_per_query_results)),
            "lang": "fr",
        }
        if tbs:
            payload["tbs"] = tbs
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    "https://api.firecrawl.dev/v1/search",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.config.firecrawl_api_key}"},
                )
                if resp.status_code != 200:
                    log.warning("Firecrawl /search non-200",
                                extra={"status": resp.status_code, "query": query[:60]})
                    return []
                data = resp.json()
                _record_cost("firecrawl", "search", "credit", 1)
                results = data.get("data", []) if isinstance(data.get("data"), list) else data.get("results", [])
                links = []
                for r in results:
                    title = r.get("title", "") or r.get("name", "")
                    snippet = r.get("description", "") or r.get("snippet", "") or r.get("content", "")
                    url = r.get("url", "") or r.get("link", "")
                    if not url:
                        continue
                    if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                        continue
                    links.append({
                        "title": title, "url": url, "snippet": snippet,
                        "engine": "firecrawl_search",
                        "published_at": r.get("publishedDate") or r.get("date") or "",
                    })
                return links
        except Exception as e:
            log.warning("Firecrawl /search error", extra={"error": str(e), "query": query[:60]})
            return []

    async def _search_tavily(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        """Tavily — search optimised for LLM context. Has native `days` recency."""
        if not self.config.tavily_api_key:
            return []
        payload: dict[str, Any] = {
            "api_key": self.config.tavily_api_key,
            "query": query,
            # Tavily caps at 20 advanced results
            "max_results": min(20, max(10, self.config.web_per_query_results)),
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": False,
        }
        if self.config.recency_days > 0:
            payload["days"] = self.config.recency_days
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post("https://api.tavily.com/search", json=payload)
                if resp.status_code != 200:
                    log.warning("Tavily non-200",
                                extra={"status": resp.status_code, "query": query[:60]})
                    return []
                data = resp.json()
                _record_cost("tavily", "search", "credit", 1)
                links = []
                for r in data.get("results", []):
                    title = r.get("title", "")
                    snippet = r.get("content", "") or r.get("snippet", "")
                    url = r.get("url", "")
                    if not url:
                        continue
                    if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                        continue
                    links.append({
                        "title": title, "url": url, "snippet": snippet,
                        "engine": "tavily",
                        "published_at": r.get("published_date") or "",
                    })
                return links
        except Exception as e:
            log.warning("Tavily error", extra={"error": str(e), "query": query[:60]})
            return []

    async def _search_arxiv(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        """arXiv — academic papers. No API key needed. Filters by submitted_date.
        arXiv often returns 503 under load; retry once after a short backoff."""
        if not self.config.arxiv_enabled:
            return []
        params: dict[str, Any] = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 10,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get("https://export.arxiv.org/api/query", params=params)
                if resp.status_code == 503:
                    await asyncio.sleep(3)
                    resp = await client.get("https://export.arxiv.org/api/query", params=params)
                if resp.status_code != 200:
                    return []
                # Parse Atom XML loosely
                xml = resp.text
                entries = re.split(r"<entry>", xml)[1:]
                cutoff = None
                if self.config.recency_days > 0:
                    cutoff = datetime.now(timezone.utc) - timedelta(days=self.config.recency_days)
                links = []
                for e in entries:
                    title_m = re.search(r"<title>([\s\S]*?)</title>", e)
                    summary_m = re.search(r"<summary>([\s\S]*?)</summary>", e)
                    link_m = re.search(r'<link[^>]+href="(http[^"]*?abs/[^"]*)"', e)
                    pub_m = re.search(r"<published>([^<]+)</published>", e)
                    if not title_m or not link_m:
                        continue
                    title = re.sub(r"\s+", " ", title_m.group(1)).strip()
                    snippet = re.sub(r"\s+", " ", (summary_m.group(1) if summary_m else "")).strip()[:400]
                    url = link_m.group(1).strip()
                    pub = pub_m.group(1).strip() if pub_m else ""
                    if cutoff and pub:
                        dt = _parse_published_date(pub)
                        if dt and dt < cutoff:
                            continue
                    if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                        continue
                    links.append({
                        "title": f"[arXiv] {title}", "url": url, "snippet": snippet,
                        "engine": "arxiv", "published_at": pub,
                    })
                return links
        except Exception as e:
            log.warning("arXiv error", extra={"error": str(e), "query": query[:60]})
            return []

    async def _search_pubmed(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        """PubMed — biomedical literature via NCBI E-utilities (no key needed)."""
        if not self.config.pubmed_enabled:
            return []
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                params: dict[str, Any] = {
                    "db": "pubmed", "term": query, "retmode": "json",
                    "retmax": 10, "sort": "pub_date",
                }
                if self.config.recency_days > 0:
                    end = datetime.now(timezone.utc).date()
                    start = end - timedelta(days=self.config.recency_days)
                    params["term"] = (
                        f"{query} AND ({start.strftime('%Y/%m/%d')}"
                        f"[Date - Publication] : "
                        f"{end.strftime('%Y/%m/%d')}[Date - Publication])"
                    )
                resp = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    params=params,
                )
                if resp.status_code != 200:
                    return []
                ids = resp.json().get("esearchresult", {}).get("idlist", [])
                if not ids:
                    return []
                # Fetch summaries for these ids
                summary_resp = await client.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                    params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
                )
                if summary_resp.status_code != 200:
                    return []
                items = summary_resp.json().get("result", {})
                links = []
                for pmid in ids:
                    rec = items.get(pmid)
                    if not rec:
                        continue
                    title = rec.get("title", "")
                    pub = rec.get("pubdate", "")
                    snippet = (rec.get("authors") or [{}])[0].get("name", "") + (
                        f" — {pub}" if pub else "")
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                    if tags and not self._filter_by_tags(f"{title} {snippet}", tags):
                        continue
                    links.append({
                        "title": f"[PubMed] {title}", "url": url, "snippet": snippet,
                        "engine": "pubmed", "published_at": pub,
                    })
                return links
        except Exception as e:
            log.warning("PubMed error", extra={"error": str(e), "query": query[:60]})
            return []

    async def _search_firecrawl_discover(self, query: str, tags: list[str] = None) -> list[dict[str, str]]:
        """Use Firecrawl /map on top domains from DDG to discover deeper pages."""
        if not self.config.firecrawl_api_key:
            return []
        # First get a few DDG results to find relevant domains
        links = []
        try:
            with DDGS() as ddgs:
                top_results = list(ddgs.text(query, max_results=3))
        except Exception:
            return []

        # Map each domain to discover related pages
        seen_domains = set()
        for res in top_results:
            url = res.get("href", "")
            if not url:
                continue
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain in seen_domains:
                continue
            seen_domains.add(domain)
            mapped_urls = await self._firecrawl_map(url, limit=15)
            for mapped_url in mapped_urls:
                if tags and not self._filter_by_tags(mapped_url, tags):
                    continue
                links.append({
                    "title": f"[Discovered] {mapped_url.split('/')[-1][:60]}",
                    "url": mapped_url,
                    "snippet": f"Page découverte via exploration du domaine {domain}",
                    "engine": "firecrawl_map"
                })
        return links

    async def search_subquestions(
        self,
        planner: dict[str, Any],
        progress_cb: Callable | None,
        run_dir: Path | None,
        tags: list[str] = None,
    ) -> dict[str, Any]:
        all_searches = []
        sub_questions = planner.get("sub_questions", [])
        if not sub_questions:
            sub_questions = [{"id": "SQ1", "question": planner.get("question_reformulated", "Dossier")}]

        total_sq = len(sub_questions)

        for idx, sq in enumerate(sub_questions, 1):
            sq_id = sq.get("id")
            query = sq.get("question")

            if tags:
                query = self._enhance_query_with_tags(query, tags)
            query = self._add_year_to_query(query)

            if run_dir:
                await emit_progress(progress_cb, run_dir, "search", f"Searching {idx}/{total_sq}: {query[:50]}...")

            # ── PARALLEL MULTI-ENGINE SEARCH ──
            # Quality-first: Firecrawl /search + Tavily on every query, plus
            # legacy engines + Wikipedia + arXiv + PubMed in parallel.
            search_tasks = [
                self._search_firecrawl(query, tags),  # primary
                self._search_tavily(query, tags),     # primary
                self._search_ddg(query, tags),
                self._search_searxng(query, tags),
                self._search_wikipedia(query, tags),
                self._search_arxiv(query, tags),
                self._search_pubmed(query, tags),
            ]
            # Domain-discovery on every query (Firecrawl /map → broaden coverage)
            if self.config.firecrawl_api_key:
                search_tasks.append(self._search_firecrawl_discover(query, tags))

            engine_results = await asyncio.gather(*search_tasks, return_exceptions=True)

            links = []
            seen_urls = set()
            for result in engine_results:
                if isinstance(result, Exception):
                    continue
                for link in result:
                    url = canonicalize_url(link.get("url", ""))
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        links.append(link)

            all_searches.append({"id": sq_id, "query": query, "links": links})

        return {"sub_questions": all_searches}

    # ─────────────────────────────────────────────
    # CORPUS BUILDING (with Firecrawl deep crawl)
    # ─────────────────────────────────────────────

    async def build_corpus(
        self,
        search_results: dict[str, Any],
        progress_cb: Callable | None,
        run_dir: Path,
        tags: list[str] = None,
    ) -> dict[str, Any]:
        unique_urls = {}
        for sq in search_results.get("sub_questions", []):
            for link in sq.get("links", []):
                u = canonicalize_url(link.get("url"))
                if u and u not in unique_urls:
                    unique_urls[u] = link

        urls_to_fetch = list(unique_urls.values())
        if run_dir:
            await emit_progress(progress_cb, run_dir, "corpus", f"Fetching {len(urls_to_fetch)} sources")

        # ── Phase 1: Fetch individual pages ──
        fetched_sources = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_fetch)
        processed = 0

        async def fetch_one(link: dict[str, str]):
            nonlocal processed
            async with semaphore:
                source = await self._fetch_url(link["url"], run_dir, tags)
                processed += 1
                if processed % 10 == 0 and run_dir:
                    await emit_progress(progress_cb, run_dir, "corpus", f"Processed {processed}/{len(urls_to_fetch)}")
                return source

        results = await asyncio.gather(*(fetch_one(l) for l in urls_to_fetch))
        fetched_sources = [r for r in results if r]

        # ── Phase 2: Deep crawl top domains with Firecrawl ──
        if self.config.firecrawl_api_key:
            fetched_sources = await self._deep_crawl_top_domains(
                fetched_sources, run_dir, tags, progress_cb
            )

        # ── Phase 3: Structured extraction on key pages ──
        if self.config.firecrawl_api_key and self.config.firecrawl_extract_enabled:
            await self._extract_structured_data(fetched_sources, run_dir, progress_cb)

        return {
            "sources": fetched_sources,
            "count": len(fetched_sources),
            "generated_at": int(time.time())
        }

    async def _deep_crawl_top_domains(
        self,
        sources: list[dict[str, Any]],
        run_dir: Path,
        tags: list[str] | None,
        progress_cb: Callable | None,
    ) -> list[dict[str, Any]]:
        """Identify top domains and crawl them deeply for more content."""
        # Count domains
        domain_counts = Counter()
        for s in sources:
            domain = s.get("domain", "")
            if domain:
                domain_counts[domain] += 1

        # Pick top N domains with most hits (likely most relevant)
        top_n = max(3, self.config.firecrawl_deep_crawl_top_n)
        top_domains = [d for d, _ in domain_counts.most_common(top_n) if domain_counts[d] >= 2]
        if not top_domains:
            return sources

        existing_urls = {s.get("canonical_url", "") for s in sources}

        if run_dir:
            await emit_progress(progress_cb, run_dir, "corpus",
                f"Deep crawl: exploring {len(top_domains)} key domains...")

        for domain in top_domains:
            # Find a representative URL for this domain
            domain_url = next(
                (s["url"] for s in sources if s.get("domain") == domain),
                f"https://{domain}"
            )

            # Use /map to find all pages
            mapped_urls = await self._firecrawl_map(domain_url, limit=20)
            new_urls = [u for u in mapped_urls if canonicalize_url(u) not in existing_urls][:self.config.firecrawl_crawl_max_pages]

            if not new_urls:
                continue

            # Fetch each discovered page
            for url in new_urls:
                source = await self._fetch_url(url, run_dir, tags)
                if source:
                    sources.append(source)
                    existing_urls.add(source.get("canonical_url", ""))

        if run_dir:
            await emit_progress(progress_cb, run_dir, "corpus",
                f"Deep crawl complete: {len(sources)} total sources")

        return sources

    async def _extract_structured_data(
        self,
        sources: list[dict[str, Any]],
        run_dir: Path,
        progress_cb: Callable | None,
    ) -> None:
        """Run Firecrawl /extract on key pages to get structured data."""
        extraction_schema = {
            "type": "object",
            "properties": {
                "key_facts": {"type": "array", "items": {"type": "string"}},
                "specifications": {"type": "object"},
                "conclusions": {"type": "array", "items": {"type": "string"}},
                "data_points": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string"},
                        "value": {"type": "string"},
                    }
                }},
            }
        }

        # Extract from first N sources (configurable; default 20)
        top_n = max(5, self.config.firecrawl_extract_top_n)
        extracted_count = 0
        for source in sources[:top_n]:
            url = source.get("url", "")
            if not url:
                continue
            extracted = await self._firecrawl_extract(url, extraction_schema)
            if extracted:
                # Append structured data to the source's content file
                content_path = run_dir / source.get("content_path", "")
                if content_path.exists():
                    existing = content_path.read_text(encoding="utf-8", errors="ignore")
                    structured_section = "\n\n---\n## STRUCTURED DATA (auto-extracted)\n"
                    if extracted.get("key_facts"):
                        structured_section += "\n### Key Facts\n" + "\n".join(f"- {f}" for f in extracted["key_facts"])
                    if extracted.get("specifications"):
                        structured_section += "\n### Specifications\n"
                        for k, v in extracted["specifications"].items():
                            structured_section += f"- **{k}**: {v}\n"
                    if extracted.get("conclusions"):
                        structured_section += "\n### Conclusions\n" + "\n".join(f"- {c}" for c in extracted["conclusions"])
                    if extracted.get("data_points"):
                        structured_section += "\n### Data Points\n"
                        for dp in extracted["data_points"]:
                            structured_section += f"- {dp.get('metric', '?')}: {dp.get('value', '?')}\n"
                    content_path.write_text(existing + structured_section, encoding="utf-8")
                    extracted_count += 1

        if extracted_count > 0 and run_dir:
            await emit_progress(progress_cb, run_dir, "corpus",
                f"Structured extraction done on {extracted_count} key sources")

    # ─────────────────────────────────────────────
    # URL FETCHING
    # ─────────────────────────────────────────────

    async def _fetch_url(self, url: str, run_dir: Path, tags: list[str] = None) -> dict[str, Any] | None:
        final_url = canonicalize_url(url)
        sid = f"SRC-{hashlib.sha1(final_url.encode()).hexdigest()[:12]}"

        # Try Firecrawl first if API key is configured
        if self.config.firecrawl_api_key:
            try:
                fc_result = await self._fetch_firecrawl(url, final_url, sid, run_dir, tags)
                if fc_result:
                    return fc_result
            except Exception as e:
                log.warning("Firecrawl failed for", extra={"url": url, "error": str(e)})

        # Fallback to basic scraping
        return await self._fetch_bs4(url, final_url, sid, run_dir, tags)

    @staticmethod
    def _filter_image_urls(items: list[dict[str, str]], base_url: str = "") -> list[dict[str, str]]:
        """Drop data URIs, tracking pixels, tiny icons, dedupe."""
        from urllib.parse import urljoin
        seen, out = set(), []
        for it in items:
            url = (it.get("url") or "").strip()
            if not url or url.startswith(("data:", "javascript:")):
                continue
            if base_url and not url.startswith(("http://", "https://")):
                url = urljoin(base_url, url)
            if not url.startswith(("http://", "https://")):
                continue
            low = url.lower()
            # Skip obvious icons / sprites / tracking
            if any(p in low for p in ("/icon", "/icons/", "/sprite", "/spacer", "1x1.", "pixel.", "favicon", "apple-touch")):
                continue
            if low.split("?")[0].endswith((".svg", ".ico", ".gif")):
                continue
            if url in seen:
                continue
            seen.add(url)
            alt = (it.get("alt") or "").strip()[:200]
            out.append({"url": url, "alt": alt})
            if len(out) >= 8:
                break
        return out

    @staticmethod
    def _images_from_markdown(md: str, base_url: str = "") -> list[dict[str, str]]:
        items = []
        for m in re.finditer(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)', md or ""):
            items.append({"alt": m.group(1), "url": m.group(2)})
        return WebResearcher._filter_image_urls(items, base_url)

    async def _fetch_firecrawl(self, url: str, final_url: str, sid: str, run_dir: Path, tags: list[str] = None) -> dict[str, Any] | None:
        """Fetch content via Firecrawl API."""
        fc_url = "https://api.firecrawl.dev/v1/scrape"
        headers = {"Authorization": f"Bearer {self.config.firecrawl_api_key}"}
        payload = {"url": url, "formats": ["markdown"]}

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(fc_url, json=payload, headers=headers)
            if resp.status_code != 200:
                return None
            _record_cost("firecrawl", "scrape", "credit", 1)

            data = resp.json().get("data", {})
            markdown = data.get("markdown", "")
            meta = data.get("metadata", {}) or {}
            title = meta.get("title", "")

            if not markdown:
                return None

            # Filter by tags if provided
            if tags and not self._filter_by_tags(f"{title} {markdown[:5000]}", tags):
                return None

            # Save content
            clean_dir = run_dir / "clean"
            clean_dir.mkdir(exist_ok=True)
            (clean_dir / f"{sid}.txt").write_text(markdown[:150000], encoding="utf-8", errors="ignore")

            # Extract images: from markdown body + og:image as the lead candidate
            images = self._images_from_markdown(markdown, final_url or url)
            og = meta.get("ogImage") or meta.get("og:image")
            if isinstance(og, str) and og:
                lead = self._filter_image_urls([{"url": og, "alt": title}], final_url or url)
                if lead:
                    images = lead + [im for im in images if im["url"] != lead[0]["url"]]

            # Try to extract a publication date from metadata or the markdown body.
            # Some Firecrawl responses return list-valued meta (multiple og: tags
            # in the same page), so normalise to a single string.
            def _first_str(v):
                if isinstance(v, (list, tuple)):
                    v = next((x for x in v if x), "")
                return str(v) if v else ""
            published = (_first_str(meta.get("article:published_time"))
                         or _first_str(meta.get("og:article:published_time"))
                         or _first_str(meta.get("publishedTime"))
                         or _first_str(meta.get("date"))
                         or "")
            if not published:
                # Heuristic: scan the first 2 KB of the body for a date pattern
                head = markdown[:2000]
                for pat in _DATE_PATTERNS:
                    m = pat.search(head)
                    if m:
                        published = m.group(0)
                        break

            return {
                "source_id": sid,
                "url": url,
                "canonical_url": final_url,
                "domain": urlparse(url).netloc if url else "",
                "title": title,
                "content_path": f"clean/{sid}.txt",
                "images": images,
                "published_at": published,
            }

    async def _fetch_bs4(self, url: str, final_url: str, sid: str, run_dir: Path, tags: list[str] = None) -> dict[str, Any] | None:
        """Fallback scraping with BeautifulSoup."""
        try:
            from bs4 import BeautifulSoup
            timeout = httpx.Timeout(self.config.web_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
                if resp.status_code >= 400: return None

                soup = BeautifulSoup(resp.text, "html.parser")

                for t in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    t.decompose()

                text = soup.get_text(separator="\n")
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)

                title = soup.title.string.strip() if soup.title else ""

                # Re-parse the original response to keep <img> tags (we decomposed scripts above on `soup`).
                soup_full = BeautifulSoup(resp.text, "html.parser")
                items = []
                # og:image first (often the hero)
                for meta_tag in soup_full.find_all("meta"):
                    prop = (meta_tag.get("property") or meta_tag.get("name") or "").lower()
                    if prop in ("og:image", "twitter:image"):
                        items.append({"url": meta_tag.get("content", ""), "alt": title})
                for img in soup_full.find_all("img"):
                    items.append({"url": img.get("src", ""), "alt": img.get("alt", "")})
                images = self._filter_image_urls(items, final_url or url)

                if tags and not self._filter_by_tags(f"{title} {text[:5000]}", tags):
                    return None

                clean_dir = run_dir / "clean"
                clean_dir.mkdir(exist_ok=True)
                (clean_dir / f"{sid}.txt").write_text(text[:150000], encoding="utf-8", errors="ignore")

                return {
                    "source_id": sid,
                    "url": url,
                    "canonical_url": final_url,
                    "domain": getattr(resp.url, "host", ""),
                    "title": title,
                    "content_path": f"clean/{sid}.txt",
                    "images": images,
                }
        except Exception:
            return None