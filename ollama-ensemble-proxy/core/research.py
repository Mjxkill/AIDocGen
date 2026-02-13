import asyncio
import hashlib
import time
import re
from pathlib import Path
from typing import Any, Callable
import httpx
from duckduckgo_search import DDGS

from .config import DossierConfig
from .utils import canonicalize_url, emit_progress

class WebResearcher:
    def __init__(self, config: DossierConfig):
        self.config = config

    async def presearch(self, question: str) -> list[dict[str, str]]:
        queries = [question]
        # Generate variants logic could be moved here if complex
        results = []
        try:
            with DDGS() as ddgs:
                for q in queries:
                    for res in ddgs.text(q, max_results=5):
                        results.append({
                            "title": res.get("title", ""),
                            "url": res.get("href", ""),
                            "snippet": res.get("body", ""),
                        })
        except Exception:
            pass
        return results

    async def search_subquestions(
        self,
        planner: dict[str, Any],
        progress_cb: Callable | None,
        run_dir: Path | None,
    ) -> dict[str, Any]:
        all_searches = []
        sub_questions = planner.get("sub_questions", [])
        if not sub_questions:
            # Fallback if no sub_questions, use the reformulated question
            sub_questions = [{"id": "SQ1", "question": planner.get("question_reformulated", "Dossier")}]

        total_sq = len(sub_questions)
        
        for idx, sq in enumerate(sub_questions, 1):
            sq_id = sq.get("id")
            query = sq.get("question")
            
            if run_dir:
                await emit_progress(progress_cb, run_dir, "search", f"Searching {idx}/{total_sq}: {query[:50]}...")

            links = []
            # Try DuckDuckGo
            try:
                await asyncio.sleep(self.config.web_request_delay)
                with DDGS() as ddgs:
                    for res in ddgs.text(query, max_results=self.config.web_per_query_results):
                        links.append({
                            "title": res.get("title"),
                            "url": res.get("href"),
                            "snippet": res.get("body"),
                            "engine": "ddg"
                        })
            except Exception as e:
                print(f"DDG Search error for {sq_id}: {e}")

            # Fallback to SearxNG if DDG failed or returned nothing
            if not links and self.config.searxng_base_url:
                try:
                    searx_links = await self._search_searxng(query)
                    links.extend(searx_links)
                except Exception as e:
                    print(f"SearxNG Search error for {sq_id}: {e}")

            # Final fallback to Wikipedia
            if not links:
                try:
                    wiki_links = await self._search_wikipedia(query)
                    links.extend(wiki_links)
                except Exception:
                    pass

            all_searches.append({
                "id": sq_id,
                "query": query,
                "links": links
            })
            
        return {"sub_questions": all_searches}

    async def _search_searxng(self, query: str) -> list[dict[str, str]]:
        url = f"{self.config.searxng_base_url.rstrip('/')}/search"
        params = {
            "q": query,
            "format": "json",
            "categories": "general",
            "language": "fr-FR"
        }
        links = []
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for res in data.get("results", [])[:self.config.web_per_query_results]:
                        links.append({
                            "title": res.get("title"),
                            "url": res.get("url"),
                            "snippet": res.get("content"),
                            "engine": "searxng"
                        })
        except Exception as e:
            print(f"SearxNG API error: {e}")
        return links

    async def _search_wikipedia(self, query: str) -> list[dict[str, str]]:
        # Search Wikipedia FR
        api_url = "https://fr.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3
        }
        links = []
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(api_url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    for res in data.get("query", {}).get("search", []):
                        page_id = res.get("pageid")
                        title = res.get("title")
                        links.append({
                            "title": title,
                            "url": f"https://fr.wikipedia.org/wiki/{title.replace(' ', '_')}",
                            "snippet": res.get("snippet"),
                            "engine": "wikipedia"
                        })
        except Exception:
            pass
        return links

    async def build_corpus(
        self,
        search_results: dict[str, Any],
        progress_cb: Callable | None,
        run_dir: Path,
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

        fetched_sources = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_fetch)
        processed = 0

        async def fetch_one(link: dict[str, str]):
            nonlocal processed
            async with semaphore:
                source = await self._fetch_url(link["url"], run_dir)
                processed += 1
                if processed % 10 == 0 and run_dir:
                    await emit_progress(progress_cb, run_dir, "corpus", f"Processed {processed}/{len(urls_to_fetch)}")
                return source

        results = await asyncio.gather(*(fetch_one(l) for l in urls_to_fetch))
        fetched_sources = [r for r in results if r]

        return {
            "sources": fetched_sources,
            "count": len(fetched_sources),
            "generated_at": int(time.time())
        }

    async def _fetch_url(self, url: str, run_dir: Path) -> dict[str, Any] | None:
        try:
            from bs4 import BeautifulSoup
            timeout = httpx.Timeout(self.config.web_timeout_seconds)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"})
                if resp.status_code >= 400: return None
                
                final_url = canonicalize_url(str(resp.url))
                sid = f"SRC-{hashlib.sha1(final_url.encode()).hexdigest()[:12]}"
                
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Remove junk
                for tags in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    tags.decompose()
                
                # Extract text
                text = soup.get_text(separator="\n")
                # Clean whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                title = soup.title.string.strip() if soup.title else ""
                
                # Save clean text
                clean_dir = run_dir / "clean"
                clean_dir.mkdir(exist_ok=True)
                (clean_dir / f"{sid}.txt").write_text(text[:150000], encoding="utf-8", errors="ignore")
                
                return {
                    "source_id": sid,
                    "url": url,
                    "canonical_url": final_url,
                    "domain": getattr(resp.url, "host", ""),
                    "title": title,
                    "content_path": f"clean/{sid}.txt"
                }
        except Exception:
            return None
