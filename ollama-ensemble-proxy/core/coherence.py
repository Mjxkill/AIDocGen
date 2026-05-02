"""
Incremental coherence checker for claims.

Embeds each claim and cross-checks against existing claims to detect
contradictions and confirmations using cosine similarity.
"""

import math
import httpx
import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StoredClaim:
    claim_id: str
    source_id: str
    text: str
    embedding: list[float]
    confidence: float = 0.5
    confirmations: int = 0
    disputes: int = 0


class CoherenceIndex:
    """In-memory vector index for claim coherence checking."""

    def __init__(self, ollama_url: str = "http://127.0.0.1:11434", embed_model: str = "mxbai-embed-large", similarity_threshold: float = 0.82, contradiction_keywords: list[str] | None = None):
        self.ollama_url = ollama_url.rstrip("/")
        self.embed_model = embed_model
        self.similarity_threshold = similarity_threshold
        self.claims: list[StoredClaim] = []
        self._dim = 0
        self.conflicts: list[dict[str, Any]] = []
        self.confirmations: list[dict[str, Any]] = []

    async def embed(self, text: str) -> list[float]:
        """Get embedding vector from Ollama."""
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(f"{self.ollama_url}/api/embed", json={
                "model": self.embed_model,
                "input": text,
            })
            resp.raise_for_status()
            embeddings = resp.json().get("embeddings", [[]])
            vec = embeddings[0] if embeddings else []
            if vec:
                self._dim = len(vec)
            return vec

    @staticmethod
    def cosine_sim(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def find_similar(self, embedding: list[float], top_k: int = 5) -> list[tuple[float, StoredClaim]]:
        """Find most similar existing claims."""
        scored = []
        for sc in self.claims:
            sim = self.cosine_sim(embedding, sc.embedding)
            if sim >= self.similarity_threshold:
                scored.append((sim, sc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def _detect_contradiction(self, text_a: str, text_b: str) -> bool:
        """Simple heuristic to detect if two similar claims contradict each other.

        Two claims are potentially contradictory if they talk about the same thing
        (high similarity) but contain different numbers/values.
        """
        import re
        # Extract numbers from both
        nums_a = set(re.findall(r'\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b', text_a))
        nums_b = set(re.findall(r'\b(?:0x[0-9a-fA-F]+|\d+(?:\.\d+)?)\b', text_b))

        if not nums_a or not nums_b:
            return False

        # If they share some numbers but differ on others, likely contradiction
        shared = nums_a & nums_b
        diff_a = nums_a - nums_b
        diff_b = nums_b - nums_a

        # If there are differing numbers and at least some topic overlap, it's suspicious
        if (diff_a or diff_b) and len(shared) < len(nums_a | nums_b) * 0.5:
            return True

        return False

    async def check_and_add(self, claim: dict[str, Any]) -> dict[str, Any]:
        """Check a new claim against existing ones and add it to the index.

        Returns enriched claim with coherence info:
          - coherence_status: "new", "confirmed", "disputed"
          - similar_claims: list of similar claim_ids
          - coherence_score: 0.0-1.0
        """
        text = claim.get("claim_text", "")
        if not text or len(text) < 10:
            return {**claim, "coherence_status": "skipped", "coherence_score": 0.5}

        try:
            embedding = await self.embed(text)
        except Exception:
            return {**claim, "coherence_status": "skipped", "coherence_score": 0.5}

        if not embedding:
            return {**claim, "coherence_status": "skipped", "coherence_score": 0.5}

        similar = self.find_similar(embedding)

        status = "new"
        similar_ids = []
        coherence_score = 0.5

        if similar:
            for sim_score, existing in similar:
                similar_ids.append(existing.claim_id)

                if self._detect_contradiction(text, existing.text):
                    # Contradiction detected
                    status = "disputed"
                    existing.disputes += 1
                    self.conflicts.append({
                        "new_claim": claim.get("claim_id"),
                        "existing_claim": existing.claim_id,
                        "new_text": text[:200],
                        "existing_text": existing.text[:200],
                        "similarity": round(sim_score, 3),
                        "new_source": claim.get("source_id"),
                        "existing_source": existing.source_id,
                    })
                else:
                    # Confirmation
                    if status != "disputed":
                        status = "confirmed"
                    existing.confirmations += 1
                    existing.confidence = min(1.0, existing.confidence + 0.1)
                    self.confirmations.append({
                        "new_claim": claim.get("claim_id"),
                        "existing_claim": existing.claim_id,
                        "similarity": round(sim_score, 3),
                    })

            if status == "confirmed":
                coherence_score = min(1.0, 0.6 + 0.1 * len(similar))
            elif status == "disputed":
                coherence_score = max(0.1, 0.5 - 0.1 * sum(1 for s, e in similar if self._detect_contradiction(text, e.text)))

        # Store in index
        self.claims.append(StoredClaim(
            claim_id=claim.get("claim_id", ""),
            source_id=claim.get("source_id", ""),
            text=text,
            embedding=embedding,
            confidence=claim.get("confidence", 0.5),
        ))

        return {
            **claim,
            "coherence_status": status,
            "coherence_score": round(coherence_score, 3),
            "similar_claims": similar_ids,
        }

    def _claim_to_dict(self, c: StoredClaim) -> dict[str, Any]:
        return {
            "claim_id": c.claim_id,
            "source_id": c.source_id,
            "text": c.text[:200],
            "confirmations": c.confirmations,
            "disputes": c.disputes,
            "confidence": round(c.confidence, 3),
        }

    def get_summary(self) -> dict[str, Any]:
        """Return coherence check summary (pure dicts, JSON-safe)."""
        return {
            "total_claims": len(self.claims),
            "conflicts_found": len(self.conflicts),
            "confirmations_found": len(self.confirmations),
            "top_conflicts": self.conflicts[:20],
            "most_confirmed": [
                self._claim_to_dict(c) for c in sorted(
                    [c for c in self.claims if c.confirmations > 0],
                    key=lambda x: x.confirmations, reverse=True
                )[:10]
            ],
            "most_disputed": [
                self._claim_to_dict(c) for c in sorted(
                    [c for c in self.claims if c.disputes > 0],
                    key=lambda x: x.disputes, reverse=True
                )[:10]
            ],
        }
