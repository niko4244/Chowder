"""Experiment E retrieval corpus, provenance, and subsystems (Phase 3).

The seed corpus is intentionally small and hand-authored. To grow it, ingest
only independently verified source records through ``ingest_verified_documents``;
that function refuses missing provenance, duplicate ids, empty documents, and
source bytes whose declared SHA-256 does not match. It never fabricates or
clones documents to meet a scale target.

Sparse-memory fitting defaults to legacy all-document self-similarity only when
no train-query split is supplied. For evaluation, declare ``train_query_ids``
and ``holdout_query_ids`` disjointly; only training query rows enter the sparse
objective, while held-out queries are scored separately against dense/BM25/
sparse results.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from rank_bm25 import BM25Okapi

sys_path = str(Path(__file__).resolve().parents[1] / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

RETRIEVAL_METHODS = ("none", "bm25", "dense", "sparse_learned")


def _spark_card() -> dict:
    return {
        "doc_id": "spark_model_card", "title": "Spark model card",
        "text": (
            "The Spark X2.5 4B model is a small decoder-only language model with "
            "131072 vocabulary entries and 4 billion parameters. It runs with a "
            "custom remote-code architecture. The Spark tokenizer is not "
            "compatible with the Qwen tokenizer family."
        ),
        "source_uri": "local model config evidence (historical seed)",
        "source_sha256": "seed-manual-verification",
    }


def _teacher_card() -> dict:
    return {
        "doc_id": "teacher_model_card", "title": "Teacher model card",
        "text": (
            "The Qwen3.8-9B teacher is a decoder-only transformer with a "
            "vocabulary size of 248320 tokens, hidden size 4096, 32 decoder "
            "layers, and approximately 9 billion parameters. The teacher "
            "checkpoint is bf16 and its lm_head is not tied to the embedding."
        ),
        "source_uri": "local model config evidence (historical seed)",
        "source_sha256": "seed-manual-verification",
    }


def build_corpus() -> list[dict]:
    """Return the seven historical seed documents, not a scale-expanded corpus."""
    docs = [
        _spark_card(),
        _teacher_card(),
        {
            "doc_id": "runtime_version_doc", "title": "version.py protocol",
            "text": "The version.py runtime task requires parse_version to return three parts, padding with zeros when fewer are provided. The test suite reports 2 passed when the fix is correct.",
            "source_uri": "historical deterministic Experiment E task notes",
            "source_sha256": "seed-manual-verification",
        },
        {
            "doc_id": "runtime_slug_doc", "title": "slugify protocol",
            "text": "The slugify runtime task requires strip and lowercase before joining words with hyphens. A correct fix reports 2 passed.",
            "source_uri": "historical deterministic Experiment E task notes",
            "source_sha256": "seed-manual-verification",
        },
        {
            "doc_id": "runtime_harness_doc", "title": "runtime harness tools",
            "text": "The runtime repair harness exposes exactly three tools: read_file, write_file, and run_tests. A green test run is the exact string '2 passed'. Failed runs report '1 failed, 1 passed' or similar. Invalid reads are nonexistent path reads and are counted.",
            "source_uri": "historical chowder.runtime_eval.py seed",
            "source_sha256": "seed-manual-verification",
        },
        {
            "doc_id": "campaign_doc", "title": "Campaign notes",
            "text": "The chowder campaign follows the RRSI paper for harness evolution regularizers. RRSI stands for Regularized Recursive Self-Improvement of Agent Harnesses. Runtime promotion gates require green rate and nonexistent-read rate targets.",
            "source_uri": "historical campaign notes",
            "source_sha256": "seed-manual-verification",
        },
        {
            "doc_id": "arith_notes", "title": "Arithmetic notes",
            "text": "Arithmetic notes: 13 times 24 equals 312. 47 plus 86 equals 133. 1024 divided by 8 equals 128. Going from 19.99 to 24.99 is a 25 percent increase. These verified products and sums are recorded for retrieval experiments.",
            "source_uri": "historical experiment arithmetic seed",
            "source_sha256": "seed-manual-verification",
        },
    ]
    return docs


def ingest_verified_documents(records: Sequence[Mapping[str, object]]) -> list[dict]:
    """Validate imported source records and return normalized corpus documents.

    Each record includes ``doc_id``, ``title``, ``text``, ``source_uri`` and
    ``source_sha256``. Hash input is the exact UTF-8 bytes of ``text``. If a
    publisher gives a content/file hash instead, store it under ``source_file_sha256``
    and continue to use the text hash as ``source_sha256``.
    """
    docs: list[dict] = []
    seen: set[str] = set()
    seen_text_hashes: set[str] = set()
    for index, raw in enumerate(records):
        doc_id = str(raw.get("doc_id", "")).strip()
        title = str(raw.get("title", "")).strip()
        text = str(raw.get("text", ""))
        uri = str(raw.get("source_uri", "")).strip()
        declared = str(raw.get("source_sha256", "")).strip().lower()
        if not doc_id or not title or not text.strip() or not uri:
            raise ValueError(f"record {index} missing id/title/nonempty text/source_uri")
        if doc_id in seen:
            raise ValueError(f"duplicate document id: {doc_id}")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual in seen_text_hashes:
            raise ValueError(f"duplicate document text: {doc_id}")
        if declared != actual:
            raise ValueError(f"source text hash mismatch for {doc_id}")
        seen.add(doc_id)
        seen_text_hashes.add(actual)
        docs.append({
            "doc_id": doc_id, "title": title, "text": text,
            "source_uri": uri, "source_sha256": actual,
            **({"source_file_sha256": str(raw["source_file_sha256"])} if raw.get("source_file_sha256") else {}),
        })
    return docs


def load_verified_corpus(path: str | Path) -> list[dict]:
    """Load only a JSON array of provenance-verified corpus records."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("documents") if isinstance(payload, Mapping) else payload
    if not isinstance(records, list):
        raise ValueError("verified corpus JSON must be a document list or {documents: [...]} ")
    return ingest_verified_documents(records)


def corpus_manifest(docs: Sequence[Mapping[str, object]]) -> dict:
    """Summarize only records that pass exact-content provenance checks."""
    verified = ingest_verified_documents(docs)
    ids = [doc["doc_id"] for doc in verified]
    return {
        "n_documents": len(verified),
        "document_ids": ids,
        "documents": [
            {
                "doc_id": doc["doc_id"],
                "source_uri": doc["source_uri"],
                "source_sha256": doc["source_sha256"],
            }
            for doc in verified
        ],
        "corpus_sha256": hashlib.sha256(
            "\n".join(f"{doc['doc_id']}:{doc['source_sha256']}" for doc in verified).encode("utf-8")
        ).hexdigest(),
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ollama_embed(texts: list[str], model: str = "nomic-embed-text") -> np.ndarray:
    """Compact dense embeddings via the local Ollama server."""
    vectors = []
    for text in texts:
        req = urllib.request.Request(
            "http://localhost:11434/api/embeddings",
            data=json.dumps({"model": model, "prompt": text}).encode(),
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            vectors.append(np.array(json.loads(response.read())["embedding"], dtype=np.float32))
    return np.vstack(vectors)


class SparseMemoryLayer(torch.nn.Module):
    """Learned bottleneck scoring all corpus slots and masking to top-k."""

    def __init__(self, dim: int, n_docs: int, code_dim: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.encoder = torch.nn.Linear(dim, code_dim, bias=False)
        self.memory = torch.nn.Parameter(torch.randn(n_docs, code_dim) * 0.02)

    def scores(self, query: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(query, torch.Tensor):
            q = query.float()
        else:
            q = torch.from_numpy(np.asarray(query, dtype=np.float32))
        h = torch.tanh(self.encoder(q))
        e = self.memory / self.memory.norm(dim=1, keepdim=True).clamp_min(1e-9)
        return e @ h

    def sparse_scores(self, query: np.ndarray | torch.Tensor, k: int = 2) -> torch.Tensor:
        scores = self.scores(query)
        topk = torch.topk(scores, min(k, scores.shape[0])).indices
        mask = torch.zeros_like(scores)
        mask[topk] = 1.0
        return (scores * mask).detach()

    def train_on(
        self, dense: np.ndarray, *, query_vectors: np.ndarray | None = None,
        query_indices: Iterable[int] | None = None,
        train_doc_indices: Iterable[int] | None = None,
        epochs: int = 400, lr: float = 3e-2,
    ) -> float:
        """Fit train-query similarities against train-document targets only."""
        dense_t = torch.from_numpy(dense.astype(np.float32))
        dense_n = dense_t / dense_t.norm(dim=1, keepdim=True).clamp_min(1e-9)
        doc_indices = list(train_doc_indices) if train_doc_indices is not None else list(range(len(dense)))
        if epochs < 1 or not math.isfinite(float(lr)) or lr <= 0:
            raise ValueError("sparse-memory training requires positive epochs and finite positive lr")
        if not doc_indices or any(index < 0 or index >= len(dense) for index in doc_indices):
            raise ValueError("sparse-memory training requires valid nonempty train document indices")
        if query_vectors is not None:
            queries = torch.from_numpy(query_vectors.astype(np.float32))
        elif query_indices is None:
            queries = dense_t[doc_indices]
        else:
            indices = list(query_indices)
            if not indices or any(index < 0 or index >= len(dense) for index in indices):
                raise ValueError("sparse-memory training requires valid nonempty query indices")
            queries = dense_t[indices]
        if queries.ndim != 2 or queries.shape[1] != dense.shape[1] or not len(queries):
            raise ValueError("sparse-memory query vectors must be a nonempty matrix matching embedding dim")
        if not torch.isfinite(queries).all():
            raise ValueError("sparse-memory query vectors must be finite")
        query_n = queries / queries.norm(dim=1, keepdim=True).clamp_min(1e-9)
        target = query_n @ dense_n[doc_indices].T
        opt = torch.optim.Adam(self.parameters(), lr=float(lr))
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            all_scores = torch.stack([self.scores(row) for row in queries])
            scores = all_scores[:, doc_indices]
            loss = torch.nn.functional.mse_loss(scores, target)
            loss.backward()
            opt.step()
        return float(loss.detach())

    def storage_bytes(self) -> int:
        return int(sum(p.numel() * p.element_size() for p in self.parameters()))


class RetrievalSubsystem:
    """Bundle no-retrieval, BM25, dense, and learned-sparse retrievers."""

    def __init__(
        self, docs: list[dict], *, seed: int = 0,
        sparse_train_queries: Sequence[Mapping[str, str]] | None = None,
        train_doc_ids: Iterable[str] | None = None,
        embed: Callable[[list[str]], np.ndarray] = _ollama_embed,
        sparse_epochs: int = 400,
    ):
        if not docs:
            raise ValueError("retrieval corpus must contain at least one document")
        ids = [str(doc.get("doc_id", "")) for doc in docs]
        if any(not doc_id for doc_id in ids) or len(ids) != len(set(ids)):
            raise ValueError("corpus doc_id values must be nonempty and unique")
        self.docs = docs
        self.texts = [d["title"] + ". " + d["text"] for d in docs]
        self.bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        self.embed = embed
        self.dense = embed(self.texts)
        if self.dense.ndim != 2 or self.dense.shape[0] != len(docs) or self.dense.shape[1] < 1:
            raise ValueError("document embedder must return one nonempty vector per document")
        if not np.isfinite(self.dense).all():
            raise ValueError("document embedder returned non-finite vectors")
        self.sparse_layer = SparseMemoryLayer(self.dense.shape[1], len(docs), seed=seed)
        by_id = {doc_id: index for index, doc_id in enumerate(ids)}
        train_docs = set(train_doc_ids) if train_doc_ids is not None else set(ids)
        unknown_docs = train_docs - set(by_id)
        if unknown_docs or not train_docs:
            raise ValueError(f"invalid sparse train document ids: {sorted(unknown_docs)[:3]}")
        train_doc_indices = [by_id[doc_id] for doc_id in ids if doc_id in train_docs]
        self.sparse_train_query_manifest: dict[str, str] = {}
        if sparse_train_queries is None:
            self.sparse_training_mode = "legacy_document_self_similarity"
            self.sparse_train_query_ids = list(ids)
            self.sparse_train_doc_ids = [doc_id for doc_id in ids if doc_id in train_docs]
            self.sparse_train_loss = self.sparse_layer.train_on(
                self.dense, query_indices=train_doc_indices,
                train_doc_indices=train_doc_indices, epochs=sparse_epochs,
            )
        else:
            self.sparse_training_mode = "train_query_dense_distillation"
            query_ids = [str(query.get("query_id", "")).strip() for query in sparse_train_queries]
            if any(not query_id for query_id in query_ids) or len(query_ids) != len(set(query_ids)):
                raise ValueError("sparse training query ids must be nonempty and unique")
            query_texts = [str(query.get("query", "")) for query in sparse_train_queries]
            if any(not text.strip() for text in query_texts):
                raise ValueError("sparse training queries must contain text")
            for query_id, query, text in zip(query_ids, sparse_train_queries, query_texts):
                relevant = set(filter(None, str(query.get("relevant_doc_ids", "")).split(",")))
                if not relevant or not relevant <= train_docs:
                    raise ValueError(f"sparse training query has missing or cross-split relevant docs: {query_id}")
                canonical = json.dumps({"query": text, "relevant_doc_ids": sorted(relevant)}, sort_keys=True, separators=(",", ":"))
                self.sparse_train_query_manifest[query_id] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            query_vectors = embed(query_texts)
            if query_vectors.shape != (len(query_ids), self.dense.shape[1]) or not np.isfinite(query_vectors).all():
                raise ValueError("sparse training embedder must return one finite vector per query")
            self.sparse_train_query_ids = query_ids
            self.sparse_train_doc_ids = [doc_id for doc_id in ids if doc_id in train_docs]
            self.sparse_train_loss = self.sparse_layer.train_on(
                self.dense, query_vectors=query_vectors,
                train_doc_indices=train_doc_indices, epochs=sparse_epochs,
            )

    def retrieve(self, query: str, *, method: str, k: int = 2) -> tuple[list[dict], float]:
        """Return (top-k docs, lookup latency in ms including query encoding)."""
        if k < 1 and method != "none":
            raise ValueError("retrieval k must be positive")
        t0 = time.perf_counter()
        if method == "none":
            picked = []
        elif method == "bm25":
            scores = self.bm25.get_scores(_tokenize(query))
            picked = [self.docs[i] for i in np.argsort(scores)[::-1][:k]]
        elif method == "dense":
            q = self.embed([query])[0]
            sims = self.dense @ (q / (np.linalg.norm(q) + 1e-9))
            picked = [self.docs[i] for i in np.argsort(sims)[::-1][:k]]
        elif method == "sparse_learned":
            q = self.embed([query])[0]
            masked = self.sparse_layer.sparse_scores(q, k=k)
            picked = [self.docs[i] for i in np.argsort(masked.numpy())[::-1][:k]]
        else:
            raise ValueError(f"unknown retrieval method: {method}")
        latency_ms = (time.perf_counter() - t0) * 1000
        return picked, latency_ms

    def context_block(self, docs: list[dict]) -> str:
        if not docs:
            return ""
        parts = ["[retrieved evidence begins]"]
        for d in docs:
            parts.append(f"[source: {d['doc_id']}] {d['title']}: {d['text']}")
        parts.append("[retrieved evidence ends]")
        return "\n".join(parts)

    def storage_bytes(self, method: str) -> int:
        if method == "dense":
            return int(self.dense.nbytes)
        if method == "sparse_learned":
            return self.sparse_layer.storage_bytes()
        return 0


def evaluate_retrieval(
    subsystem: RetrievalSubsystem,
    queries: Sequence[Mapping[str, str]],
    *,
    train_query_ids: Iterable[str],
    holdout_query_ids: Iterable[str],
    train_doc_ids: Iterable[str],
    holdout_doc_ids: Iterable[str],
    methods: Sequence[str] = ("bm25", "dense", "sparse_learned"),
    k: int = 2,
) -> dict:
    """Compare hit@k on untouched queries whose gold docs were held out of sparse fitting."""
    train_ids = set(train_query_ids)
    held_ids = set(holdout_query_ids)
    train_docs = set(train_doc_ids)
    held_docs = set(holdout_doc_ids)
    if k < 1:
        raise ValueError("retrieval k must be positive")
    if subsystem.sparse_training_mode != "train_query_dense_distillation":
        raise ValueError("retrieval eval requires sparse memory fit on explicit train queries")
    if train_ids & held_ids:
        raise ValueError("retrieval train/holdout query ids must be disjoint")
    if train_docs & held_docs:
        raise ValueError("retrieval train/holdout document ids must be disjoint")
    if set(subsystem.sparse_train_query_ids) != train_ids:
        raise ValueError("sparse memory was not fit on exactly the declared training query ids")
    if set(subsystem.sparse_train_doc_ids) != train_docs:
        raise ValueError("sparse memory was not fit on exactly the declared training document ids")
    if not train_ids or not held_ids or not train_docs:
        raise ValueError("retrieval evaluation requires nonempty train/holdout queries and train documents")
    if not held_docs:
        raise ValueError("retrieval evaluation requires held-out relevant documents")
    if (train_ids | held_ids) - set(str(query.get("query_id", "")) for query in queries):
        raise ValueError("declared query split includes an absent query id")
    if not train_docs <= {doc["doc_id"] for doc in subsystem.docs} or not held_docs <= {doc["doc_id"] for doc in subsystem.docs}:
        raise ValueError("declared document split includes a document absent from corpus")

    by_id = {str(query["query_id"]): query for query in queries}
    if len(by_id) != len(queries):
        raise ValueError("retrieval query ids must be unique")
    if set(by_id) != train_ids | held_ids:
        raise ValueError("query records must exactly match the declared train/holdout split")
    for query_id, query in by_id.items():
        if not str(query.get("query", "")).strip():
            raise ValueError(f"retrieval query text is empty: {query_id}")
        relevant = set(filter(None, str(query.get("relevant_doc_ids", "")).split(",")))
        expected_docs = train_docs if query_id in train_ids else held_docs
        if not relevant or not relevant <= expected_docs:
            raise ValueError(f"query has missing or cross-split gold documents: {query_id}")
        if query_id in train_ids:
            canonical = json.dumps({
                "query": str(query["query"]), "relevant_doc_ids": sorted(relevant),
            }, sort_keys=True, separators=(",", ":"))
            query_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if subsystem.sparse_train_query_manifest.get(query_id) != query_hash:
                raise ValueError(f"sparse training query content/labels differ from declared train row: {query_id}")
    output = {
        "train_query_ids": sorted(train_ids), "holdout_query_ids": sorted(held_ids),
        "train_doc_ids": sorted(train_docs), "holdout_doc_ids": sorted(held_docs), "methods": {},
    }
    for method in methods:
        rows = []
        for query_id in sorted(held_ids):
            query = by_id[query_id]
            picked, latency_ms = subsystem.retrieve(query["query"], method=method, k=k)
            retrieved = [str(doc["doc_id"]) for doc in picked]
            relevant = set(filter(None, query["relevant_doc_ids"].split(",")))
            rows.append({
                "query_id": query_id, "retrieved": retrieved,
                "hit": bool(relevant & set(retrieved)), "latency_ms": latency_ms,
            })
        output["methods"][method] = {
            "hit_at_k": sum(row["hit"] for row in rows) / len(rows),
            "mean_latency_ms": sum(row["latency_ms"] for row in rows) / len(rows),
            "n": len(rows), "queries": rows,
            "storage_bytes": subsystem.storage_bytes(method),
        }
    output["sparse_training_mode"] = subsystem.sparse_training_mode
    output["sparse_train_query_ids"] = subsystem.sparse_train_query_ids
    output["sparse_train_doc_ids"] = subsystem.sparse_train_doc_ids
    return output


def build_verified_markdown_corpus(
    source_root: str | Path, *, min_chars: int = 180, max_chars: int = 1400,
    exclude_names: Iterable[str] = ("HANDOFF.md",),
    exclude_relative_paths: Iterable[str] = (),
) -> list[dict]:
    """Split local Markdown sources into exact, hashed text windows.

    Each output body is a byte-decoded substring formed from contiguous source
    lines; records retain source path, file digest, and line interval. No
    templating, duplication, generated facts, or content rewriting is used.
    """
    root = Path(source_root).resolve()
    if min_chars < 1 or max_chars < min_chars:
        raise ValueError("require 1 <= min_chars <= max_chars")
    excluded = set(exclude_names)
    excluded_paths = {Path(item).as_posix() for item in exclude_relative_paths}
    docs: list[dict] = []
    for path in sorted(root.rglob("*.md")):
        relative = path.relative_to(root)
        if (
            path.name in excluded
            or relative.as_posix() in excluded_paths
            or any(part.startswith(".") for part in relative.parts)
        ):
            continue
        raw = path.read_bytes()
        file_sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8")
        lines = text.splitlines(keepends=True)
        chunks: list[tuple[int, int, str]] = []
        current: list[str] = []
        start = 1
        end = 0
        for line_no, line in enumerate(lines, 1):
            if len(line) > max_chars:
                if current:
                    body = "".join(current)
                    if len(body.strip()) >= min_chars:
                        chunks.append((start, end, body))
                    current = []
                for offset in range(0, len(line), max_chars):
                    fragment = line[offset:offset + max_chars]
                    if len(fragment.strip()) >= min_chars:
                        chunks.append((line_no, line_no, fragment))
                continue
            if current and len("".join(current)) + len(line) > max_chars:
                body = "".join(current)
                if len(body.strip()) >= min_chars:
                    chunks.append((start, end, body))
                current = []
                start = line_no
            if not current:
                start = line_no
            current.append(line)
            end = line_no
        if current:
            body = "".join(current)
            if len(body.strip()) >= min_chars:
                chunks.append((start, end, body))
        seen_chunk_hashes: set[str] = set()
        for index, (first_line, last_line, body) in enumerate(chunks):
            chunk_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
            if chunk_hash in seen_chunk_hashes:
                continue
            seen_chunk_hashes.add(chunk_hash)
            doc_id = f"md:{relative.as_posix()}#L{first_line}-L{last_line}:{index:04d}"
            docs.append({
                "doc_id": doc_id,
                "title": f"{relative.as_posix()} lines {first_line}-{last_line}",
                "text": body,
                "source_uri": f"{relative.as_posix()}#L{first_line}-L{last_line}",
                "source_sha256": chunk_hash,
                "source_file_sha256": file_sha,
            })
    return ingest_verified_documents(docs)


def write_verified_markdown_corpus(
    source_root: str | Path, output_path: str | Path, *, min_documents: int = 300,
    min_chars: int = 180, max_chars: int = 1400,
    exclude_relative_paths: Iterable[str] = (),
) -> dict:
    """Write the corpus only when the configured scale target is truly met."""
    docs = build_verified_markdown_corpus(
        source_root, min_chars=min_chars, max_chars=max_chars,
        exclude_relative_paths=exclude_relative_paths,
    )
    if len(docs) < min_documents:
        raise ValueError(
            f"only {len(docs)} verified nonempty Markdown chunks found; "
            f"need {min_documents}; no corpus artifact written"
        )
    payload = {"manifest": corpus_manifest(docs), "documents": docs}
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-documents", type=int, default=300)
    parser.add_argument("--min-chars", type=int, default=180)
    parser.add_argument("--max-chars", type=int, default=1400)
    args = parser.parse_args()
    payload = write_verified_markdown_corpus(
        args.markdown_root, args.out, min_documents=args.min_documents,
        min_chars=args.min_chars, max_chars=args.max_chars,
    )
    print(json.dumps(payload["manifest"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
