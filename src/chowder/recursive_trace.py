from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .database import connect_database


SCHEMA = """
CREATE TABLE IF NOT EXISTS recursive_repair_sessions (
    session_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    policy_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    initial_candidate_ids_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    stop_reason TEXT,
    stop_detail TEXT
);
CREATE TABLE IF NOT EXISTS recursive_repair_hops (
    session_id TEXT NOT NULL,
    depth INTEGER NOT NULL,
    target_experiment_id TEXT NOT NULL,
    failure_signature TEXT NOT NULL,
    target_score REAL NOT NULL,
    score_improvement REAL,
    remaining_budget_after REAL NOT NULL,
    produced_candidate_ids_json TEXT NOT NULL,
    promoted_experiment_id TEXT,
    PRIMARY KEY(session_id, depth),
    FOREIGN KEY(session_id) REFERENCES recursive_repair_sessions(session_id)
);
"""


class RecursiveRepairTraceStore:
    """Durable event/checkpoint store for bounded recursive repair.

    The trace store and RunRegistry share one SQLite ownership/migration
    contract. Opening either component therefore applies the same application-id,
    WAL, foreign-key, and schema-version guarantees before component tables are
    touched.
    """

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = connect_database(self.path)
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RecursiveRepairTraceStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def begin(
        self,
        *,
        session_id: str,
        policy: Mapping[str, Any],
        metadata: Mapping[str, Any],
        initial_candidate_ids: tuple[str, ...],
        state: Mapping[str, Any],
    ) -> None:
        if not session_id.strip():
            raise ValueError("recursive repair session_id is required")
        if len(initial_candidate_ids) != len(set(initial_candidate_ids)):
            raise ValueError("recursive repair session initial candidate IDs must be unique")
        with self._conn:
            self._conn.execute(
                """INSERT INTO recursive_repair_sessions
                   (session_id, status, policy_json, metadata_json,
                    initial_candidate_ids_json, state_json, stop_reason, stop_detail)
                   VALUES (?, 'running', ?, ?, ?, ?, NULL, NULL)""",
                (
                    session_id,
                    self._json(dict(policy)),
                    self._json(dict(metadata)),
                    self._json(list(initial_candidate_ids)),
                    self._json(dict(state)),
                ),
            )

    def record_hop(
        self,
        *,
        session_id: str,
        depth: int,
        target_experiment_id: str,
        failure_signature: str,
        target_score: float,
        score_improvement: float | None,
        remaining_budget_after: float,
        produced_candidate_ids: tuple[str, ...],
        promoted_experiment_id: str | None,
        state: Mapping[str, Any],
    ) -> None:
        if depth <= 0:
            raise ValueError("recursive repair hop depth must be positive")
        if len(failure_signature) != 64:
            raise ValueError("recursive repair failure signature must be SHA-256")
        if not produced_candidate_ids:
            raise ValueError("recursive repair hop must record produced candidates")
        if len(produced_candidate_ids) != len(set(produced_candidate_ids)):
            raise ValueError("recursive repair hop candidate IDs must be unique")
        with self._conn:
            row = self._conn.execute(
                "SELECT status FROM recursive_repair_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise ValueError("recursive repair session does not exist")
            if row[0] != "running":
                raise ValueError("cannot append a hop to a completed recursive repair session")
            self._conn.execute(
                """INSERT INTO recursive_repair_hops
                   (session_id, depth, target_experiment_id, failure_signature,
                    target_score, score_improvement, remaining_budget_after,
                    produced_candidate_ids_json, promoted_experiment_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    depth,
                    target_experiment_id,
                    failure_signature,
                    target_score,
                    score_improvement,
                    remaining_budget_after,
                    self._json(list(produced_candidate_ids)),
                    promoted_experiment_id,
                ),
            )
            self._conn.execute(
                "UPDATE recursive_repair_sessions SET state_json = ? WHERE session_id = ?",
                (self._json(dict(state)), session_id),
            )

    def finish(
        self,
        *,
        session_id: str,
        stop_reason: str,
        stop_detail: str,
        state: Mapping[str, Any],
    ) -> None:
        if not stop_reason.strip():
            raise ValueError("recursive repair stop_reason is required")
        with self._conn:
            cursor = self._conn.execute(
                """UPDATE recursive_repair_sessions
                   SET status = 'completed', state_json = ?, stop_reason = ?, stop_detail = ?
                   WHERE session_id = ? AND status = 'running'""",
                (
                    self._json(dict(state)),
                    stop_reason,
                    stop_detail,
                    session_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("recursive repair session is missing or already terminal")

    def fail(
        self,
        *,
        session_id: str,
        error_detail: str,
        state: Mapping[str, Any],
    ) -> None:
        with self._conn:
            cursor = self._conn.execute(
                """UPDATE recursive_repair_sessions
                   SET status = 'failed', state_json = ?, stop_reason = 'error', stop_detail = ?
                   WHERE session_id = ? AND status = 'running'""",
                (self._json(dict(state)), error_detail, session_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("recursive repair session is missing or already terminal")

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            """SELECT status, policy_json, metadata_json, initial_candidate_ids_json,
                      state_json, stop_reason, stop_detail
               FROM recursive_repair_sessions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        (
            status,
            policy_json,
            metadata_json,
            initial_ids_json,
            state_json,
            stop_reason,
            stop_detail,
        ) = row
        return {
            "session_id": session_id,
            "status": status,
            "policy": json.loads(policy_json),
            "metadata": json.loads(metadata_json),
            "initial_candidate_ids": tuple(json.loads(initial_ids_json)),
            "state": json.loads(state_json),
            "stop_reason": stop_reason,
            "stop_detail": stop_detail,
        }

    def list_hops(self, session_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            """SELECT depth, target_experiment_id, failure_signature, target_score,
                      score_improvement, remaining_budget_after,
                      produced_candidate_ids_json, promoted_experiment_id
               FROM recursive_repair_hops
               WHERE session_id = ? ORDER BY depth""",
            (session_id,),
        )
        return tuple(
            {
                "session_id": session_id,
                "depth": depth,
                "target_experiment_id": target_experiment_id,
                "failure_signature": failure_signature,
                "target_score": target_score,
                "score_improvement": score_improvement,
                "remaining_budget_after": remaining_budget_after,
                "produced_candidate_ids": tuple(json.loads(produced_ids_json)),
                "promoted_experiment_id": promoted_experiment_id,
            }
            for (
                depth,
                target_experiment_id,
                failure_signature,
                target_score,
                score_improvement,
                remaining_budget_after,
                produced_ids_json,
                promoted_experiment_id,
            ) in rows
        )
