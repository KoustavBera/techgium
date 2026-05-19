"""
SQLite-backed persistence for patients, screenings, and reports.

The application stores a JSON payload for each screening so report generation
and chat context can survive process restarts without requiring Redis.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.extraction.base import PhysiologicalSystem
from app.core.inference.risk_engine import RiskScore, SystemRiskResult, TrustedRiskResult

logger = logging.getLogger(__name__)


def _normalize_json(value: Any) -> Any:
    """Recursively normalize Python objects into JSON-safe values."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return _normalize_json(asdict(value))
    if isinstance(value, dict):
        return {str(_normalize_json(key)): _normalize_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_json(item) for item in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _normalize_json(value.to_dict())
    return value


class PersistenceService:
    """Repository for durable screening/report storage (supports SQLite & Postgres)."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.is_postgres = database_url.startswith("postgresql")
        self._lock = threading.Lock()
        
        if not self.is_postgres:
            self.db_path = self._resolve_db_path(database_url)
        else:
            self.db_path = None

    def _resolve_db_path(self, database_url: str) -> Path:
        if not database_url.startswith("sqlite:///"):
            return Path("health_screening.db") # fallback

        raw_path = database_url[len("sqlite:///"):]
        path = Path(raw_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    def _connect(self) -> Any:
        if self.is_postgres:
            import psycopg2
            from urllib.parse import urlparse, unquote
            # Parse the URL properly so %-encoded chars (like %40 for @) in
            # the password are decoded before being passed to psycopg2.
            parsed = urlparse(self.database_url)
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=unquote(parsed.username or ""),
                password=unquote(parsed.password or ""),
                dbname=parsed.path.lstrip("/"),
                sslmode="require",  # Supabase requires SSL
            )
            conn.autocommit = True
            return conn
        else:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

    @contextmanager
    def _get_conn(self):
        """
        Context manager that yields an open connection and guarantees cleanup.

        psycopg2 connections don't support `with conn:` as a plain context
        manager for connection lifecycle — only for transaction management.
        This wrapper handles both DB types uniformly.
        """
        conn = self._connect()
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _execute(self, conn: Any, sql: str, params: tuple = ()) -> Any:
        """Execute SQL with correct placeholder and cursor for the database type."""
        if self.is_postgres:
            # Convert SQLite '?' placeholders to Postgres '%s'
            sql = sql.replace("?", "%s")
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute(sql, params)
            return cur
        else:
            return conn.execute(sql, params)

    def init_db(self) -> None:
        if not self.is_postgres:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
        with self._get_conn() as conn:
            schema = """
                CREATE TABLE IF NOT EXISTS patients (
                    patient_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS screenings (
                    screening_id TEXT PRIMARY KEY,
                    patient_id TEXT NOT NULL,
                    screened_at TEXT NOT NULL,
                    overall_risk_level TEXT NOT NULL,
                    overall_risk_score REAL NOT NULL,
                    overall_confidence REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'completed',
                    screening_payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                );

                CREATE TABLE IF NOT EXISTS reports (
                    report_id TEXT PRIMARY KEY,
                    screening_id TEXT NOT NULL UNIQUE,
                    patient_id TEXT NOT NULL,
                    pdf_path TEXT NOT NULL,
                    report_summary_text TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (screening_id) REFERENCES screenings(screening_id),
                    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
                );
            """
            
            if self.is_postgres:
                cur = conn.cursor()
                cur.execute(schema)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient_screened_at ON screenings(patient_id, screened_at DESC);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_reports_patient_generated_at ON reports(patient_id, generated_at DESC);")
            else:
                conn.executescript("PRAGMA foreign_keys = ON;" + schema)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_screenings_patient_screened_at ON screenings(patient_id, screened_at DESC);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_reports_patient_generated_at ON reports(patient_id, generated_at DESC);")
                conn.commit()

        logger.info(f"Persistence service ready ({'PostgreSQL' if self.is_postgres else 'SQLite'})")

    def create_or_get_patient(self, patient_id: str) -> None:
        now = datetime.now().isoformat()
        with self._lock, self._get_conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO patients (patient_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (patient_id, now, now),
            )
            if not self.is_postgres:
                conn.commit()

    def save_screening(self, result: Dict[str, Any], extra_payload: Optional[Dict[str, Any]] = None) -> None:
        self.create_or_get_patient(result["patient_id"])
        extra_payload = extra_payload or {}

        payload = {
            "screening_id": result["screening_id"],
            "patient_id": result["patient_id"],
            "timestamp": result["timestamp"],
            "overall_risk_level": result["overall_risk_level"],
            "overall_risk_score": result["overall_risk_score"],
            "overall_confidence": result["overall_confidence"],
            "system_results": result.get("system_results", []),
            "system_results_internal": result.get("system_results_internal", {}),
            "trusted_results": result.get("trusted_results", {}),
            "composite_risk": result.get("composite_risk"),
            "rejected_systems": result.get("rejected_systems", []),
            "trust_metadata": result.get("trust_metadata"),
            "validation_status": result.get("validation_status"),
            "requires_review": result.get("requires_review", False),
            "clinical_findings": extra_payload.get("clinical_findings", []),
            "clinical_summary": extra_payload.get("clinical_summary", {}),
        }
        payload_json = json.dumps(_normalize_json(payload))
        timestamp = result["timestamp"].isoformat()
        created_at = datetime.now().isoformat()

        with self._lock, self._get_conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO screenings (
                    screening_id,
                    patient_id,
                    screened_at,
                    overall_risk_level,
                    overall_risk_score,
                    overall_confidence,
                    status,
                    screening_payload_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(screening_id) DO UPDATE SET
                    patient_id=excluded.patient_id,
                    screened_at=excluded.screened_at,
                    overall_risk_level=excluded.overall_risk_level,
                    overall_risk_score=excluded.overall_risk_score,
                    overall_confidence=excluded.overall_confidence,
                    status=excluded.status,
                    screening_payload_json=excluded.screening_payload_json
                """,
                (
                    result["screening_id"],
                    result["patient_id"],
                    timestamp,
                    result["overall_risk_level"],
                    float(result["overall_risk_score"]),
                    float(result["overall_confidence"]),
                    "completed",
                    payload_json,
                    created_at,
                ),
            )
            if not self.is_postgres:
                conn.commit()

    def get_screening_by_id(self, screening_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT screening_id, patient_id, screened_at, screening_payload_json
                FROM screenings
                WHERE screening_id = ?
                """,
                (screening_id,),
            ).fetchone()

        if row is None:
            return None

        payload = json.loads(row["screening_payload_json"])
        return {
            "screening_id": row["screening_id"],
            "patient_id": row["patient_id"],
            "timestamp": datetime.fromisoformat(row["screened_at"]),
            "system_results": self._deserialize_system_results(payload.get("system_results_internal", {})),
            "trusted_results": self._deserialize_trusted_results(payload.get("trusted_results", {})),
            "composite_risk": self._deserialize_risk_score(payload.get("composite_risk")),
            "rejected_systems": payload.get("rejected_systems", []),
            "response_system_results": payload.get("system_results", []),
            "clinical_findings": payload.get("clinical_findings", []),
            "clinical_summary": payload.get("clinical_summary", {}),
            "overall_risk_level": payload.get("overall_risk_level"),
            "overall_risk_score": payload.get("overall_risk_score", 0.0),
            "overall_confidence": payload.get("overall_confidence", 0.0),
            "validation_status": payload.get("validation_status"),
            "requires_review": payload.get("requires_review", False),
            "screening_payload_json": payload,
        }

    def get_latest_screening_by_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT screening_id
                FROM screenings
                WHERE lower(patient_id) = lower(?)
                ORDER BY screened_at DESC
                LIMIT 1
                """,
                (patient_id,),
            ).fetchone()

        if row is None:
            return None
        return self.get_screening_by_id(row["screening_id"])

    def save_report(
        self,
        screening_id: str,
        patient_id: str,
        report_id: str,
        pdf_path: str,
        generated_at: datetime,
        report_summary_text: str,
    ) -> None:
        self.create_or_get_patient(patient_id)
        created_at = datetime.now().isoformat()

        with self._lock, self._get_conn() as conn:
            self._execute(
                conn,
                """
                INSERT INTO reports (
                    report_id,
                    screening_id,
                    patient_id,
                    pdf_path,
                    report_summary_text,
                    generated_at,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(screening_id) DO UPDATE SET
                    report_id=excluded.report_id,
                    patient_id=excluded.patient_id,
                    pdf_path=excluded.pdf_path,
                    report_summary_text=excluded.report_summary_text,
                    generated_at=excluded.generated_at
                """,
                (
                    report_id,
                    screening_id,
                    patient_id,
                    pdf_path,
                    report_summary_text,
                    generated_at.isoformat(),
                    created_at,
                ),
            )
            if not self.is_postgres:
                conn.commit()

    def get_report_by_id(self, report_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT report_id, screening_id, patient_id, pdf_path, report_summary_text, generated_at
                FROM reports
                WHERE report_id = ?
                """,
                (report_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "report_id": row["report_id"],
            "screening_id": row["screening_id"],
            "patient_id": row["patient_id"],
            "pdf_path": row["pdf_path"],
            "report_summary_text": row["report_summary_text"],
            "generated_at": row["generated_at"],
        }

    def get_latest_report_by_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        with self._get_conn() as conn:
            row = self._execute(
                conn,
                """
                SELECT r.report_id, r.screening_id, r.patient_id, r.pdf_path, r.report_summary_text, r.generated_at
                FROM reports r
                JOIN screenings s ON s.screening_id = r.screening_id
                WHERE lower(r.patient_id) = lower(?)
                ORDER BY s.screened_at DESC
                LIMIT 1
                """,
                (patient_id,),
            ).fetchone()

        if row is None:
            return None

        return {
            "report_id": row["report_id"],
            "screening_id": row["screening_id"],
            "patient_id": row["patient_id"],
            "pdf_path": row["pdf_path"],
            "report_summary_text": row["report_summary_text"],
            "generated_at": row["generated_at"],
        }

    def get_patient_report_context(self, patient_id: str) -> str:
        report = self.get_latest_report_by_patient(patient_id)
        if not report:
            return ""
        return report.get("report_summary_text") or ""

    def _deserialize_risk_score(self, data: Optional[Dict[str, Any]]) -> Optional[RiskScore]:
        if not data:
            return None
        return RiskScore(
            name=data.get("name", "overall"),
            score=float(data.get("score", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            contributing_biomarkers=list(data.get("contributing_biomarkers", [])),
            explanation=data.get("explanation", ""),
        )

    def _deserialize_system_results(self, data: Dict[str, Dict[str, Any]]) -> Dict[PhysiologicalSystem, SystemRiskResult]:
        results: Dict[PhysiologicalSystem, SystemRiskResult] = {}
        for system_name, item in data.items():
            system = PhysiologicalSystem.from_string(system_name)
            results[system] = SystemRiskResult(
                system=system,
                overall_risk=self._deserialize_risk_score(item.get("overall_risk")) or RiskScore(
                    name=f"{system.value}_overall",
                    score=0.0,
                    confidence=0.0,
                ),
                sub_risks=[
                    self._deserialize_risk_score(sub_risk)
                    for sub_risk in item.get("sub_risks", [])
                    if sub_risk is not None
                ],
                biomarker_summary=item.get("biomarker_summary", {}),
                alerts=item.get("alerts", []),
            )
        return results

    def _deserialize_trusted_results(self, data: Dict[str, Dict[str, Any]]) -> Dict[PhysiologicalSystem, TrustedRiskResult]:
        results: Dict[PhysiologicalSystem, TrustedRiskResult] = {}
        for system_name, item in data.items():
            system = PhysiologicalSystem.from_string(system_name)
            risk_result = item.get("risk_result")
            results[system] = TrustedRiskResult(
                risk_result=(
                    SystemRiskResult(
                        system=system,
                        overall_risk=self._deserialize_risk_score(risk_result.get("overall_risk")) or RiskScore(
                            name=f"{system.value}_overall",
                            score=0.0,
                            confidence=0.0,
                        ),
                        sub_risks=[
                            self._deserialize_risk_score(sub_risk)
                            for sub_risk in risk_result.get("sub_risks", [])
                            if sub_risk is not None
                        ],
                        biomarker_summary=risk_result.get("biomarker_summary", {}),
                        alerts=risk_result.get("alerts", []),
                    )
                    if risk_result else None
                ),
                is_trusted=bool(item.get("is_trusted", True)),
                was_rejected=bool(item.get("was_rejected", False)),
                rejection_reason=item.get("rejection_reason", ""),
                trust_adjusted_confidence=float(item.get("trust_adjusted_confidence", 1.0)),
                caveats=list(item.get("caveats", [])),
            )
        return results


def build_report_summary_text(report: Any, screening_id: str) -> str:
    """Build the single persisted chatbot context from the canonical report."""
    lines: List[str] = [
        f"## Patient Screening Report - {report.generated_at.strftime('%d %b %Y, %I:%M %p')}",
        f"- Patient ID: {report.patient_id}",
        f"- Screening ID: {screening_id}",
        f"- Report ID: {report.report_id}",
        f"- Overall Risk: {report.overall_risk_level.value.upper()} "
        f"(score {report.overall_risk_score:.1f}/100, confidence {report.overall_confidence:.0%})",
    ]

    if getattr(report, "interpretation_summary", ""):
        lines.extend(["", "### Summary", report.interpretation_summary.strip()])

    system_summaries = getattr(report, "system_summaries", {}) or {}
    if system_summaries:
        lines.extend(["", "### System Highlights"])
        for system, details in system_summaries.items():
            system_name = system.value if hasattr(system, "value") else str(system)
            risk_level = details.get("risk_level")
            risk_label = risk_level.value.upper() if hasattr(risk_level, "value") else str(risk_level).upper()
            lines.append(f"- {system_name.replace('_', ' ').title()}: {risk_label}")
            for alert in details.get("alerts", [])[:2]:
                lines.append(f"  - {alert}")

    recommendations = getattr(report, "recommendations", []) or []
    if recommendations:
        lines.extend(["", "### Recommendations"])
        for rec in recommendations[:5]:
            lines.append(f"- {rec}")

    caveats = getattr(report, "caveats", []) or []
    if caveats:
        lines.extend(["", "### Caveats"])
        for caveat in caveats[:3]:
            lines.append(f"- {caveat}")

    return "\n".join(lines)
