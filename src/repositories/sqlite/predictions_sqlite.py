from __future__ import annotations

import sqlite3
from typing import Optional

from ..predictions import Prediction, PredictionsRepo


class PredictionsRepoSqlite(PredictionsRepo):
    """SQLite implementation of :class:`PredictionsRepo`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY,
                match_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                prob_home REAL NOT NULL CHECK(prob_home BETWEEN 0 AND 1),
                prob_draw REAL NOT NULL CHECK(prob_draw BETWEEN 0 AND 1),
                prob_away REAL NOT NULL CHECK(prob_away BETWEEN 0 AND 1),
                predicted_result TEXT NOT NULL CHECK(predicted_result IN ('1','X','2')),
                is_correct INTEGER CHECK(is_correct IN (0,1)),
                result_status TEXT NOT NULL DEFAULT 'PENDING' CHECK(result_status IN ('WIN','LOSE','PENDING')),
                CHECK (ABS(prob_home + prob_draw + prob_away - 1) <= 1e-6),
                FOREIGN KEY (match_id) REFERENCES matches(match_id)
            )
            """
        )
        # Backfill result_status column if DB pre-exists without it
        try:
            cols = {
                row[1] for row in self._conn.execute("PRAGMA table_info(predictions)").fetchall()
            }
            if "result_status" not in cols:
                self._conn.execute(
                    "ALTER TABLE predictions ADD COLUMN result_status TEXT NOT NULL DEFAULT 'PENDING'"
                )
        except sqlite3.DatabaseError:
            pass
        # Ensure unique constraint on (match_id, model_name) for idempotent upserts
        try:
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_match_model ON predictions(match_id, model_name)"
            )
        except sqlite3.IntegrityError:
            # Deduplicate by keeping the latest prediction per (match_id, model_name)
            self._conn.execute(
                """
                DELETE FROM predictions
                WHERE prediction_id NOT IN (
                    SELECT MAX(prediction_id) FROM predictions GROUP BY match_id, model_name
                )
                """
            )
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_predictions_match_model ON predictions(match_id, model_name)"
            )
        self._conn.commit()

    def get_by_id(self, prediction_id: int) -> Optional[Prediction]:
        cur = self._conn.execute(
            """
            SELECT prediction_id, match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct, result_status
            FROM predictions WHERE prediction_id = ?
            """,
            (prediction_id,),
        )
        row = cur.fetchone()
        if row:
            return Prediction(
                id=row[0],
                match_id=row[1],
                model_name=row[2],
                prob_home=row[3],
                prob_draw=row[4],
                prob_away=row[5],
                predicted_result=row[6],
                is_correct=bool(row[7]) if row[7] is not None else None,
                result_status=(str(row[8]) if row[8] is not None else "PENDING"),
            )
        return None

    def list_by_match(
        self, match_id: int, *, limit: int = 100, offset: int = 0
    ) -> list[Prediction]:
        cur = self._conn.execute(
            """
            SELECT prediction_id, match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct, result_status
            FROM predictions
            WHERE match_id = ?
            LIMIT ? OFFSET ?
            """,
            (match_id, limit, offset),
        )
        rows = cur.fetchall()
        return [
            Prediction(
                id=r[0],
                match_id=r[1],
                model_name=r[2],
                prob_home=r[3],
                prob_draw=r[4],
                prob_away=r[5],
                predicted_result=r[6],
                is_correct=(bool(r[7]) if r[7] is not None else None),
                result_status=(str(r[8]) if r[8] is not None else "PENDING"),
            )
            for r in rows
        ]

    def insert(self, prediction: Prediction) -> int:
        # Normalize to avoid strict floating equality issues in SQLite by snapping
        # to a binary-friendly grid so sum equals exactly 1.0 in IEEE754.
        ph, pd, pa = (
            float(prediction.prob_home),
            float(prediction.prob_draw),
            float(prediction.prob_away),
        )
        total = ph + pd + pa
        if total <= 0:
            raise ValueError("Probabilities must be positive and sum to 1")
        ph /= total
        pd /= total
        # snap to 1/2^20 grid
        SCALE = 1 << 20
        ih = max(0, min(SCALE, int(round(ph * SCALE))))
        id_ = max(0, min(SCALE, int(round(pd * SCALE))))
        if ih + id_ > SCALE:
            # adjust draw first
            id_ = SCALE - ih
        ia = SCALE - ih - id_
        ph = ih / SCALE
        pd = id_ / SCALE
        pa = ia / SCALE

        # Upsert on (match_id, model_name)
        cur = self._conn.execute(
            """
            INSERT INTO predictions (match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct, result_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id, model_name) DO UPDATE SET
                prob_home=excluded.prob_home,
                prob_draw=excluded.prob_draw,
                prob_away=excluded.prob_away,
                predicted_result=excluded.predicted_result,
                is_correct=COALESCE(excluded.is_correct, predictions.is_correct),
                result_status=COALESCE(excluded.result_status, predictions.result_status)
            """,
            (
                prediction.match_id,
                prediction.model_name,
                ph,
                pd,
                pa,
                prediction.predicted_result,
                int(prediction.is_correct) if prediction.is_correct is not None else None,
                (
                    prediction.result_status
                    if getattr(prediction, "result_status", None)
                    else "PENDING"
                ),
            ),
        )
        self._conn.commit()

        rowid = cur.lastrowid
        if rowid is None:
            raise RuntimeError("SQLite insert failed: no lastrowid (table: predictions)")
        return int(rowid)

    def update(self, prediction: Prediction) -> None:
        # Normalize and snap to binary-friendly grid for exact sum 1.0
        ph, pd, pa = (
            float(prediction.prob_home),
            float(prediction.prob_draw),
            float(prediction.prob_away),
        )
        total = ph + pd + pa
        if total <= 0:
            raise ValueError("Probabilities must be positive and sum to 1")
        ph /= total
        pd /= total
        SCALE = 1 << 20
        ih = max(0, min(SCALE, int(round(ph * SCALE))))
        id_ = max(0, min(SCALE, int(round(pd * SCALE))))
        if ih + id_ > SCALE:
            id_ = SCALE - ih
        ia = SCALE - ih - id_
        ph = ih / SCALE
        pd = id_ / SCALE
        pa = ia / SCALE
        # Prefer upsert semantics on (match_id, model_name)
        self._conn.execute(
            """
            INSERT INTO predictions (match_id, model_name, prob_home, prob_draw, prob_away, predicted_result, is_correct, result_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id, model_name) DO UPDATE SET
                prob_home=excluded.prob_home,
                prob_draw=excluded.prob_draw,
                prob_away=excluded.prob_away,
                predicted_result=excluded.predicted_result,
                is_correct=COALESCE(excluded.is_correct, predictions.is_correct),
                result_status=COALESCE(excluded.result_status, predictions.result_status)
            """,
            (
                prediction.match_id,
                prediction.model_name,
                ph,
                pd,
                pa,
                prediction.predicted_result,
                int(prediction.is_correct) if prediction.is_correct is not None else None,
                (
                    prediction.result_status
                    if getattr(prediction, "result_status", None)
                    else "PENDING"
                ),
            ),
        )
        self._conn.commit()

    def delete(self, prediction_id: int) -> None:
        self._conn.execute("DELETE FROM predictions WHERE prediction_id = ?", (prediction_id,))
        self._conn.commit()

    def mark_correct(self, prediction_id: int, is_correct: bool) -> None:
        status = "WIN" if is_correct else "LOSE"
        self._conn.execute(
            "UPDATE predictions SET is_correct = ?, result_status = ? WHERE prediction_id = ?",
            (int(is_correct), status, prediction_id),
        )
        self._conn.commit()
