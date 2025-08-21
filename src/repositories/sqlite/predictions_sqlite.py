from __future__ import annotations

import sqlite3
from typing import Optional

from ..predictions import Prediction, PredictionsRepo


class PredictionsRepoSqlite(PredictionsRepo):
    """SQLite implementation of :class:`PredictionsRepo`."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER NOT NULL,
                prob_home REAL NOT NULL,
                prob_draw REAL NOT NULL,
                prob_away REAL NOT NULL,
                is_correct INTEGER
            )
            """
        )
        self._conn.commit()

    def get_by_id(self, prediction_id: int) -> Optional[Prediction]:
        cur = self._conn.execute(
            "SELECT id, match_id, prob_home, prob_draw, prob_away, is_correct FROM predictions WHERE id = ?",
            (prediction_id,),
        )
        row = cur.fetchone()
        if row:
            return Prediction(
                row[0], row[1], row[2], row[3], row[4], bool(row[5]) if row[5] is not None else None
            )
        return None

    def list_by_match(
        self, match_id: int, *, limit: int = 100, offset: int = 0
    ) -> list[Prediction]:
        cur = self._conn.execute(
            """
            SELECT id, match_id, prob_home, prob_draw, prob_away, is_correct
            FROM predictions WHERE match_id = ? LIMIT ? OFFSET ?
            """,
            (match_id, limit, offset),
        )
        rows = cur.fetchall()
        return [
            Prediction(r[0], r[1], r[2], r[3], r[4], bool(r[5]) if r[5] is not None else None)
            for r in rows
        ]

    def insert(self, prediction: Prediction) -> int:
        total = prediction.prob_home + prediction.prob_draw + prediction.prob_away
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Probabilities must sum to 1")
        cur = self._conn.execute(
            """
            INSERT INTO predictions (match_id, prob_home, prob_draw, prob_away, is_correct)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction.match_id,
                prediction.prob_home,
                prediction.prob_draw,
                prediction.prob_away,
                int(prediction.is_correct) if prediction.is_correct is not None else None,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update(self, prediction: Prediction) -> None:
        total = prediction.prob_home + prediction.prob_draw + prediction.prob_away
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Probabilities must sum to 1")
        self._conn.execute(
            """
            UPDATE predictions
            SET match_id = ?, prob_home = ?, prob_draw = ?, prob_away = ?, is_correct = ?
            WHERE id = ?
            """,
            (
                prediction.match_id,
                prediction.prob_home,
                prediction.prob_draw,
                prediction.prob_away,
                int(prediction.is_correct) if prediction.is_correct is not None else None,
                prediction.id,
            ),
        )
        self._conn.commit()

    def delete(self, prediction_id: int) -> None:
        self._conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        self._conn.commit()

    def mark_correct(self, prediction_id: int, is_correct: bool) -> None:
        self._conn.execute(
            "UPDATE predictions SET is_correct = ? WHERE id = ?",
            (int(is_correct), prediction_id),
        )
        self._conn.commit()
