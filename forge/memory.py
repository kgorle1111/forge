"""Long-term memory: one SQLite file holds cards, schedule, attempts, and lessons.

The system "gets smarter as you use it" through this file: every attempt is
recorded, ease factors adapt per concept, weak concepts surface first in
reviews and get interleaved into new sessions.
"""
# ponytail: FTS5 keyword recall stands in for a vector DB (Chroma/FAISS),
# swap in embeddings when cross-topic semantic similarity measurably matters
import datetime as dt
import json
import os
import sqlite3
import time

from .sources import (glossary_terms, manifest_text, relevant_chunks,
                      source_citation_block as build_source_citation_block,
                      source_digest, unsupported_terms)

DEFAULT_DB = os.path.expanduser("~/.forge/forge.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards(
  id INTEGER PRIMARY KEY,
  topic TEXT NOT NULL,
  concept TEXT NOT NULL,
  ease REAL NOT NULL DEFAULT 2.5,
  interval_days REAL NOT NULL DEFAULT 0,
  due REAL NOT NULL,
  reps INTEGER NOT NULL DEFAULT 0,
  lapses INTEGER NOT NULL DEFAULT 0,
  suspended INTEGER NOT NULL DEFAULT 0,
  best_answer TEXT NOT NULL DEFAULT '',
  retired INTEGER NOT NULL DEFAULT 0,
  pinned INTEGER NOT NULL DEFAULT 0,
  UNIQUE(topic, concept)
);
CREATE TABLE IF NOT EXISTS attempts(
  id INTEGER PRIMARY KEY,
  card_id INTEGER NOT NULL REFERENCES cards(id),
  ts REAL NOT NULL,
  score REAL NOT NULL,
  missed_json TEXT NOT NULL DEFAULT '[]',
  feedback TEXT NOT NULL DEFAULT '',
  confidence REAL,
  answer TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS progress(
  topic TEXT PRIMARY KEY,
  concepts_json TEXT NOT NULL,
  concept_idx INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS learner_notes(
  topic TEXT NOT NULL,
  concept TEXT NOT NULL,
  note TEXT NOT NULL,
  ts REAL NOT NULL,
  PRIMARY KEY(topic, concept)
);
CREATE TABLE IF NOT EXISTS source_docs(
  topic TEXT NOT NULL,
  name TEXT NOT NULL,
  digest TEXT NOT NULL,
  text TEXT NOT NULL,
  chunks_json TEXT NOT NULL,
  ts REAL NOT NULL,
  PRIMARY KEY(topic, name)
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(topic, concept, lesson);
"""

DAY = 86400.0


def _rowdict(row) -> dict:
    return dict(row) if isinstance(row, sqlite3.Row) else dict(row)


def retrievability(interval_days: float, due: float, now: float | None = None):
    """FSRS-style recall-probability estimate: R = 0.9 exactly when the review
    lands on schedule, decaying exponentially past it. None for unreviewed cards."""
    if interval_days <= 0:
        return None
    now = now or time.time()
    elapsed = max(0.0, (now - (due - interval_days * DAY)) / DAY)
    return 0.9 ** (elapsed / interval_days)


def risk_score(card, now: float | None = None) -> float:
    """0..1-ish retrieval urgency: overdue + low recall + low ease + lapses."""
    now = now or time.time()
    c = _rowdict(card)
    due_in = (c["due"] - now) / DAY
    overdue = max(0.0, -due_in)
    recall = retrievability(c["interval_days"], c["due"], now)
    recall_risk = 0.5 if recall is None else max(0.0, 1.0 - recall)
    ease_risk = max(0.0, min(1.0, (2.5 - c["ease"]) / 1.2))
    lapse_risk = min(1.0, c["lapses"] / 4)
    due_risk = 1.0 if due_in <= 0 else max(0.0, 1.0 - due_in / 7)
    score = 0.40 * due_risk + 0.30 * recall_risk + 0.15 * ease_risk + 0.15 * lapse_risk
    return round(score + min(0.25, overdue / 30), 4)


def risk_reasons(card, now: float | None = None) -> list[str]:
    now = now or time.time()
    c = _rowdict(card)
    reasons = []
    due_in = (c["due"] - now) / DAY
    if due_in <= 0:
        reasons.append("due now" if due_in > -1 else f"{abs(due_in):.1f}d overdue")
    recall = retrievability(c["interval_days"], c["due"], now)
    if recall is not None and recall < 0.75:
        reasons.append(f"recall {recall * 100:.0f}%")
    if c["lapses"]:
        reasons.append(f"{c['lapses']} lapse(s)")
    if c["ease"] < 2.0:
        reasons.append(f"low ease {c['ease']:.2f}")
    return reasons or ["steady"]


class Memory:
    def __init__(self, path: str | None = None):
        path = path or os.environ.get("FORGE_DB", DEFAULT_DB)
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.path = path
        self._migrate()

    def _migrate(self):
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(cards)")}
        if "suspended" not in cols:
            self.db.execute(
                "ALTER TABLE cards ADD COLUMN suspended INTEGER NOT NULL DEFAULT 0")
        if "best_answer" not in cols:
            self.db.execute(
                "ALTER TABLE cards ADD COLUMN best_answer TEXT NOT NULL DEFAULT ''")
        if "retired" not in cols:
            self.db.execute(
                "ALTER TABLE cards ADD COLUMN retired INTEGER NOT NULL DEFAULT 0")
        if "pinned" not in cols:
            self.db.execute(
                "ALTER TABLE cards ADD COLUMN pinned INTEGER NOT NULL DEFAULT 0")
        attempt_cols = {r["name"] for r in self.db.execute("PRAGMA table_info(attempts)")}
        for col, spec in {
            "missed_json": "TEXT NOT NULL DEFAULT '[]'",
            "feedback": "TEXT NOT NULL DEFAULT ''",
            "confidence": "REAL",
            "answer": "TEXT NOT NULL DEFAULT ''",
        }.items():
            if col not in attempt_cols:
                self.db.execute(f"ALTER TABLE attempts ADD COLUMN {col} {spec}")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS learner_notes(
              topic TEXT NOT NULL,
              concept TEXT NOT NULL,
              note TEXT NOT NULL,
              ts REAL NOT NULL,
              PRIMARY KEY(topic, concept)
            )""")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS source_docs(
              topic TEXT NOT NULL,
              name TEXT NOT NULL,
              digest TEXT NOT NULL,
              text TEXT NOT NULL,
              chunks_json TEXT NOT NULL,
              ts REAL NOT NULL,
              PRIMARY KEY(topic, name)
            )""")
        self.db.commit()

    def record(self, topic: str, concept: str, score: float, lesson: str = "",
               now: float | None = None, missed: list[str] | None = None,
               feedback: str = "", confidence: float | None = None,
               answer: str = "") -> dict:
        """SM-2 update. Interval growth == ease (default 2.5x), which lands each
        review inside the 10-40% retention-decay window (the "10-20% rule")."""
        now = now or time.time()
        self.db.execute(
            "INSERT OR IGNORE INTO cards(topic, concept, due) VALUES(?,?,?)",
            (topic, concept, now))
        card = self.db.execute(
            "SELECT * FROM cards WHERE topic=? AND concept=?", (topic, concept)).fetchone()
        ease, interval, reps, lapses = card["ease"], card["interval_days"], card["reps"], card["lapses"]
        q = round(score * 5)  # SM-2 quality 0..5
        if q < 3:
            reps, interval, lapses = 0, 1.0, lapses + 1
        else:
            reps += 1
            interval = 1.0 if reps == 1 else 6.0 if reps == 2 else interval * ease
            # R1 fix: unbroken perfect streaks compounded to decades — cap like
            # Anki (ease ceiling 2.5, interval ceiling 1 year)
            interval = min(interval, 365.0)
            ease = min(2.5, max(1.3, ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02)))
        due = now + interval * DAY
        self.db.execute(
            "UPDATE cards SET ease=?, interval_days=?, due=?, reps=?, lapses=?, retired=0 WHERE id=?",
            (ease, interval, due, reps, lapses, card["id"]))
        self.db.execute(
            "INSERT INTO attempts(card_id, ts, score, missed_json, feedback, confidence, answer) "
            "VALUES(?,?,?,?,?,?,?)",
            (card["id"], now, score, json.dumps(missed or []),
             feedback[:500], confidence, answer[:2000]))
        if score >= 0.8 and answer.strip():
            self.db.execute("UPDATE cards SET best_answer=? WHERE id=?",
                            (answer.strip()[:2000], card["id"]))
        if lesson:
            self.db.execute("DELETE FROM notes WHERE topic=? AND concept=?", (topic, concept))
            self.db.execute("INSERT INTO notes(topic, concept, lesson) VALUES(?,?,?)",
                            (topic, concept, lesson))
        self.db.commit()
        return {"topic": topic, "concept": concept, "score": score,
                "next_interval_days": interval, "due": due, "ease": ease}

    def due_cards(self, now: float | None = None) -> list[sqlite3.Row]:
        now = now or time.time()
        return self.db.execute(
            "SELECT * FROM cards WHERE due <= ? AND suspended = 0 AND retired = 0 ORDER BY due",
            (now,)).fetchall()

    def due_queue(self, limit: int | None = None, days: int | None = None,
                  now: float | None = None) -> list[dict]:
        now = now or time.time()
        cutoff = now if days is None else now + days * DAY
        rows = self.db.execute(
            "SELECT * FROM cards WHERE due <= ? AND suspended = 0 AND retired = 0",
            (cutoff,)).fetchall()
        cards = [self._card_summary(r, now) for r in rows]
        cards.sort(key=lambda r: (-r["yield_score"], r["due"]))
        return cards[:limit] if limit else cards

    def weakest(self, n: int = 5, exclude_topic: str = "") -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM cards WHERE topic != ? AND suspended = 0 AND retired = 0 "
            "ORDER BY lapses DESC, ease ASC LIMIT ?",
            (exclude_topic, n)).fetchall()

    def weak_cards(self, n: int = 10, now: float | None = None) -> list[dict]:
        now = now or time.time()
        cards = [self._card_summary(r, now) for r in self.stats()
                 if not r["suspended"] and not r["retired"]]
        cards.sort(key=lambda r: (-r["risk"], -r["lapses"], r["ease"], r["topic"], r["concept"]))
        return cards[:n]

    def lesson_for(self, topic: str, concept: str) -> str:
        row = self.db.execute(
            "SELECT lesson FROM notes WHERE topic=? AND concept=?", (topic, concept)).fetchone()
        return row["lesson"] if row else ""

    def recall(self, query: str, n: int = 5) -> list[sqlite3.Row]:
        safe = " ".join(w for w in query.split() if w.isalnum())
        if not safe:
            return []
        return self.db.execute(
            "SELECT topic, concept, lesson FROM notes WHERE notes MATCH ? LIMIT ?",
            (safe, n)).fetchall()

    def search_lessons(self, query: str, n: int = 10) -> list[dict]:
        rows = self.recall(query, n)
        return [dict(r) | {"excerpt": _excerpt(r["lesson"], query)} for r in rows]

    def semantic_recall(self, query: str, n: int = 10) -> list[dict]:
        q = set(_semantic_terms(query))
        if not q:
            return []
        out = []
        for r in self.db.execute("SELECT topic, concept, lesson FROM notes"):
            terms = set(_semantic_terms(r["lesson"] + " " + r["concept"]))
            if not terms:
                continue
            overlap = len(q & terms)
            related = sum(1 for a in q for b in terms
                          if a != b and (a in b or b in a))
            score = (overlap + 0.25 * related) / max(1, len(q | terms))
            if score:
                out.append(dict(r) | {"score": round(score, 4),
                                      "excerpt": _excerpt(r["lesson"], query)})
        out.sort(key=lambda r: (-r["score"], r["topic"], r["concept"]))
        return out[:n]

    def forecast(self, days: int = 14, now: float | None = None) -> list[int]:
        """Review workload per day for the next `days` days (overdue counts as day 0)."""
        now = now or time.time()
        counts = [0] * days
        for r in self.db.execute("SELECT due FROM cards WHERE suspended = 0 AND retired = 0"):
            d = max(0, int((r["due"] - now) // DAY))
            if d < days:
                counts[d] += 1
        return counts

    def streak(self, now: float | None = None) -> int:
        """Consecutive days (ending today, or yesterday as grace) with >=1 attempt."""
        now = now or time.time()
        days = {dt.date.fromtimestamp(r["ts"])
                for r in self.db.execute("SELECT ts FROM attempts")}
        d = dt.date.fromtimestamp(now)
        if d not in days:
            d -= dt.timedelta(days=1)
        n = 0
        while d in days:
            n += 1
            d -= dt.timedelta(days=1)
        return n

    def topic_summaries(self, now: float | None = None) -> list[dict]:
        now = now or time.time()
        rows = [self._card_summary(r, now) for r in self.stats()]
        by_topic: dict[str, list[dict]] = {}
        for r in rows:
            by_topic.setdefault(r["topic"], []).append(r)
        active = {r["topic"]: r["concept_idx"] for r in
                  self.db.execute("SELECT topic, concept_idx FROM progress")}
        out = []
        for topic, cards in sorted(by_topic.items()):
            avg = sum(c["avg_score"] or 0 for c in cards) / len(cards)
            active_cards = [c for c in cards if not c["suspended"] and not c["retired"]]
            risk = max((c["risk"] for c in active_cards), default=0.0)
            due = sum(1 for c in active_cards if c["due_in_days"] <= 0)
            out.append({"topic": topic, "concepts": len(cards), "due": due,
                        "weak": sum(1 for c in active_cards if c["risk"] >= 0.5),
                        "avg_score": avg, "risk": risk,
                        "health": round(max(0.0, 1.0 - risk) * (0.5 + 0.5 * avg), 4),
                        "active_idx": active.get(topic)})
        for topic, idx in active.items():
            if topic not in by_topic:
                out.append({"topic": topic, "concepts": 0, "due": 0, "weak": 0,
                            "avg_score": 0.0, "risk": 0.0, "health": 0.0,
                            "active_idx": idx})
        return out

    def card_detail(self, topic: str, concept: str,
                    attempts: int = 10, now: float | None = None) -> dict | None:
        now = now or time.time()
        row = self.db.execute("""
            SELECT c.*, COUNT(a.id) AS n_attempts, AVG(a.score) AS avg_score
            FROM cards c LEFT JOIN attempts a ON a.card_id = c.id
            WHERE c.topic=? AND c.concept=?
            GROUP BY c.id""", (topic, concept)).fetchone()
        if not row:
            return None
        detail = self._card_summary(row, now)
        detail["lesson"] = self.lesson_for(topic, concept)
        detail["learner_note"] = self.learner_note(topic, concept)
        detail["attempts"] = self.attempt_timeline(attempts, topic, concept)
        detail["missed_points"] = self.missed_points(topic, concept, 10)
        return detail

    def attempt_timeline(self, limit: int = 20, topic: str = "",
                         concept: str = "") -> list[dict]:
        where, params = [], []
        if topic:
            where.append("c.topic=?")
            params.append(topic)
        if concept:
            where.append("c.concept=?")
            params.append(concept)
        clause = "WHERE " + " AND ".join(where) if where else ""
        params.append(limit)
        return [dict(r) for r in self.db.execute(f"""
            SELECT a.ts, a.score, a.missed_json, a.feedback, a.confidence,
                   a.answer, c.topic, c.concept
            FROM attempts a JOIN cards c ON c.id = a.card_id
            {clause}
            ORDER BY a.ts DESC LIMIT ?""", params)]

    def review_debt(self, now: float | None = None) -> dict:
        now = now or time.time()
        due = self.due_queue(now=now)
        risks = [c["risk"] for c in due]
        overdue_days = [max(0.0, -c["due_in_days"]) for c in due]
        return {"due": len(due), "peak_risk": max(risks, default=0.0),
                "avg_risk": sum(risks) / len(risks) if risks else 0.0,
                "max_overdue_days": max(overdue_days, default=0.0),
                "review_days_at_20": (len(due) + 19) // 20}

    def analytics_summary(self, now: float | None = None) -> dict:
        now = now or time.time()
        cards = [self._card_summary(r, now) for r in self.stats()]
        attempts = self.attempt_timeline(100000)
        by_topic: dict[str, list[dict]] = {}
        for c in cards:
            by_topic.setdefault(c["topic"], []).append(c)
        success = {}
        for topic, rows in by_topic.items():
            topic_attempts = [a for a in attempts if a["topic"] == topic]
            success[topic] = (sum(1 for a in topic_attempts if a["score"] >= 0.8)
                              / len(topic_attempts) if topic_attempts else 0.0)
        expensive = sorted(cards, key=lambda c: (-c["n_attempts"], -c["lapses"],
                                                 c["topic"], c["concept"]))[:10]
        best = sorted(cards, key=lambda c: (-(c["recall"] or 0), -c["avg_score"],
                                            c["topic"], c["concept"]))[:10]
        neglected = sorted(cards, key=lambda c: c["due"])[:10]
        return {
            "attempts_per_day": _counts_by_day(attempts),
            "success_rate_by_topic": success,
            "lapses_by_topic": {t: sum(c["lapses"] for c in rows)
                                for t, rows in by_topic.items()},
            "ease_distribution": _buckets([c["ease"] for c in cards], 0.5),
            "longest_neglected": neglected,
            "newly_learned_this_week": sum(1 for c in cards if c["reps"] == 1),
            "reviews_due_this_week": len(self.due_queue(days=7, now=now)),
            "reviews_completed_this_week": sum(1 for a in attempts
                                               if now - a["ts"] <= 7 * DAY),
            "average_score": (sum(a["score"] for a in attempts) / len(attempts)
                              if attempts else 0.0),
            "average_retry_count": (sum(max(0, c["n_attempts"] - 1) for c in cards)
                                    / len(cards) if cards else 0.0),
            "most_expensive": expensive,
            "best_retained": best,
            "review_workload": self.forecast(14, now),
            "streak_calendar": sorted(_counts_by_day(attempts)),
            "weekly_learning_volume": sum(1 for a in attempts if now - a["ts"] <= 7 * DAY),
        }

    def high_yield_queue(self, limit: int = 20, days: int | None = None,
                         now: float | None = None) -> list[dict]:
        return self.due_queue(limit=limit, days=days, now=now)

    def sprint_cards(self, mode: str = "mixed", limit: int = 10,
                     topic: str = "", now: float | None = None) -> list[dict]:
        now = now or time.time()
        cards = [self._card_summary(r, now) for r in self.stats()
                 if not r["suspended"] and not r["retired"]]
        if topic:
            cards = [c for c in cards if c["topic"] == topic]
        if mode == "weak":
            cards.sort(key=lambda c: (-c["risk"], -c["lapses"], c["topic"], c["concept"]))
        elif mode == "newest":
            cards.sort(key=lambda c: (-c["id"], c["topic"], c["concept"]))
        elif mode == "oldest":
            cards.sort(key=lambda c: (c["id"], c["topic"], c["concept"]))
        elif mode == "overdue":
            cards = [c for c in cards if c["due_in_days"] <= 0]
            cards.sort(key=lambda c: (c["due"], -c["risk"]))
        elif mode == "lapse":
            cards = [c for c in cards if c["lapses"] > 0]
            cards.sort(key=lambda c: (-c["lapses"], -c["risk"]))
        else:
            cards.sort(key=lambda c: (-c["yield_score"], c["due"]))
        return cards[:limit]

    def stuck_cards(self, n: int = 10, min_lapses: int = 1,
                    now: float | None = None) -> list[dict]:
        now = now or time.time()
        cards = [self._card_summary(r, now) for r in self.stats()
                 if not r["suspended"] and not r["retired"] and r["lapses"] >= min_lapses]
        cards.sort(key=lambda r: (-r["lapses"], -r["risk"], r["topic"], r["concept"]))
        return cards[:n]

    def progress_rows(self) -> list[dict]:
        rows = []
        for r in self.db.execute("SELECT topic, concepts_json, concept_idx FROM progress ORDER BY topic"):
            try:
                concepts = json.loads(r["concepts_json"])
            except json.JSONDecodeError:
                concepts = []
            idx = min(r["concept_idx"], max(0, len(concepts) - 1)) if concepts else 0
            rows.append({"topic": r["topic"], "concept_idx": r["concept_idx"],
                         "total": len(concepts),
                         "next_concept": concepts[idx].get("name", "") if concepts else ""})
        return rows

    def due_now(self, topic: str, concept: str, now: float | None = None) -> dict:
        now = now or time.time()
        card = self._card(topic, concept)
        self.db.execute("UPDATE cards SET due=?, suspended=0 WHERE id=?",
                        (now, card["id"]))
        self.db.commit()
        detail = self.card_detail(topic, concept, now=now)
        return detail or {}

    def suspend_card(self, topic: str, concept: str) -> dict:
        card = self._card(topic, concept)
        self.db.execute("UPDATE cards SET suspended=1 WHERE id=?", (card["id"],))
        self.db.commit()
        return dict(self._card(topic, concept))

    def unsuspend_card(self, topic: str, concept: str) -> dict:
        card = self._card(topic, concept)
        self.db.execute("UPDATE cards SET suspended=0 WHERE id=?", (card["id"],))
        self.db.commit()
        return dict(self._card(topic, concept))

    def retire_card(self, topic: str, concept: str) -> dict:
        card = self._card(topic, concept)
        self.db.execute("UPDATE cards SET retired=1, pinned=0 WHERE id=?",
                        (card["id"],))
        self.db.commit()
        return dict(self._card(topic, concept))

    def unretire_card(self, topic: str, concept: str) -> dict:
        card = self._card(topic, concept)
        self.db.execute("UPDATE cards SET retired=0 WHERE id=?", (card["id"],))
        self.db.commit()
        return dict(self._card(topic, concept))

    def pin_card(self, topic: str, concept: str, now: float | None = None) -> dict:
        now = now or time.time()
        card = self._card(topic, concept)
        self.db.execute(
            "UPDATE cards SET pinned=1, retired=0, suspended=0, due=MIN(due, ?) WHERE id=?",
            (now, card["id"]))
        self.db.commit()
        return dict(self._card(topic, concept))

    def unpin_card(self, topic: str, concept: str) -> dict:
        card = self._card(topic, concept)
        self.db.execute("UPDATE cards SET pinned=0 WHERE id=?", (card["id"],))
        self.db.commit()
        return dict(self._card(topic, concept))

    def shift_due(self, days: float, topic: str = "", concept: str = "") -> int:
        where, params = ["retired=0"], []
        if topic:
            where.append("topic=?")
            params.append(topic)
        if concept:
            where.append("concept=?")
            params.append(concept)
        params.insert(0, days * DAY)
        sql = f"UPDATE cards SET due = due + ? WHERE {' AND '.join(where)}"
        cur = self.db.execute(sql, params)
        self.db.commit()
        return cur.rowcount

    def rename_topic(self, old: str, new: str) -> dict:
        old, new = old.strip(), new.strip()
        if not old or not new:
            raise ValueError("topic names cannot be blank")
        if old == new:
            return {"topic": new, "cards": 0}
        mine = {r["concept"] for r in
                self.db.execute("SELECT concept FROM cards WHERE topic=?", (old,))}
        if not mine:
            raise ValueError(f"topic not found: {old}")
        conflicts = mine & {r["concept"] for r in
                            self.db.execute("SELECT concept FROM cards WHERE topic=?", (new,))}
        if conflicts:
            raise ValueError(f"rename would collide on concept: {sorted(conflicts)[0]}")
        with self.db:
            self.db.execute("UPDATE cards SET topic=? WHERE topic=?", (new, old))
            self.db.execute("UPDATE notes SET topic=? WHERE topic=?", (new, old))
            self.db.execute("UPDATE progress SET topic=? WHERE topic=?", (new, old))
            self.db.execute("UPDATE source_docs SET topic=? WHERE topic=?", (new, old))
        return {"topic": new, "cards": len(mine)}

    def rename_concept(self, topic: str, old: str, new: str) -> dict:
        topic, old, new = topic.strip(), old.strip(), new.strip()
        if not topic or not old or not new:
            raise ValueError("topic and concept names cannot be blank")
        if old == new:
            return {"topic": topic, "concept": new}
        card = self._card(topic, old)
        if self.db.execute("SELECT 1 FROM cards WHERE topic=? AND concept=?",
                           (topic, new)).fetchone():
            raise ValueError(f"concept already exists: {new}")
        with self.db:
            self.db.execute("UPDATE cards SET concept=? WHERE id=?", (new, card["id"]))
            self.db.execute("UPDATE notes SET concept=? WHERE topic=? AND concept=?",
                            (new, topic, old))
            saved = self.load_progress(topic)
            if saved:
                concepts, idx = saved
                for c in concepts:
                    if c.get("name") == old:
                        c["name"] = new
                self.db.execute(
                    "INSERT OR REPLACE INTO progress(topic, concepts_json, concept_idx) VALUES(?,?,?)",
                    (topic, json.dumps(concepts), idx))
        return {"topic": topic, "concept": new}

    def merge_topic(self, old: str, target: str) -> dict:
        old, target = old.strip(), target.strip()
        if not old or not target or old == target:
            raise ValueError("old and target topics must be different nonblank names")
        old_cards = self.db.execute("SELECT * FROM cards WHERE topic=?", (old,)).fetchall()
        if not old_cards:
            raise ValueError(f"topic not found: {old}")
        moved, merged = 0, 0
        with self.db:
            for card in old_cards:
                existing = self.db.execute(
                    "SELECT * FROM cards WHERE topic=? AND concept=?",
                    (target, card["concept"])).fetchone()
                if existing:
                    self._merge_card_rows(card, existing)
                    merged += 1
                else:
                    self.db.execute("UPDATE cards SET topic=? WHERE id=?",
                                    (target, card["id"]))
                    self.db.execute("UPDATE notes SET topic=? WHERE topic=? AND concept=?",
                                    (target, old, card["concept"]))
                    self.db.execute(
                        "UPDATE learner_notes SET topic=? WHERE topic=? AND concept=?",
                        (target, old, card["concept"]))
                    moved += 1
            self.db.execute("DELETE FROM progress WHERE topic=?", (old,))
            for doc in self.db.execute(
                    "SELECT name, digest, text, chunks_json, ts FROM source_docs WHERE topic=?",
                    (old,)).fetchall():
                self.db.execute(
                    "INSERT OR REPLACE INTO source_docs(topic, name, digest, text, chunks_json, ts) "
                    "VALUES(?,?,?,?,?,?)",
                    (target, doc["name"], doc["digest"], doc["text"],
                     doc["chunks_json"], doc["ts"]))
            self.db.execute("DELETE FROM source_docs WHERE topic=?", (old,))
        return {"topic": target, "moved": moved, "merged": merged}

    def merge_concept(self, topic: str, old: str, target: str) -> dict:
        old_card = self._card(topic, old)
        target_card = self._card(topic, target)
        with self.db:
            self._merge_card_rows(old_card, target_card)
        return {"topic": topic, "concept": target}

    def delete_topic(self, topic: str, confirm: str) -> int:
        if confirm != topic:
            raise ValueError("delete-topic requires --confirm with the exact topic")
        ids = [r["id"] for r in self.db.execute("SELECT id FROM cards WHERE topic=?",
                                                (topic,))]
        with self.db:
            for cid in ids:
                self.db.execute("DELETE FROM attempts WHERE card_id=?", (cid,))
            self.db.execute("DELETE FROM cards WHERE topic=?", (topic,))
            self.db.execute("DELETE FROM notes WHERE topic=?", (topic,))
            self.db.execute("DELETE FROM learner_notes WHERE topic=?", (topic,))
            self.db.execute("DELETE FROM progress WHERE topic=?", (topic,))
            self.db.execute("DELETE FROM source_docs WHERE topic=?", (topic,))
        return len(ids)

    def delete_concept(self, topic: str, concept: str, confirm: str) -> int:
        if confirm != f"{topic}/{concept}":
            raise ValueError("delete-concept requires --confirm TOPIC/CONCEPT")
        card = self._card(topic, concept)
        with self.db:
            self.db.execute("DELETE FROM attempts WHERE card_id=?", (card["id"],))
            self.db.execute("DELETE FROM cards WHERE id=?", (card["id"],))
            self.db.execute("DELETE FROM notes WHERE topic=? AND concept=?",
                            (topic, concept))
            self.db.execute("DELETE FROM learner_notes WHERE topic=? AND concept=?",
                            (topic, concept))
        return 1

    def missing_lessons(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("""
            SELECT c.topic, c.concept FROM cards c
            LEFT JOIN notes n ON n.topic = c.topic AND n.concept = c.concept
            WHERE n.lesson IS NULL OR n.lesson = ''
            ORDER BY c.topic, c.concept""")]

    def duplicate_concepts(self) -> list[dict]:
        rows = self.db.execute("SELECT topic, concept FROM cards").fetchall()
        seen: dict[str, list[dict]] = {}
        for r in rows:
            key = " ".join(r["concept"].lower().split())
            seen.setdefault(key, []).append(dict(r))
        return [{"concept_key": k, "cards": v} for k, v in seen.items() if len(v) > 1]

    def missed_points(self, topic: str = "", concept: str = "",
                      limit: int = 20) -> list[dict]:
        where, params = [], []
        if topic:
            where.append("c.topic=?")
            params.append(topic)
        if concept:
            where.append("c.concept=?")
            params.append(concept)
        clause = "WHERE " + " AND ".join(where) if where else ""
        counts: dict[tuple[str, str, str], int] = {}
        for r in self.db.execute(f"""
            SELECT c.topic, c.concept, a.missed_json
            FROM attempts a JOIN cards c ON c.id = a.card_id
            {clause}""", params):
            try:
                missed = json.loads(r["missed_json"] or "[]")
            except json.JSONDecodeError:
                missed = []
            for point in missed:
                key = (r["topic"], r["concept"], str(point))
                counts[key] = counts.get(key, 0) + 1
        rows = [{"topic": t, "concept": c, "point": p, "count": n}
                for (t, c, p), n in counts.items()]
        rows.sort(key=lambda r: (-r["count"], r["topic"], r["concept"], r["point"]))
        return rows[:limit]

    def calibration(self) -> dict:
        rows = self.db.execute("""
            SELECT confidence, score FROM attempts WHERE confidence IS NOT NULL
            ORDER BY ts DESC""").fetchall()
        if not rows:
            return {"n": 0, "avg_confidence": None, "avg_score": None,
                    "calibration_gap": None}
        avg_conf = sum(r["confidence"] for r in rows) / len(rows)
        avg_score = sum(r["score"] for r in rows) / len(rows)
        return {"n": len(rows), "avg_confidence": avg_conf,
                "avg_score": avg_score,
                "calibration_gap": avg_score - avg_conf / 5.0}

    def avalanche_warning(self, days: int = 14, daily_budget: int = 20,
                          now: float | None = None) -> dict:
        fc = self.forecast(days, now)
        peak = max(fc, default=0)
        total = sum(fc)
        overloaded = [i for i, c in enumerate(fc) if c > daily_budget]
        return {"days": days, "daily_budget": daily_budget, "forecast": fc,
                "total": total, "peak_day": fc.index(peak) if fc else 0,
                "peak_count": peak, "overloaded_days": overloaded,
                "warning": bool(overloaded or total > daily_budget * max(1, days // 2))}

    def performance_by_weekday(self) -> list[dict]:
        return self._performance_buckets(lambda ts: dt.datetime.fromtimestamp(ts).strftime("%a"))

    def performance_by_hour(self) -> list[dict]:
        return self._performance_buckets(lambda ts: f"{dt.datetime.fromtimestamp(ts).hour:02d}:00")

    def learner_note(self, topic: str, concept: str) -> str:
        row = self.db.execute(
            "SELECT note FROM learner_notes WHERE topic=? AND concept=?",
            (topic, concept)).fetchone()
        return row["note"] if row else ""

    def set_learner_note(self, topic: str, concept: str, note: str,
                         replace_lesson: bool = False,
                         now: float | None = None) -> dict:
        now = now or time.time()
        self._card(topic, concept)
        note = note.strip()
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO learner_notes(topic, concept, note, ts) VALUES(?,?,?,?)",
                (topic, concept, note, now))
            if replace_lesson:
                self.db.execute("DELETE FROM notes WHERE topic=? AND concept=?",
                                (topic, concept))
                self.db.execute("INSERT INTO notes(topic, concept, lesson) VALUES(?,?,?)",
                                (topic, concept, note))
        return {"topic": topic, "concept": concept, "note": note}

    def note_from_best_answer(self, topic: str, concept: str,
                              replace_lesson: bool = False) -> dict:
        card = self._card(topic, concept)
        if not card["best_answer"]:
            raise ValueError("no passing answer stored for this concept yet")
        return self.set_learner_note(topic, concept, card["best_answer"],
                                     replace_lesson=replace_lesson)

    def retrieval_prompts(self, topic: str, concept: str) -> list[dict]:
        lesson = self.lesson_for(topic, concept)
        weak = next((r for r in self.weak_cards(5) if r["concept"] != concept), None)
        prompts = [
            ("counterexample", f"Give a plausible counterexample or non-example for '{concept}', then explain why it does not fit."),
            ("compare", f"Compare '{concept}' with '{weak['concept'] if weak else 'a nearby concept'}' without using the lesson text."),
            ("breaks", f"What would break or become false if the central mechanism of '{concept}' were removed?"),
            ("one_sentence", f"Teach '{concept}' back in one precise sentence from memory."),
        ]
        if lesson:
            prompts.append(("key_points", "Recover the required key points: " +
                            "; ".join(_key_points(lesson)[:5])))
        prompts += [
            ("three_mechanisms", f"List three mechanisms or moving parts inside '{concept}', then say how they interact."),
            ("missing_step", f"Recover the step most likely to be skipped when explaining '{concept}'."),
            ("predict_result", f"Predict what happens when the main input to '{concept}' changes, and justify it."),
            ("failure_mode", f"Explain the failure mode of '{concept}' without naming the topic first."),
            ("boundary_condition", f"Name a boundary condition where '{concept}' stops applying cleanly."),
            ("life_example", f"Give a real-life example of '{concept}' and map each part back to the lesson."),
            ("non_example", f"Give a non-example of '{concept}' and explain why it fails."),
            ("dependency_order", f"Order the dependencies needed to understand '{concept}' from first to last."),
            ("define_then_use", f"Define '{concept}' precisely, then use it in one concrete case."),
            ("notation_to_words", f"If '{concept}' has notation or jargon, translate it into plain words."),
            ("words_to_notation", f"Translate a plain-language description of '{concept}' back into its formal terms."),
            ("proof_sketch", f"Give the shortest proof sketch or causal argument for why '{concept}' works."),
            ("debug_misconception", f"State a tempting misconception about '{concept}', then debug it."),
            ("hidden_assumption", f"Identify a hidden assumption behind '{concept}'."),
            ("tempting_wrong", f"Explain why the tempting wrong answer for '{concept}' is wrong."),
            ("diagram_label", f"Imagine a diagram of '{concept}'; recover the labels and arrows from memory."),
            ("invariant", f"State what remains invariant while '{concept}' operates."),
            ("bottleneck", f"Find the bottleneck or limiting factor in '{concept}'."),
            ("three_sentence", f"Explain '{concept}' in exactly three sentences."),
            ("no_jargon", f"Explain '{concept}' without jargon."),
            ("jargon_correctly", f"Use the key jargon for '{concept}' correctly in one sentence."),
            ("concrete_object", f"Explain '{concept}' using one concrete object or scene."),
        ]
        return [{"kind": k, "prompt": p} for k, p in prompts]

    def cloze_cards(self, topic: str = "", limit: int = 50) -> list[dict]:
        where, params = [], []
        if topic:
            where.append("topic=?")
            params.append(topic)
        clause = "WHERE " + " AND ".join(where) if where else ""
        params.append(limit)
        rows = self.db.execute(
            f"SELECT topic, concept, lesson FROM notes {clause} ORDER BY topic, concept LIMIT ?",
            params).fetchall()
        out = []
        for r in rows:
            key = _key_points(r["lesson"])[:1]
            if not key:
                continue
            front = r["lesson"].replace(key[0], "{{c1::" + key[0] + "}}", 1)
            out.append({"topic": r["topic"], "concept": r["concept"], "cloze": front})
        return out

    def maintenance_report(self, now: float | None = None) -> dict:
        now = now or time.time()
        cards = [dict(r) for r in self.stats()]
        attempts = [dict(r) for r in self.db.execute("SELECT * FROM attempts")]
        card_ids = {c["id"] for c in cards}
        source_docs = self._table("source_docs")
        return {
            "integrity": self.db.execute("PRAGMA integrity_check").fetchone()[0],
            "cards": len(cards),
            "attempts": len(attempts),
            "notes": len(self._table("notes", ["topic", "concept", "lesson"])),
            "source_docs": len(source_docs),
            "missing_lessons": self.missing_lessons(),
            "duplicates": self.duplicate_concepts(),
            "orphan_attempts": [a for a in attempts if a["card_id"] not in card_ids],
            "orphan_notes": self._orphan_notes(),
            "active_progress": len(self._table("progress")),
            "future_due_outliers": [c for c in cards if c["due"] - now > 365 * DAY],
            "negative_intervals": [c for c in cards if c["interval_days"] < 0],
            "bad_scores": [dict(r) for r in self.db.execute(
                "SELECT * FROM attempts WHERE score < 0 OR score > 1")],
            "db_bytes": os.path.getsize(self.path) if self.path != ":memory:" and os.path.exists(self.path) else 0,
        }

    def source_manifest(self, topic: str) -> dict:
        docs = []
        for r in self.db.execute("""
            SELECT name, digest, text, chunks_json FROM source_docs
            WHERE topic=? ORDER BY name""", (topic,)):
            try:
                chunks = json.loads(r["chunks_json"] or "[]")
            except json.JSONDecodeError:
                chunks = []
            docs.append({"name": r["name"], "digest": r["digest"],
                         "text": r["text"], "chunks": chunks})
        return {"docs": docs,
                "digest": source_digest("\n".join(d["text"] for d in docs))}

    def save_source_manifest(self, topic: str, manifest: dict) -> dict:
        old = self.source_manifest(topic)
        old_digest = old["digest"] if old.get("docs") else ""
        docs = manifest.get("docs", [])
        new_digest = manifest.get("digest") or source_digest(
            "\n".join(d.get("text", "") for d in docs))
        with self.db:
            self.db.execute("DELETE FROM source_docs WHERE topic=?", (topic,))
            for i, doc in enumerate(docs, 1):
                name = doc.get("name") or f"source-{i}"
                self.db.execute(
                    "INSERT OR REPLACE INTO source_docs(topic, name, digest, text, chunks_json, ts) "
                    "VALUES(?,?,?,?,?,?)",
                    (topic, name, doc.get("digest", ""),
                     doc.get("text", ""), json.dumps(doc.get("chunks", [])),
                     time.time()))
        return {"topic": topic, "docs": len(docs), "digest": new_digest,
                "old_digest": old_digest, "changed": old_digest != new_digest}

    def source_snippets(self, topic: str, query: str, limit: int = 4) -> list[dict]:
        return relevant_chunks(self.source_manifest(topic), query, limit)

    def source_citation_block(self, topic: str, query: str,
                              limit: int = 4) -> str:
        return build_source_citation_block(self.source_manifest(topic), query, limit)

    def source_glossary(self, topic: str, limit: int = 20) -> list[dict]:
        return glossary_terms(manifest_text(self.source_manifest(topic)), limit)

    def source_report(self, topic: str) -> dict:
        manifest = self.source_manifest(topic)
        docs = manifest.get("docs", [])
        return {
            "topic": topic,
            "digest": manifest.get("digest", ""),
            "docs": [{"name": d.get("name", ""), "digest": d.get("digest", ""),
                      "chars": len(d.get("text", "")),
                      "chunks": len(d.get("chunks", []))}
                     for d in docs],
            "chunks": sum(len(d.get("chunks", [])) for d in docs),
            "glossary": self.source_glossary(topic, 12),
            "coverage": self.source_coverage(topic),
        }

    def source_quiz(self, topic: str, kind: str = "glossary",
                    limit: int = 8) -> list[dict]:
        manifest = self.source_manifest(topic)
        if not manifest.get("docs"):
            return []
        if kind == "section":
            rows = []
            for chunk in relevant_chunks(manifest, topic, limit):
                rows.append({"kind": "section", "label": chunk["label"],
                             "prompt": f"From memory, reconstruct the key claim in [{chunk['label']}] ({chunk['source']})."})
            return rows
        if kind == "quote":
            return [{"kind": "quote", "label": c["label"],
                     "prompt": f"Which source chunk supports this idea? {c['text'][:120].replace(chr(10), ' ')}"}
                    for c in relevant_chunks(manifest, topic, limit)]
        return [{"kind": "glossary", "term": r["term"],
                 "prompt": f"Define '{r['term']}' from the source, then use it in one sentence."}
                for r in self.source_glossary(topic, limit)]

    def source_coverage(self, topic: str) -> list[dict]:
        manifest = self.source_manifest(topic)
        source_text = manifest_text(manifest)
        if not source_text:
            return []
        rows = self.db.execute("""
            SELECT c.topic, c.concept, COALESCE(n.lesson, '') AS lesson
            FROM cards c LEFT JOIN notes n ON n.topic = c.topic AND n.concept = c.concept
            WHERE c.topic=? ORDER BY c.concept""", (topic,)).fetchall()
        if not rows:
            saved = self.load_progress(topic)
            rows = [{"topic": topic, "concept": c.get("name", ""),
                     "lesson": c.get("summary", "")}
                    for c in (saved[0] if saved else [])]
        out = []
        for r in rows:
            text = f"{r['concept']} {r['lesson']}"
            snippets = relevant_chunks(manifest, text, 3)
            supported = [s for s in snippets if s["score"] > 0]
            out.append({
                "topic": r["topic"],
                "concept": r["concept"],
                "coverage": round(max((s["score"] for s in snippets), default=0.0), 4),
                "snippets": len(supported),
                "best_label": supported[0]["label"] if supported else "",
                "unsupported": unsupported_terms(r["lesson"], source_text, 8),
            })
        return out

    def unsupported_claims(self, topic: str, concept: str = "",
                           limit: int = 20) -> list[dict]:
        source_text = manifest_text(self.source_manifest(topic))
        if not source_text:
            return []
        where, params = ["c.topic=?"], [topic]
        if concept:
            where.append("c.concept=?")
            params.append(concept)
        rows = self.db.execute(f"""
            SELECT c.topic, c.concept, COALESCE(n.lesson, '') AS lesson
            FROM cards c LEFT JOIN notes n ON n.topic = c.topic AND n.concept = c.concept
            WHERE {' AND '.join(where)} ORDER BY c.concept""", params).fetchall()
        return [{"topic": r["topic"], "concept": r["concept"],
                 "terms": unsupported_terms(r["lesson"], source_text, limit)}
                for r in rows]

    def export_json(self) -> dict:
        return {
            "version": 1,
            "cards": self._table("cards"),
            "attempts": self._table("attempts"),
            "progress": self._table("progress"),
            "notes": self._table("notes", ["topic", "concept", "lesson"]),
            "learner_notes": self._table("learner_notes"),
            "source_docs": self._table("source_docs"),
        }

    def import_json(self, data: dict, merge: bool = False) -> dict:
        if data.get("version") != 1:
            raise ValueError("unsupported export version")
        if not merge:
            with self.db:
                self.db.execute("DELETE FROM attempts")
                self.db.execute("DELETE FROM cards")
                self.db.execute("DELETE FROM progress")
                self.db.execute("DELETE FROM notes")
                self.db.execute("DELETE FROM learner_notes")
                self.db.execute("DELETE FROM source_docs")
        id_map: dict[int, int] = {}
        with self.db:
            for row in data.get("cards", []):
                old_id = row.get("id")
                cols = [c for c in row if c != "id"]
                vals = [row[c] for c in cols]
                placeholders = ",".join("?" for _ in cols)
                self.db.execute(
                    f"INSERT OR REPLACE INTO cards({','.join(cols)}) VALUES({placeholders})",
                    vals)
                new = self.db.execute(
                    "SELECT id FROM cards WHERE topic=? AND concept=?",
                    (row["topic"], row["concept"])).fetchone()
                if old_id is not None:
                    id_map[int(old_id)] = new["id"]
            for row in data.get("attempts", []):
                row = dict(row)
                row.pop("id", None)
                row["card_id"] = id_map.get(int(row["card_id"]), row["card_id"])
                cols = list(row)
                vals = [row[c] for c in cols]
                placeholders = ",".join("?" for _ in cols)
                self.db.execute(
                    f"INSERT INTO attempts({','.join(cols)}) VALUES({placeholders})",
                    vals)
            for row in data.get("progress", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO progress(topic, concepts_json, concept_idx) VALUES(?,?,?)",
                    (row["topic"], row["concepts_json"], row["concept_idx"]))
            for row in data.get("notes", []):
                self.db.execute("INSERT INTO notes(topic, concept, lesson) VALUES(?,?,?)",
                                (row["topic"], row["concept"], row["lesson"]))
            for row in data.get("learner_notes", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO learner_notes(topic, concept, note, ts) VALUES(?,?,?,?)",
                    (row["topic"], row["concept"], row["note"], row["ts"]))
            for row in data.get("source_docs", []):
                self.db.execute(
                    "INSERT OR REPLACE INTO source_docs(topic, name, digest, text, chunks_json, ts) "
                    "VALUES(?,?,?,?,?,?)",
                    (row["topic"], row["name"], row["digest"], row["text"],
                     row["chunks_json"], row["ts"]))
        return {"cards": len(data.get("cards", [])),
                "attempts": len(data.get("attempts", [])),
                "source_docs": len(data.get("source_docs", []))}

    def transcript(self, limit: int = 100, topic: str = "",
                   concept: str = "") -> str:
        lines = ["# Forge Retrieval Transcript", ""]
        for a in reversed(self.attempt_timeline(limit, topic, concept)):
            stamp = dt.datetime.fromtimestamp(a["ts"]).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {stamp} | {a['score']:.2f} | {a['topic']} / {a['concept']}")
        return "\n".join(lines) + "\n"

    def save_progress(self, topic: str, concepts: list, idx: int):
        self.db.execute(
            "INSERT OR REPLACE INTO progress(topic, concepts_json, concept_idx) VALUES(?,?,?)",
            (topic, json.dumps(concepts), idx))
        self.db.commit()

    def load_progress(self, topic: str):
        row = self.db.execute(
            "SELECT concepts_json, concept_idx FROM progress WHERE topic=?", (topic,)).fetchone()
        return (json.loads(row["concepts_json"]), row["concept_idx"]) if row else None

    def clear_progress(self, topic: str):
        self.db.execute("DELETE FROM progress WHERE topic=?", (topic,))
        self.db.commit()

    def stats(self) -> list[sqlite3.Row]:
        return self.db.execute("""
            SELECT c.*, COUNT(a.id) AS n_attempts, AVG(a.score) AS avg_score
            FROM cards c LEFT JOIN attempts a ON a.card_id = c.id
            GROUP BY c.id ORDER BY c.topic, c.id""").fetchall()

    def session_receipt(self, hours: float = 6.0) -> dict:
        """WS-DEMO: printable receipt of the most recent session window.

        A "session" here = attempts landed in the last `hours` hours; the
        product owns no session id, so we use a time window. All data comes
        from existing cards/attempts tables — no schema change.
        """
        now = time.time()
        cutoff = now - hours * 3600
        rows = self.db.execute("""
            SELECT a.ts, a.score, c.topic, c.concept, c.due, c.interval_days,
                   c.reps, c.lapses
            FROM attempts a JOIN cards c ON c.id = a.card_id
            WHERE a.ts >= ? ORDER BY a.ts""", (cutoff,)).fetchall()
        if not rows:
            return {"empty": True, "hours": hours}
        seen: dict[tuple, dict] = {}
        for r in rows:
            k = (r["topic"], r["concept"])
            d = seen.setdefault(k, {"topic": r["topic"], "concept": r["concept"],
                                    "attempts": 0, "scores": [], "mastered": False,
                                    "next_due": r["due"], "reps": r["reps"],
                                    "lapses": r["lapses"]})
            d["attempts"] += 1
            d["scores"].append(r["score"])
        concepts = []
        for d in seen.values():
            avg = sum(d["scores"]) / len(d["scores"])
            best = max(d["scores"])
            d["mastered"] = best >= 0.8
            d["avg_score"] = round(avg, 2)
            d["best_score"] = round(best, 2)
            d["next_due_days"] = round((d["next_due"] - now) / DAY, 1)
            concepts.append(d)
        weakest = min(concepts, key=lambda c: c["best_score"])
        first_ts = rows[0]["ts"]
        last_ts = rows[-1]["ts"]
        return {
            "empty": False,
            "started_at": first_ts,
            "ended_at": last_ts,
            "minutes": round((last_ts - first_ts) / 60, 1),
            "attempted": len(concepts),
            "mastered": sum(1 for c in concepts if c["mastered"]),
            "trapped": sum(1 for c in concepts if not c["mastered"]),
            "concepts": concepts,
            "weakest": {"topic": weakest["topic"], "concept": weakest["concept"],
                        "best_score": weakest["best_score"]},
            "learner": os.environ.get("FORGE_LEARNER", "default"),
        }

    def curve_series(self, topic: str, concept: str,
                     ahead_days: int = 30) -> dict:
        """WS-DEMO: retrievability curve — one point per day between the last
        review and +ahead_days, plus review-reset markers from attempts.
        """
        card = self.db.execute(
            "SELECT * FROM cards WHERE topic=? AND concept=?",
            (topic, concept)).fetchone()
        if not card:
            return {"points": [], "resets": []}
        attempts = self.db.execute(
            "SELECT ts, score FROM attempts WHERE card_id=? ORDER BY ts",
            (card["id"],)).fetchall()
        if not attempts:
            return {"points": [], "resets": []}
        last_ts = attempts[-1]["ts"]
        interval = card["interval_days"]
        due = card["due"]
        end = last_ts + ahead_days * DAY
        step = DAY
        points = []
        t = last_ts
        while t <= end:
            r = retrievability(interval, due, now=t)
            if r is not None:
                points.append({"t": t, "recall": round(r, 4)})
            t += step
        resets = [{"t": a["ts"], "score": a["score"]} for a in attempts]
        return {"topic": topic, "concept": concept,
                "points": points, "resets": resets,
                "current_recall": retrievability(interval, due),
                "interval_days": interval, "due": due}

    def _card_summary(self, row, now: float) -> dict:
        d = _rowdict(row)
        d["avg_score"] = d.get("avg_score") or 0.0
        d["recall"] = retrievability(d["interval_days"], d["due"], now)
        d["due_in_days"] = (d["due"] - now) / DAY
        d["risk"] = risk_score(d, now)
        pin_boost = 0.2 if d.get("pinned") else 0.0
        d["yield_score"] = round(d["risk"] / 2.0 + pin_boost, 4)  # expected strengthening per ~2 min review
        d["reasons"] = risk_reasons(d, now)
        if d.get("pinned"):
            d["reasons"] = ["pinned"] + d["reasons"]
        return d

    def _card(self, topic: str, concept: str) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM cards WHERE topic=? AND concept=?",
                              (topic, concept)).fetchone()
        if not row:
            raise ValueError(f"concept not found: {topic} / {concept}")
        return row

    def _merge_card_rows(self, old_card, target_card):
        self.db.execute("UPDATE attempts SET card_id=? WHERE card_id=?",
                        (target_card["id"], old_card["id"]))
        target_lesson = self.lesson_for(target_card["topic"], target_card["concept"])
        old_lesson = self.lesson_for(old_card["topic"], old_card["concept"])
        if old_lesson and not target_lesson:
            self.db.execute("INSERT INTO notes(topic, concept, lesson) VALUES(?,?,?)",
                            (target_card["topic"], target_card["concept"], old_lesson))
        target_note = self.learner_note(target_card["topic"], target_card["concept"])
        old_note = self.learner_note(old_card["topic"], old_card["concept"])
        if old_note and not target_note:
            self.db.execute(
                "INSERT OR REPLACE INTO learner_notes(topic, concept, note, ts) VALUES(?,?,?,?)",
                (target_card["topic"], target_card["concept"], old_note, time.time()))
        best = target_card["best_answer"] or old_card["best_answer"]
        self.db.execute("""
            UPDATE cards SET ease=MIN(ease, ?), interval_days=MAX(interval_days, ?),
              due=MIN(due, ?), reps=MAX(reps, ?), lapses=lapses + ?,
              suspended=MIN(suspended, ?), retired=MIN(retired, ?), pinned=MAX(pinned, ?),
              best_answer=?
            WHERE id=?""",
            (old_card["ease"], old_card["interval_days"], old_card["due"],
             old_card["reps"], old_card["lapses"], old_card["suspended"],
             old_card["retired"], old_card["pinned"], best, target_card["id"]))
        self.db.execute("DELETE FROM cards WHERE id=?", (old_card["id"],))
        self.db.execute("DELETE FROM notes WHERE topic=? AND concept=?",
                        (old_card["topic"], old_card["concept"]))
        self.db.execute("DELETE FROM learner_notes WHERE topic=? AND concept=?",
                        (old_card["topic"], old_card["concept"]))

    def _orphan_notes(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("""
            SELECT n.topic, n.concept FROM notes n
            LEFT JOIN cards c ON c.topic = n.topic AND c.concept = n.concept
            WHERE c.id IS NULL""")]

    def rebuild_fts(self) -> int:
        rows = self.db.execute("SELECT topic, concept, lesson FROM notes").fetchall()
        with self.db:
            self.db.execute("DELETE FROM notes")
            for r in rows:
                self.db.execute("INSERT INTO notes(topic, concept, lesson) VALUES(?,?,?)",
                                (r["topic"], r["concept"], r["lesson"]))
        return len(rows)

    def fix(self) -> dict:
        """Repair the obvious, reversible-only issues maintenance_report flags:
        orphan attempts/notes and a stale FTS index. Never touches cards/scores."""
        report = self.maintenance_report()
        with self.db:
            for a in report["orphan_attempts"]:
                self.db.execute("DELETE FROM attempts WHERE id=?", (a["id"],))
            for n in report["orphan_notes"]:
                self.db.execute("DELETE FROM notes WHERE topic=? AND concept=?",
                                (n["topic"], n["concept"]))
        rebuilt = self.rebuild_fts()
        if self.path != ":memory:":
            self.db.execute("VACUUM")
        return {"orphan_attempts_removed": len(report["orphan_attempts"]),
                "orphan_notes_removed": len(report["orphan_notes"]),
                "fts_rows_rebuilt": rebuilt}

    def _performance_buckets(self, keyfn) -> list[dict]:
        buckets: dict[str, list[float]] = {}
        for r in self.db.execute("SELECT ts, score FROM attempts"):
            buckets.setdefault(keyfn(r["ts"]), []).append(r["score"])
        rows = [{"bucket": k, "attempts": len(v), "avg_score": sum(v) / len(v)}
                for k, v in buckets.items()]
        rows.sort(key=lambda r: r["bucket"])
        return rows

    def _table(self, name: str, cols: list[str] | None = None) -> list[dict]:
        fields = ", ".join(cols) if cols else "*"
        return [dict(r) for r in self.db.execute(f"SELECT {fields} FROM {name}")]


def _excerpt(text: str, query: str, width: int = 180) -> str:
    words = [w.lower() for w in query.split() if w.isalnum()]
    lower = text.lower()
    pos = min((lower.find(w) for w in words if lower.find(w) >= 0), default=0)
    start = max(0, pos - width // 3)
    snippet = text[start:start + width].replace("\n", " ")
    return snippet.strip()


def _counts_by_day(attempts: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for a in attempts:
        day = dt.date.fromtimestamp(a["ts"]).isoformat()
        out[day] = out.get(day, 0) + 1
    return out


def _buckets(values: list[float], width: float) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        lo = int(v / width) * width
        key = f"{lo:.1f}-{lo + width:.1f}"
        out[key] = out.get(key, 0) + 1
    return out


def _key_points(text: str) -> list[str]:
    seen, points = set(), []
    for w in re_words(text):
        if w in seen or len(w) < 5:
            continue
        seen.add(w)
        points.append(w)
    return points


def _semantic_terms(text: str) -> list[str]:
    return [w.lower() for w in re_words(text) if len(w) >= 4]


def re_words(text: str) -> list[str]:
    import re
    return re.findall(r"[A-Za-z][A-Za-z0-9-]+", text)
