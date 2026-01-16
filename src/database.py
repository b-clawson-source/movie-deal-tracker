"""
Database layer for subscriber management.
Uses SQLite for simplicity and easy deployment.
"""

import sqlite3
import secrets
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Subscriber:
    """Represents a subscriber."""
    id: int
    email: str
    list_url: str
    created_at: str
    last_checked: Optional[str]
    unsubscribe_token: str
    active: bool


class Database:
    """SQLite database for managing subscribers and deal history."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "data" / "subscribers.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscribers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    list_url TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_checked TEXT,
                    unsubscribe_token TEXT UNIQUE NOT NULL,
                    active INTEGER DEFAULT 1
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS notified_deals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subscriber_id INTEGER NOT NULL,
                    deal_hash TEXT NOT NULL,
                    notified_at TEXT NOT NULL,
                    FOREIGN KEY (subscriber_id) REFERENCES subscribers(id),
                    UNIQUE(subscriber_id, deal_hash)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subscribers_active
                ON subscribers(active)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_notified_deals_subscriber
                ON notified_deals(subscriber_id)
            """)

            conn.commit()

    def add_subscriber(self, email: str, list_url: str) -> Optional[Subscriber]:
        """Add a new subscriber or update existing one."""
        token = secrets.token_urlsafe(32)
        now = datetime.now().isoformat()

        try:
            with self._get_connection() as conn:
                # Check if subscriber exists
                existing = conn.execute(
                    "SELECT * FROM subscribers WHERE email = ?",
                    (email,)
                ).fetchone()

                if existing:
                    # Update existing subscriber
                    conn.execute("""
                        UPDATE subscribers
                        SET list_url = ?, active = 1
                        WHERE email = ?
                    """, (list_url, email))
                    conn.commit()

                    return self.get_subscriber_by_email(email)
                else:
                    # Insert new subscriber
                    conn.execute("""
                        INSERT INTO subscribers (email, list_url, created_at, unsubscribe_token, active)
                        VALUES (?, ?, ?, ?, 1)
                    """, (email, list_url, now, token))
                    conn.commit()

                    return self.get_subscriber_by_email(email)

        except sqlite3.Error as e:
            logger.error(f"Failed to add subscriber: {e}")
            return None

    def get_subscriber_by_email(self, email: str) -> Optional[Subscriber]:
        """Get subscriber by email."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM subscribers WHERE email = ?",
                (email,)
            ).fetchone()

            if row:
                return Subscriber(
                    id=row["id"],
                    email=row["email"],
                    list_url=row["list_url"],
                    created_at=row["created_at"],
                    last_checked=row["last_checked"],
                    unsubscribe_token=row["unsubscribe_token"],
                    active=bool(row["active"]),
                )
            return None

    def get_subscriber_by_token(self, token: str) -> Optional[Subscriber]:
        """Get subscriber by unsubscribe token."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM subscribers WHERE unsubscribe_token = ?",
                (token,)
            ).fetchone()

            if row:
                return Subscriber(
                    id=row["id"],
                    email=row["email"],
                    list_url=row["list_url"],
                    created_at=row["created_at"],
                    last_checked=row["last_checked"],
                    unsubscribe_token=row["unsubscribe_token"],
                    active=bool(row["active"]),
                )
            return None

    def get_active_subscribers(self) -> List[Subscriber]:
        """Get all active subscribers."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM subscribers WHERE active = 1"
            ).fetchall()

            return [
                Subscriber(
                    id=row["id"],
                    email=row["email"],
                    list_url=row["list_url"],
                    created_at=row["created_at"],
                    last_checked=row["last_checked"],
                    unsubscribe_token=row["unsubscribe_token"],
                    active=bool(row["active"]),
                )
                for row in rows
            ]

    def unsubscribe(self, token: str) -> bool:
        """Unsubscribe a user by their token."""
        try:
            with self._get_connection() as conn:
                result = conn.execute("""
                    UPDATE subscribers
                    SET active = 0
                    WHERE unsubscribe_token = ?
                """, (token,))
                conn.commit()
                return result.rowcount > 0
        except sqlite3.Error as e:
            logger.error(f"Failed to unsubscribe: {e}")
            return False

    def update_last_checked(self, subscriber_id: int):
        """Update the last_checked timestamp for a subscriber."""
        now = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    UPDATE subscribers
                    SET last_checked = ?
                    WHERE id = ?
                """, (now, subscriber_id))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to update last_checked: {e}")

    def is_deal_notified(self, subscriber_id: int, deal_hash: str) -> bool:
        """Check if a deal has already been notified to this subscriber."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT 1 FROM notified_deals
                WHERE subscriber_id = ? AND deal_hash = ?
            """, (subscriber_id, deal_hash)).fetchone()
            return row is not None

    def mark_deal_notified(self, subscriber_id: int, deal_hash: str):
        """Mark a deal as notified for a subscriber."""
        now = datetime.now().isoformat()
        try:
            with self._get_connection() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO notified_deals (subscriber_id, deal_hash, notified_at)
                    VALUES (?, ?, ?)
                """, (subscriber_id, deal_hash, now))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to mark deal notified: {e}")

    def filter_new_deals(self, subscriber_id: int, deals: list) -> list:
        """Filter deals to only those not yet notified to this subscriber."""
        new_deals = []
        for deal in deals:
            if not self.is_deal_notified(subscriber_id, deal.deal_hash):
                new_deals.append(deal)
                self.mark_deal_notified(subscriber_id, deal.deal_hash)
        return new_deals

    def get_subscriber_count(self) -> int:
        """Get total count of active subscribers."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM subscribers WHERE active = 1"
            ).fetchone()
            return row["count"] if row else 0


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db
