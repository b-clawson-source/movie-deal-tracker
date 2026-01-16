"""
Database layer for subscriber management.
Supports PostgreSQL (production) and SQLite (local development).
"""

import os
import sqlite3
import secrets
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Check if we're using PostgreSQL
DATABASE_URL = os.getenv("DATABASE_URL")
USE_POSTGRES = DATABASE_URL is not None

if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor


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
    """Database for managing subscribers and deal history."""

    def __init__(self, db_path: Optional[Path] = None):
        self.use_postgres = USE_POSTGRES

        if self.use_postgres:
            self.database_url = DATABASE_URL
            logger.info("Using PostgreSQL database")
        else:
            if db_path is None:
                db_path = Path(__file__).parent.parent / "data" / "subscribers.db"
            self.db_path = db_path
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using SQLite database at {self.db_path}")

        self._init_db()

    def _get_connection(self):
        """Get a database connection."""
        if self.use_postgres:
            conn = psycopg2.connect(self.database_url)
            return conn
        else:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn

    def _placeholder(self, index: int = 1) -> str:
        """Return the appropriate placeholder for the database type."""
        return "%s" if self.use_postgres else "?"

    def _init_db(self):
        """Initialize database tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            if self.use_postgres:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS subscribers (
                        id SERIAL PRIMARY KEY,
                        email TEXT UNIQUE NOT NULL,
                        list_url TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        last_checked TEXT,
                        unsubscribe_token TEXT UNIQUE NOT NULL,
                        active INTEGER DEFAULT 1
                    )
                """)

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notified_deals (
                        id SERIAL PRIMARY KEY,
                        subscriber_id INTEGER NOT NULL REFERENCES subscribers(id),
                        deal_hash TEXT NOT NULL,
                        notified_at TEXT NOT NULL,
                        UNIQUE(subscriber_id, deal_hash)
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_subscribers_active
                    ON subscribers(active)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_notified_deals_subscriber
                    ON notified_deals(subscriber_id)
                """)
            else:
                cursor.execute("""
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

                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS notified_deals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        subscriber_id INTEGER NOT NULL,
                        deal_hash TEXT NOT NULL,
                        notified_at TEXT NOT NULL,
                        FOREIGN KEY (subscriber_id) REFERENCES subscribers(id),
                        UNIQUE(subscriber_id, deal_hash)
                    )
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_subscribers_active
                    ON subscribers(active)
                """)

                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_notified_deals_subscriber
                    ON notified_deals(subscriber_id)
                """)

            conn.commit()
        finally:
            conn.close()

    def _row_to_subscriber(self, row) -> Subscriber:
        """Convert a database row to a Subscriber object."""
        if self.use_postgres:
            return Subscriber(
                id=row[0],
                email=row[1],
                list_url=row[2],
                created_at=row[3],
                last_checked=row[4],
                unsubscribe_token=row[5],
                active=bool(row[6]),
            )
        else:
            return Subscriber(
                id=row["id"],
                email=row["email"],
                list_url=row["list_url"],
                created_at=row["created_at"],
                last_checked=row["last_checked"],
                unsubscribe_token=row["unsubscribe_token"],
                active=bool(row["active"]),
            )

    def add_subscriber(self, email: str, list_url: str) -> Optional[Subscriber]:
        """Add a new subscriber or update existing one."""
        token = secrets.token_urlsafe(32)
        now = datetime.now().isoformat()
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Check if subscriber exists
            cursor.execute(
                f"SELECT * FROM subscribers WHERE email = {p}",
                (email,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing subscriber
                cursor.execute(f"""
                    UPDATE subscribers
                    SET list_url = {p}, active = 1
                    WHERE email = {p}
                """, (list_url, email))
                conn.commit()
            else:
                # Insert new subscriber
                cursor.execute(f"""
                    INSERT INTO subscribers (email, list_url, created_at, unsubscribe_token, active)
                    VALUES ({p}, {p}, {p}, {p}, 1)
                """, (email, list_url, now, token))
                conn.commit()

            return self.get_subscriber_by_email(email)

        except Exception as e:
            logger.error(f"Failed to add subscriber: {e}")
            conn.rollback()
            return None
        finally:
            conn.close()

    def get_subscriber_by_email(self, email: str) -> Optional[Subscriber]:
        """Get subscriber by email."""
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM subscribers WHERE email = {p}",
                (email,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_subscriber(row)
            return None
        finally:
            conn.close()

    def get_subscriber_by_token(self, token: str) -> Optional[Subscriber]:
        """Get subscriber by unsubscribe token."""
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM subscribers WHERE unsubscribe_token = {p}",
                (token,)
            )
            row = cursor.fetchone()

            if row:
                return self._row_to_subscriber(row)
            return None
        finally:
            conn.close()

    def get_active_subscribers(self) -> List[Subscriber]:
        """Get all active subscribers."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM subscribers WHERE active = 1")
            rows = cursor.fetchall()

            return [self._row_to_subscriber(row) for row in rows]
        finally:
            conn.close()

    def unsubscribe(self, token: str) -> bool:
        """Unsubscribe a user by their token."""
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE subscribers
                SET active = 0
                WHERE unsubscribe_token = {p}
            """, (token,))
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to unsubscribe: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def update_last_checked(self, subscriber_id: int):
        """Update the last_checked timestamp for a subscriber."""
        now = datetime.now().isoformat()
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE subscribers
                SET last_checked = {p}
                WHERE id = {p}
            """, (now, subscriber_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to update last_checked: {e}")
            conn.rollback()
        finally:
            conn.close()

    def is_deal_notified(self, subscriber_id: int, deal_hash: str) -> bool:
        """Check if a deal has already been notified to this subscriber."""
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT 1 FROM notified_deals
                WHERE subscriber_id = {p} AND deal_hash = {p}
            """, (subscriber_id, deal_hash))
            row = cursor.fetchone()
            return row is not None
        finally:
            conn.close()

    def mark_deal_notified(self, subscriber_id: int, deal_hash: str):
        """Mark a deal as notified for a subscriber."""
        now = datetime.now().isoformat()
        p = "%s" if self.use_postgres else "?"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if self.use_postgres:
                cursor.execute(f"""
                    INSERT INTO notified_deals (subscriber_id, deal_hash, notified_at)
                    VALUES ({p}, {p}, {p})
                    ON CONFLICT (subscriber_id, deal_hash) DO NOTHING
                """, (subscriber_id, deal_hash, now))
            else:
                cursor.execute(f"""
                    INSERT OR IGNORE INTO notified_deals (subscriber_id, deal_hash, notified_at)
                    VALUES ({p}, {p}, {p})
                """, (subscriber_id, deal_hash, now))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark deal notified: {e}")
            conn.rollback()
        finally:
            conn.close()

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
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM subscribers WHERE active = 1")
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            conn.close()


# Global database instance
_db: Optional[Database] = None


def get_db() -> Database:
    """Get or create the global database instance."""
    global _db
    if _db is None:
        _db = Database()
    return _db
