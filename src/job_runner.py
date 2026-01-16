"""
Job runner for processing all subscribers.
Handles the multi-user deal checking workflow.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv

from .database import get_db, Subscriber
from .letterboxd_scraper import get_movies_from_list
from .edition_matcher import EditionMatcher
from .deal_finder import DealFinder, Deal
from .notifier import EmailNotifier

logger = logging.getLogger(__name__)

# Load environment
load_dotenv()


class JobRunner:
    """Runs deal checks for all subscribers."""

    def __init__(self):
        self.config = self._load_config()
        self.matcher = self._create_matcher()
        self.finder = self._create_finder()
        self.notifier = self._create_notifier()
        self.db = get_db()

    def _load_config(self) -> dict:
        """Load configuration."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def _create_matcher(self) -> EditionMatcher:
        """Create edition matcher."""
        base_path = Path(__file__).parent.parent
        return EditionMatcher(
            model_name=self.config["matching"]["model"],
            examples_path=base_path / "config" / "examples.yaml",
            similarity_threshold=self.config["matching"]["similarity_threshold"],
        )

    def _create_finder(self) -> DealFinder:
        """Create deal finder."""
        api_key = os.getenv("SERPAPI_KEY", "")
        if not api_key:
            raise ValueError("SERPAPI_KEY not set in environment")

        return DealFinder(
            api_key=api_key,
            matcher=self.matcher,
            max_price=self.config["search"]["max_price"],
            requests_per_minute=self.config["search"]["requests_per_minute"],
        )

    def _create_notifier(self) -> EmailNotifier:
        """Create email notifier."""
        api_key = os.getenv("RESEND_API_KEY", "")
        if not api_key:
            raise ValueError("RESEND_API_KEY not set in environment")

        from_email = os.getenv("EMAIL_FROM", "Movie Deal Tracker <deals@resend.dev>")

        return EmailNotifier(api_key=api_key, from_email=from_email)

    def run_all_subscribers(self):
        """Process all active subscribers."""
        subscribers = self.db.get_active_subscribers()
        logger.info(f"Processing {len(subscribers)} active subscribers")

        for subscriber in subscribers:
            try:
                self._process_subscriber(subscriber)
            except Exception as e:
                logger.error(f"Failed to process subscriber {subscriber.email}: {e}")

        logger.info("Finished processing all subscribers")

    def _process_subscriber(self, subscriber: Subscriber):
        """Process a single subscriber."""
        logger.info(f"Processing subscriber: {subscriber.email}")

        # Get movies from their list
        try:
            movies = get_movies_from_list(subscriber.list_url)
            logger.info(f"Found {len(movies)} movies in list for {subscriber.email}")
        except Exception as e:
            logger.error(f"Failed to scrape list for {subscriber.email}: {e}")
            return

        if not movies:
            logger.warning(f"No movies found in list for {subscriber.email}")
            return

        # Search for deals
        all_deals = self.finder.find_deals(movies)
        logger.info(f"Found {len(all_deals)} total deals for {subscriber.email}")

        # Filter to new deals for this subscriber
        new_deals = self.db.filter_new_deals(subscriber.id, all_deals)
        logger.info(f"New deals for {subscriber.email}: {len(new_deals)}")

        # Send notification if we have new deals
        if new_deals:
            self._send_notification(subscriber, new_deals)

        # Update last checked timestamp
        self.db.update_last_checked(subscriber.id)

    def _send_notification(self, subscriber: Subscriber, deals: list):
        """Send deal notification to subscriber."""
        logger.info(f"Sending notification to {subscriber.email} with {len(deals)} deals")

        # Get base URL for unsubscribe link
        base_url = os.getenv("BASE_URL", "http://localhost:5000")
        unsubscribe_url = f"{base_url}/unsubscribe/{subscriber.unsubscribe_token}"

        success = self.notifier.send_deals_to(
            recipient_email=subscriber.email,
            deals=deals,
            unsubscribe_url=unsubscribe_url,
        )

        if success:
            logger.info(f"Notification sent to {subscriber.email}")
        else:
            logger.error(f"Failed to send notification to {subscriber.email}")


def run_job():
    """Entry point for running the job."""
    logger.info("Starting deal check job...")
    runner = JobRunner()
    runner.run_all_subscribers()
    logger.info("Deal check job complete!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    run_job()
