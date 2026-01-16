#!/usr/bin/env python3
"""
Movie Deal Tracker - Main Entry Point

Monitors Letterboxd lists for special physical movie editions on sale.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports for heavy dependencies
# These are imported inside functions that need them
# to allow --list-movies to work without all deps

# Setup logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # File handler
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "tracker.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def load_config() -> dict:
    """Load configuration from YAML and environment."""
    config_path = Path(__file__).parent / "config" / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load environment variables
    load_dotenv()

    # Inject secrets from environment
    config["search"]["serpapi_key"] = os.getenv("SERPAPI_KEY", "")
    config["email"]["sender"] = os.getenv("EMAIL_SENDER", "")
    config["email"]["password"] = os.getenv("EMAIL_PASSWORD", "")
    config["email"]["recipient"] = os.getenv("EMAIL_RECIPIENT", "")

    return config


def validate_config(config: dict) -> bool:
    """Validate required configuration."""
    errors = []

    if not config["search"]["serpapi_key"]:
        errors.append("SERPAPI_KEY not set in environment")

    if not config["email"]["sender"]:
        errors.append("EMAIL_SENDER not set in environment")

    if not config["email"]["password"]:
        errors.append("EMAIL_PASSWORD not set in environment")

    if not config["email"]["recipient"]:
        errors.append("EMAIL_RECIPIENT not set in environment")

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nCopy .env.example to .env and fill in your credentials.")
        return False

    return True


def create_components(config: dict):
    """Create all application components."""
    from src.edition_matcher import EditionMatcher
    from src.deal_finder import DealFinder, DealTracker
    from src.notifier import EmailNotifier

    base_path = Path(__file__).parent

    # Edition matcher
    matcher = EditionMatcher(
        model_name=config["matching"]["model"],
        examples_path=base_path / "config" / "examples.yaml",
        similarity_threshold=config["matching"]["similarity_threshold"],
    )

    # Deal finder
    finder = DealFinder(
        api_key=config["search"]["serpapi_key"],
        matcher=matcher,
        max_price=config["search"]["max_price"],
        requests_per_minute=config["search"]["requests_per_minute"],
    )

    # Deal tracker
    tracker = DealTracker(base_path / "data" / "found_deals.json")

    # Email notifier
    notifier = EmailNotifier(
        smtp_server=config["email"]["smtp_server"],
        smtp_port=config["email"]["smtp_port"],
        sender_email=config["email"]["sender"],
        sender_password=config["email"]["password"],
        recipient_email=config["email"]["recipient"],
    )

    return matcher, finder, tracker, notifier


def run_deal_check(config: dict):
    """Run a single deal check."""
    from src.letterboxd_scraper import get_movies_from_list

    logger = logging.getLogger(__name__)
    logger.info("Starting deal check...")

    # Create components
    matcher, finder, tracker, notifier = create_components(config)

    # Get movies from Letterboxd
    list_url = config["letterboxd"]["list_url"]
    logger.info(f"Fetching movies from: {list_url}")
    movies = get_movies_from_list(list_url)
    logger.info(f"Found {len(movies)} movies in list")

    if not movies:
        logger.warning("No movies found in list!")
        return

    # Search for deals
    logger.info("Searching for deals...")
    all_deals = finder.find_deals(movies)
    logger.info(f"Found {len(all_deals)} total deals")

    # Filter to new deals only
    new_deals = tracker.filter_new_deals(all_deals)
    logger.info(f"New deals (not seen recently): {len(new_deals)}")

    # Send notification if we have new deals
    if new_deals:
        logger.info("Sending notification...")
        success = notifier.send_deals(new_deals)
        if success:
            logger.info("Notification sent successfully!")
        else:
            logger.error("Failed to send notification")
    else:
        logger.info("No new deals to notify")

    logger.info("Deal check complete!")


def list_movies(config: dict):
    """List movies from Letterboxd list."""
    from src.letterboxd_scraper import get_movies_from_list

    list_url = config["letterboxd"]["list_url"]
    print(f"Fetching movies from: {list_url}\n")

    movies = get_movies_from_list(list_url)

    print(f"Found {len(movies)} movies:\n")
    for i, movie in enumerate(movies, 1):
        print(f"{i:3}. {movie}")


def test_email(config: dict):
    """Send a test email."""
    from src.notifier import EmailNotifier

    print("Sending test email...")

    notifier = EmailNotifier(
        smtp_server=config["email"]["smtp_server"],
        smtp_port=config["email"]["smtp_port"],
        sender_email=config["email"]["sender"],
        sender_password=config["email"]["password"],
        recipient_email=config["email"]["recipient"],
    )

    success = notifier.send_test()
    if success:
        print("Test email sent successfully!")
    else:
        print("Failed to send test email. Check your credentials.")
        sys.exit(1)


def start_scheduler(config: dict):
    """Start the scheduler daemon."""
    from src.scheduler import create_scheduler

    logger = logging.getLogger(__name__)

    def job():
        run_deal_check(config)

    scheduler = create_scheduler(
        job_func=job,
        run_at=config["schedule"]["run_at"],
    )

    logger.info("Starting scheduler daemon...")
    scheduler.start(run_immediately=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Movie Deal Tracker - Find special edition movies on sale"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run as scheduler daemon",
    )
    parser.add_argument(
        "--list-movies",
        action="store_true",
        help="List movies from your Letterboxd list",
    )
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Send a test email notification",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Load config
    try:
        config = load_config()
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    # Handle --list-movies (doesn't require full config)
    if args.list_movies:
        list_movies(config)
        return

    # Validate config for other operations
    if not validate_config(config):
        sys.exit(1)

    # Handle commands
    if args.test_email:
        test_email(config)
    elif args.schedule:
        start_scheduler(config)
    else:
        # Default: run once
        run_deal_check(config)


if __name__ == "__main__":
    main()
