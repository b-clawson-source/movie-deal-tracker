#!/usr/bin/env python3
"""
Movie Deal Tracker - Web Application

A simple Flask app that allows users to subscribe to deal notifications
by entering their Letterboxd list URL and email address.
"""

import os
import re
import logging
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from dotenv import load_dotenv

# Import deal finding components
from src.letterboxd_scraper import Movie
from src.edition_classifier import EditionClassifier
from src.deal_finder import DealFinder

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Import database after app setup
from src.database import get_db


def is_valid_letterboxd_url(url: str) -> bool:
    """Validate that a URL is a valid Letterboxd list URL."""
    pattern = r"^https?://letterboxd\.com/[\w_-]+/list/[\w_-]+/?$"
    return bool(re.match(pattern, url, re.IGNORECASE))


def is_valid_email(email: str) -> bool:
    """Basic email validation."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


@app.route("/")
def index():
    """Landing page with subscription form."""
    db = get_db()
    subscriber_count = db.get_subscriber_count()
    return render_template("index.html", subscriber_count=subscriber_count)


@app.route("/subscribe", methods=["POST"])
def subscribe():
    """Handle new subscription."""
    email = request.form.get("email", "").strip().lower()
    list_url = request.form.get("list_url", "").strip()

    # Validate inputs
    errors = []

    if not email:
        errors.append("Email is required")
    elif not is_valid_email(email):
        errors.append("Please enter a valid email address")

    if not list_url:
        errors.append("Letterboxd list URL is required")
    elif not is_valid_letterboxd_url(list_url):
        errors.append("Please enter a valid Letterboxd list URL (e.g., https://letterboxd.com/username/list/list-name/)")

    if errors:
        for error in errors:
            flash(error, "error")
        return redirect(url_for("index"))

    # Add subscriber to database
    db = get_db()
    subscriber = db.add_subscriber(email, list_url)

    if subscriber:
        logger.info(f"New subscriber: {email} -> {list_url}")
        return render_template("success.html", email=email)
    else:
        flash("Something went wrong. Please try again.", "error")
        return redirect(url_for("index"))


@app.route("/unsubscribe/<token>")
def unsubscribe(token: str):
    """Handle unsubscribe request."""
    db = get_db()

    # Get subscriber info before unsubscribing
    subscriber = db.get_subscriber_by_token(token)

    if subscriber:
        success = db.unsubscribe(token)
        if success:
            logger.info(f"Unsubscribed: {subscriber.email}")
            return render_template("unsubscribed.html", email=subscriber.email)

    # Invalid or expired token
    return render_template("unsubscribed.html", email=None, error=True)


@app.route("/search", methods=["GET", "POST"])
def search():
    """On-demand movie search page."""
    deals = []
    movie_title = ""
    error = None
    searched = False

    if request.method == "POST":
        movie_title = request.form.get("movie_title", "").strip()
        searched = True

        if not movie_title:
            error = "Please enter a movie title"
        else:
            # Check for SerpAPI key
            serpapi_key = os.getenv("SERPAPI_KEY")
            if not serpapi_key:
                error = "Search is temporarily unavailable"
                logger.error("SERPAPI_KEY not configured")
            else:
                try:
                    # Create a Movie object from the title
                    movie = Movie(title=movie_title)

                    # Initialize classifier and deal finder
                    classifier = EditionClassifier()
                    finder = DealFinder(
                        api_key=serpapi_key,
                        classifier=classifier,
                        max_price=100.0,  # Show more results for on-demand search
                        requests_per_minute=30,
                    )

                    # Search for deals
                    deals = finder.search_movie(movie)
                    logger.info(f"Search for '{movie_title}' found {len(deals)} deals")

                except Exception as e:
                    logger.error(f"Search failed: {e}")
                    error = "Search failed. Please try again."

    return render_template(
        "search.html",
        deals=deals,
        movie_title=movie_title,
        error=error,
        searched=searched,
    )


@app.route("/health")
def health():
    """Health check endpoint for deployment platforms."""
    return {"status": "ok"}


if __name__ == "__main__":
    # Development server
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting development server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
