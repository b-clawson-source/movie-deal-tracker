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
from src.letterboxd_scraper import Movie, LetterboxdScraper
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
    max_price = request.form.get("max_price", "20")
    check_frequency = request.form.get("check_frequency", "daily")

    # Parse and validate max_price
    try:
        max_price = float(max_price)
        if max_price < 5:
            max_price = 5
        elif max_price > 100:
            max_price = 100
    except ValueError:
        max_price = 20.0

    # Validate check_frequency
    if check_frequency not in ["daily", "weekly", "monthly"]:
        check_frequency = "daily"

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
    subscriber = db.add_subscriber(email, list_url, max_price, check_frequency)

    if subscriber:
        logger.info(f"New subscriber: {email} -> {list_url} (max: ${max_price}, freq: {check_frequency})")
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
    max_price = 100.0
    error = None
    searched = False

    if request.method == "POST":
        movie_title = request.form.get("movie_title", "").strip()
        max_price_str = request.form.get("max_price", "100")
        searched = True

        # Parse max_price
        try:
            max_price = float(max_price_str)
            if max_price < 5:
                max_price = 5
            elif max_price > 200:
                max_price = 200
        except ValueError:
            max_price = 100.0

        if not movie_title:
            error = "Please enter a movie title or Letterboxd URL"
        else:
            # Check for SerpAPI key
            serpapi_key = os.getenv("SERPAPI_KEY")
            if not serpapi_key:
                error = "Search is temporarily unavailable"
                logger.error("SERPAPI_KEY not configured")
            else:
                try:
                    # Check if input is a Letterboxd film URL
                    letterboxd_match = re.match(r'^https?://letterboxd\.com/film/([^/]+)/?$', movie_title)

                    if letterboxd_match:
                        # Fetch movie details from Letterboxd
                        scraper = LetterboxdScraper()
                        movie = Movie(title="", letterboxd_url=movie_title)
                        scraper.fetch_movie_details(movie)

                        # Extract title from URL slug if fetch failed
                        if not movie.title:
                            slug = letterboxd_match.group(1)
                            movie.title = slug.replace("-", " ").title()

                        logger.info(f"Letterboxd lookup: {movie}")
                    else:
                        # Parse year from title if provided (e.g., "The Thing (1982)" or "The Thing 1982")
                        title = movie_title
                        year = None
                        year_match = re.search(r'\s*\(?(\d{4})\)?$', movie_title)
                        if year_match:
                            year = int(year_match.group(1))
                            title = movie_title[:year_match.start()].strip()

                        # Create a Movie object
                        movie = Movie(title=title, year=year)

                    # Initialize classifier and deal finder
                    classifier = EditionClassifier()
                    finder = DealFinder(
                        api_key=serpapi_key,
                        classifier=classifier,
                        max_price=max_price,
                        requests_per_minute=30,
                    )

                    # Search for deals
                    deals = finder.search_movie(movie)
                    search_desc = str(movie)
                    logger.info(f"Search for {search_desc} (max ${max_price}) found {len(deals)} deals")

                except Exception as e:
                    logger.error(f"Search failed: {e}")
                    error = "Search failed. Please try again."

    return render_template(
        "search.html",
        deals=deals,
        movie_title=movie_title,
        max_price=max_price,
        error=error,
        searched=searched,
    )


@app.route("/health")
def health():
    """Health check endpoint for deployment platforms."""
    return {"status": "ok"}


@app.route("/admin/cache-status")
def cache_status():
    """Get cache status and sale period info."""
    from src.sale_periods import get_cache_status

    # Verify admin key
    admin_key = os.getenv("ADMIN_KEY", "")
    provided_key = request.args.get("key", "") or request.headers.get("X-Admin-Key", "")

    if not admin_key or provided_key != admin_key:
        return {"error": "Unauthorized"}, 401

    db = get_db()
    cache_stats = db.get_cache_stats()
    sale_status = get_cache_status()

    return {
        "cache": cache_stats,
        "sale_period": sale_status,
    }


@app.route("/admin/debug-search", methods=["GET"])
def debug_search():
    """Debug search to see raw results before filtering."""
    from src.edition_classifier import EditionClassifier
    from src.deal_finder import DealFinder

    # Verify admin key
    admin_key = os.getenv("ADMIN_KEY", "")
    provided_key = request.args.get("key", "") or request.headers.get("X-Admin-Key", "")

    if not admin_key or provided_key != admin_key:
        return {"error": "Unauthorized"}, 401

    movie_title = request.args.get("title", "House")
    movie_year = request.args.get("year", "1977")
    max_price = float(request.args.get("max_price", "100"))

    try:
        year = int(movie_year) if movie_year else None
    except:
        year = None

    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        return {"error": "SERPAPI_KEY not configured"}, 500

    # Create movie and build query
    movie = Movie(title=movie_title, year=year)
    classifier = EditionClassifier()
    finder = DealFinder(
        api_key=serpapi_key,
        classifier=classifier,
        max_price=max_price,
        requests_per_minute=30,
    )

    query = finder._build_query(movie)

    # Execute raw search
    try:
        raw_results = finder._execute_search(query)
        shopping_results = raw_results.get("shopping_results", [])
    except Exception as e:
        return {"error": str(e), "query": query}, 500

    # Analyze each result
    analysis = []
    for item in shopping_results[:10]:  # First 10 results
        title = item.get("title", "")
        price_str = item.get("price", "")
        source = item.get("source", "")

        # Extract price
        price = finder._extract_price(price_str)

        # Check edition
        is_special, confidence, edition_type = classifier.is_special_edition(title)

        # Check year
        year_valid = True
        if year:
            year_valid = finder._validate_year(title, year)

        analysis.append({
            "title": title,
            "price": price,
            "price_str": price_str,
            "source": source,
            "is_special_edition": is_special,
            "edition_type": edition_type,
            "year_valid": year_valid,
            "would_include": is_special and year_valid and price is not None and price <= max_price and "ebay" not in source.lower()
        })

    return {
        "query": query,
        "movie": {"title": movie_title, "year": year},
        "max_price": max_price,
        "total_raw_results": len(shopping_results),
        "analysis": analysis
    }


@app.route("/admin/clear-cache", methods=["POST"])
def clear_cache():
    """Clear the search cache."""
    # Verify admin key
    admin_key = os.getenv("ADMIN_KEY", "")
    provided_key = request.args.get("key", "") or request.headers.get("X-Admin-Key", "")

    if not admin_key or provided_key != admin_key:
        return {"error": "Unauthorized"}, 401

    db = get_db()
    deleted = db.clear_all_cache()

    logger.info(f"Cache cleared via admin endpoint ({deleted} entries)")
    return {"status": "ok", "entries_cleared": deleted}


@app.route("/admin/run-check", methods=["POST"])
def admin_run_check():
    """Manually trigger a deal check for all subscribers (runs in background)."""
    import threading

    # Verify admin key
    admin_key = os.getenv("ADMIN_KEY", "")
    provided_key = request.args.get("key", "") or request.headers.get("X-Admin-Key", "")

    if not admin_key or provided_key != admin_key:
        return {"error": "Unauthorized"}, 401

    # Check for force flag to bypass frequency check
    force = request.args.get("force", "").lower() == "true"
    # Check for resend flag to send all deals (not just new ones)
    resend = request.args.get("resend", "").lower() == "true"

    def run_check():
        try:
            from src.job_runner import JobRunner
            logger.info(f"Background deal check started (force={force}, resend={resend})")
            runner = JobRunner()
            runner.run_all_subscribers(force=force, resend=resend)
            logger.info("Background deal check completed")
        except Exception as e:
            logger.error(f"Background deal check failed: {e}")

    # Start in background thread
    thread = threading.Thread(target=run_check, daemon=True)
    thread.start()

    logger.info(f"Manual deal check triggered via admin endpoint (force={force}, resend={resend})")
    return {"status": "ok", "message": f"Deal check started (force={force}, resend={resend}). Check logs."}


if __name__ == "__main__":
    # Development server
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logger.info(f"Starting development server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
