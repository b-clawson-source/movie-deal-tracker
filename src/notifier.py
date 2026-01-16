"""
Email notification system for deal alerts.
"""

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List
from datetime import datetime

from .deal_finder import Deal

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Sends email notifications for found deals."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str,
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email

    def send_deals(self, deals: List[Deal]) -> bool:
        """Send email notification with deal summary."""
        if not deals:
            logger.info("No deals to notify")
            return True

        subject = f"Movie Deal Alert: {len(deals)} special edition(s) under $20!"
        body = self._format_email_body(deals)

        return self._send_email(subject, body)

    def send_deals_to(
        self,
        recipient_email: str,
        deals: List[Deal],
        unsubscribe_url: str = ""
    ) -> bool:
        """Send email notification to a specific recipient with unsubscribe link."""
        if not deals:
            logger.info("No deals to notify")
            return True

        subject = f"Movie Deal Alert: {len(deals)} special edition(s) under $20!"
        body = self._format_email_body(deals, unsubscribe_url=unsubscribe_url)

        return self._send_email(subject, body, recipient_email=recipient_email)

    def send_test(self) -> bool:
        """Send a test email to verify configuration."""
        subject = "Movie Deal Tracker - Test Email"
        body = """
        <html>
        <body>
        <h2>Test Email</h2>
        <p>Your Movie Deal Tracker email notifications are configured correctly!</p>
        <p>You will receive alerts when special editions of movies from your
        Letterboxd list are found at $20 or below.</p>
        <p><em>Sent at: {}</em></p>
        </body>
        </html>
        """.format(
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        return self._send_email(subject, body)

    def _format_email_body(self, deals: List[Deal], unsubscribe_url: str = "") -> str:
        """Format deals into HTML email body."""
        deals_html = ""

        for deal in deals:
            deals_html += f"""
            <div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 8px;">
                <h3 style="margin-top: 0; color: #333;">{deal.movie_title}</h3>
                <p><strong>Edition:</strong> {deal.product_title}</p>
                <p><strong>Price:</strong> <span style="color: #28a745; font-size: 1.2em;">${deal.price:.2f}</span></p>
                <p><strong>Retailer:</strong> {deal.retailer}</p>
                <p><strong>Match Score:</strong> {deal.similarity_score:.1%}</p>
                <p><a href="{deal.url}" style="background-color: #007bff; color: white; padding: 8px 16px; text-decoration: none; border-radius: 4px;">View Deal</a></p>
            </div>
            """

        # Build unsubscribe footer
        unsubscribe_html = ""
        if unsubscribe_url:
            unsubscribe_html = f"""
            <p style="margin-top: 10px;">
                <a href="{unsubscribe_url}" style="color: #999; font-size: 0.8em;">Unsubscribe from these alerts</a>
            </p>
            """

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px;">
                Movie Deal Alert
            </h1>
            <p>Found <strong>{len(deals)}</strong> special edition(s) under $20!</p>
            {deals_html}
            <hr style="margin-top: 30px;">
            <p style="color: #666; font-size: 0.9em;">
                This alert was sent by Movie Deal Tracker.<br>
                Monitoring your Letterboxd list for collector's editions.
            </p>
            {unsubscribe_html}
        </body>
        </html>
        """

        return body

    def _send_email(self, subject: str, body: str, recipient_email: str = "") -> bool:
        """Send an email via SMTP."""
        # Use provided recipient or fall back to instance recipient
        to_email = recipient_email or self.recipient_email

        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.sender_email
            msg["To"] = to_email

            # Attach HTML body
            html_part = MIMEText(body, "html")
            msg.attach(html_part)

            # Connect and send
            logger.info(f"Connecting to {self.smtp_server}:{self.smtp_port}")
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {e}")
            logger.error(
                "If using Gmail, make sure to use an App Password, "
                "not your regular password"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


def create_notifier(
    smtp_server: str,
    smtp_port: int,
    sender_email: str,
    sender_password: str,
    recipient_email: str,
) -> EmailNotifier:
    """Factory function to create an EmailNotifier."""
    return EmailNotifier(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        sender_email=sender_email,
        sender_password=sender_password,
        recipient_email=recipient_email,
    )


if __name__ == "__main__":
    print("Notifier module loaded. Run via main.py for full functionality.")
