import os
import json
import uuid
from datetime import datetime

from flask import Flask, render_template, request
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

app = Flask(__name__)

# Load environment variables from .env if present
load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "sql123")
DB_NAME = os.getenv("DB_NAME", "customer_feedback")

# ---------------------------
# Database helper
# ---------------------------

def get_db_connection():
    """Create and return a MySQL database connection."""
    return mysql.connector.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )

# ---------------------------
# Gemini response schema
# ---------------------------

class FeedbackAnalysis(BaseModel):
    summary: str = Field(description="2-3 line summary of the feedback")
    issues: list[str] = Field(description="Main issues as short bullet points")
    sentiment: str = Field(description="Positive, Neutral, or Negative")

# ---------------------------
# Gemini summarization
# ---------------------------

def summarize_with_gemini(feedback_text: str) -> dict:
    """Call Gemini API to summarize and analyze feedback.

    Returns a dict with keys: summary, issues, sentiment.
    Falls back to a safe default if the API fails.
    """
    if not GEMINI_API_KEY:
        return {
            "summary": "Gemini API key is missing.",
            "issues": [],
            "sentiment": "Neutral",
        }

    prompt = (
        "You are analyzing customer feedback."
        " Return JSON only with this exact schema:"
        " {\"summary\": \"\", \"issues\": [], \"sentiment\": \"\"}."
        "\n\nRules:"
        "\n- summary: 2-3 short lines"
        "\n- issues: list of short bullet-like strings"
        "\n- sentiment: Positive, Neutral, or Negative"
        "\n\nFeedback:\n" + feedback_text
    )

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FeedbackAnalysis,
                temperature=0.2,
            ),
        )

        # response.text should be JSON per response_mime_type
        data = json.loads(response.text)
        issues = data.get("issues", [])
        if isinstance(issues, str):
            issues = [s.strip() for s in issues.split(",") if s.strip()]
        return {
            "summary": data.get("summary", ""),
            "issues": issues,
            "sentiment": str(data.get("sentiment", "Neutral")).title(),
        }
    except Exception as exc:
        # Graceful fallback on API errors
        return {
            "summary": "Could not analyze feedback due to an API error.",
            "issues": ["Gemini API error"],
            "sentiment": "Neutral",
        }

# ---------------------------
# Routes
# ---------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    message = None
    analysis = None

    if request.method == "POST":
        customer_name = request.form.get("customer_name", "").strip()
        product_name = request.form.get("product_name", "").strip()
        feedback_text = request.form.get("feedback_text", "").strip()

        # Validate required feedback
        if not feedback_text:
            message = "Feedback text is required."
            return render_template("index.html", message=message)
        if not product_name:
            message = "Product or service name is required."
            return render_template("index.html", message=message)

        feedback_id = str(uuid.uuid4())
        analysis = summarize_with_gemini(feedback_text)

        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO feedbacks
                (feedback_id, customer_name, product_name, original_feedback,
                 summary, issues, sentiment, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    feedback_id,
                    customer_name or None,
                    product_name,
                    feedback_text,
                    analysis.get("summary", ""),
                    ", ".join(analysis.get("issues", [])),
                    analysis.get("sentiment", "Neutral"),
                    datetime.utcnow(),
                ),
            )
            conn.commit()
            cursor.close()
            conn.close()
            message = f"Feedback submitted successfully"
        except Error as exc:
            # Log full DB error to console for debugging
            print(f"Database error: {exc}")
            # Show a safe message to the user; append details in debug mode
            message = "Database error: Could not save feedback."
            if app.debug:
                message = f"{message} Details: {exc}"

    return render_template("index.html", message=message)

# ---------------------------
# Entry point
# ---------------------------

if __name__ == "__main__":
    app.run()

