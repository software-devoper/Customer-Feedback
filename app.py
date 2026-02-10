import os
import json
import uuid

from flask import Flask, render_template, request
import psycopg2
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------
# App setup
# ---------------------------------
app = Flask(__name__)
load_dotenv()

# ---------------------------------
# Config
# ---------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")

# ---------------------------------
# DB connection
# ---------------------------------
def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode=DB_SSLMODE,
    )

# ---------------------------------
# Gemini response schema
# ---------------------------------
class FeedbackAnalysis(BaseModel):
    summary: str = Field(description="2-3 line summary")
    issues: list[str] = Field(description="List of issues")
    sentiment: str = Field(description="Positive, Neutral, or Negative")

# ---------------------------------
# Gemini summarization
# ---------------------------------
def summarize_with_gemini(feedback_text: str) -> dict:
    if not GEMINI_API_KEY:
        return {
            "summary": "Gemini API key missing.",
            "issues": [],
            "sentiment": "Neutral",
        }

    prompt = f"""
Analyze the customer feedback below.

Return ONLY valid JSON with this schema:
{{
  "summary": "",
  "issues": [],
  "sentiment": ""
}}

Rules:
- summary: 2–3 short lines
- issues: short bullet points
- sentiment: Positive, Neutral, or Negative

Feedback:
{feedback_text}
"""

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

        data = json.loads(response.text)
        return {
            "summary": data.get("summary", ""),
            "issues": data.get("issues", []),
            "sentiment": data.get("sentiment", "Neutral").title(),
        }

    except Exception as e:
        print("Gemini error:", e)
        return {
            "summary": "AI analysis failed.",
            "issues": ["Gemini API error"],
            "sentiment": "Neutral",
        }

# ---------------------------------
# Routes
# ---------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    message = None

    if request.method == "POST":
        customer_name = request.form.get("customer_name", "").strip()
        product_name = request.form.get("product_name", "").strip()
        feedback_text = request.form.get("feedback_text", "").strip()

        if not product_name or not feedback_text:
            message = "Product name and feedback are required."
            return render_template("index.html", message=message)

        analysis = summarize_with_gemini(feedback_text)
        feedback_id = str(uuid.uuid4())

        try:
            conn = get_db_connection()
            cur = conn.cursor()

            # Create table (PostgreSQL)
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS "pgcrypto";

                CREATE TABLE IF NOT EXISTS feedbacks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    feedback_id UUID,
                    customer_name TEXT,
                    product_name TEXT NOT NULL,
                    original_feedback TEXT NOT NULL,
                    summary TEXT,
                    issues TEXT,
                    sentiment TEXT,
                    created_at TIMESTAMPTZ DEFAULT now()
                );
            """)

            # Insert feedback
            cur.execute(
                """
                INSERT INTO feedbacks
                (feedback_id, customer_name, product_name, original_feedback,
                 summary, issues, sentiment)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    feedback_id,
                    customer_name or None,
                    product_name,
                    feedback_text,
                    analysis["summary"],
                    ", ".join(analysis["issues"]),
                    analysis["sentiment"],
                ),
            )

            conn.commit()
            cur.close()
            conn.close()

            message = "Feedback submitted successfully!"

        except Exception as e:
            print("Database error:", e)
            message = "Database error. Check logs."

    return render_template("index.html", message=message)

# ---------------------------------
# Entry point
# ---------------------------------
if __name__ == "__main__":
    app.run()
