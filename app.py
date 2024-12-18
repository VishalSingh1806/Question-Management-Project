import os
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
import logging

# Flask app
app = Flask(__name__)
CORS(app)

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Database constants
DB_PATH = r"/root/Question-Management-Project/knowledge_base.db"
SIMILARITY_THRESHOLD = 0.7
logging.basicConfig(level=logging.DEBUG)


# --- Database Repository ---
class DatabaseRepository:
    def __init__(self, db_path):
        self.db_path = db_path

    def connect(self):
        """Create a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")  # Ensure FK constraints
        return conn

    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False):
        """Execute a query and optionally fetch results."""
        conn = self.connect()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params or [])
            if fetch_one:
                result = cursor.fetchone()
            elif fetch_all:
                result = cursor.fetchall()
            else:
                result = None
            conn.commit()  # Explicitly commit changes
            return result
        except sqlite3.Error as e:
            conn.rollback()
            logging.error(f"Database error: {e}")
            raise
        finally:
            conn.close()


# Initialize repository
db_repo = DatabaseRepository(DB_PATH)


# --- Utility Functions ---
def compute_embedding(text):
    """Compute embedding using Sentence-BERT."""
    return model.encode(text).reshape(1, -1)  # Ensure 384 dimensions


def save_or_update_question(db_repo, question, answer, embedding):
    """Save or update a QA pair in the database."""
    try:
        # Check if the question exists in the database
        existing_entry = db_repo.execute_query(
            "SELECT id FROM ValidatedQA WHERE question = ?",
            (question,),
            fetch_one=True,
        )

        if existing_entry:
            # Update the existing row
            logging.debug(f"Updating answer for question: {question}")
            db_repo.execute_query(
                "UPDATE ValidatedQA SET answer = ?, embedding = ? WHERE id = ?",
                (answer, embedding.tobytes(), existing_entry[0]),
            )
        else:
            # Insert a new row
            logging.debug(f"Inserting new question-answer pair: {question}")
            db_repo.execute_query(
                "INSERT INTO ValidatedQA (question, answer, embedding) VALUES (?, ?, ?)",
                (question, answer, embedding.tobytes()),
            )
        logging.debug("Changes committed to the database.")
    except Exception as e:
        logging.error(f"Error saving or updating to DB: {e}")
        raise


def fetch_best_match(db_repo, user_embedding):
    """Query the ValidatedQA table for the best match."""
    rows = db_repo.execute_query(
        "SELECT question, answer, embedding FROM ValidatedQA", fetch_all=True
    )

    max_similarity = 0.0
    best_answer = None

    for _, db_answer, db_embedding in rows:
        db_embedding_array = np.frombuffer(db_embedding, dtype=np.float32).reshape(1, -1)
        similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = db_answer

    if max_similarity >= SIMILARITY_THRESHOLD:
        return best_answer, float(max_similarity)
    return None, 0.0


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/questions", methods=["GET"])
def get_questions():
    """Fetch all questions from the database."""
    try:
        rows = db_repo.execute_query(
            "SELECT question, answer FROM ValidatedQA", fetch_all=True
        )
        if not rows:
            logging.debug("No questions found in the database.")
            return jsonify({"questions": []})

        questions = [{"question": row[0], "answer": row[1]} for row in rows]
        logging.debug(f"Fetched questions: {questions}")
        return jsonify({"questions": questions})
    except Exception as e:
        logging.error(f"Error fetching questions: {e}")
        return jsonify({"error": "Failed to fetch questions"}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Handle a question and return the answer from the database."""
    try:
        data = request.json
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Generate embedding for user query
        user_embedding = compute_embedding(question)

        # Query database for the best match
        db_answer, confidence = fetch_best_match(db_repo, user_embedding)

        # Return the database answer for validation
        return jsonify({
            "database": db_answer or "No match found in database.",
            "confidence": confidence
        })
    except Exception as e:
        logging.error(f"Error in /ask route: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/add", methods=["POST"])
def add_to_database():
    """Add or update a question-answer pair in the database."""
    try:
        data = request.json
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            return jsonify({"error": "Both question and answer are required."}), 400

        # Compute embedding for the question
        embedding = compute_embedding(question)

        # Add or update the question-answer pair in the database
        save_or_update_question(db_repo, question, answer, embedding)

        # Verify the update
        updated_entry = db_repo.execute_query(
            "SELECT id, question, answer FROM ValidatedQA WHERE question = ?",
            (question,),
            fetch_one=True,
        )

        if not updated_entry:
            logging.error(f"Failed to update or find the question: {question}")
            return jsonify({"error": "Failed to update the question in the database."}), 500

        logging.debug(f"Updated entry: {updated_entry}")
        return jsonify({"message": "Answer submitted and saved successfully!", "updated_entry": updated_entry})
    except Exception as e:
        logging.error(f"Error in /add route: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Wrap Flask app as ASGI for Uvicorn
asgi_app = WsgiToAsgi(app)

# --- Run App ---
if __name__ == "__main__":
    # Ensure the database exists
    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ValidatedQA (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT UNIQUE,
                answer TEXT,
                embedding BLOB
            )
        """)
        conn.close()

    app.run(debug=True)
