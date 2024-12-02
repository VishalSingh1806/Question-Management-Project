import os
import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS

# Flask app
app = Flask(__name__)
CORS(app)

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Database constants
DB_PATH = r"/root/Question-Management-Project/database/knowledge_base.db"
SIMILARITY_THRESHOLD = 0.7


# --- Utility Functions ---
def connect_db():
    """Create a connection to the SQLite database."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def compute_embedding(text):
    """Compute embedding using Sentence-BERT."""
    return model.encode(text).reshape(1, -1)  # Ensure 384 dimensions


def save_or_update_db(conn, question, answer, embedding):
    """Save or update a QA pair in the database."""
    try:
        cursor = conn.cursor()
        # Check if the question exists in the database
        cursor.execute("SELECT id FROM ValidatedQA WHERE question = ?", (question,))
        result = cursor.fetchone()
        
        if result:
            # Update the existing row
            print(f"Updating answer for question: {question}")
            cursor.execute(
                "UPDATE ValidatedQA SET answer = ?, embedding = ? WHERE id = ?",
                (answer, embedding.tobytes(), result[0]),
            )
        else:
            # Insert a new row
            print(f"Inserting new question-answer pair: {question}")
            cursor.execute(
                "INSERT INTO ValidatedQA (question, answer, embedding) VALUES (?, ?, ?)",
                (question, answer, embedding.tobytes()),
            )
        conn.commit()
        print("Successfully saved or updated the question-answer pair.")
    except Exception as e:
        print(f"Error saving or updating to DB: {e}")
        raise


def query_validated_qa(user_embedding):
    """Query the ValidatedQA table for the best match."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
    rows = cursor.fetchall()
    conn.close()

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
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT question, answer FROM ValidatedQA")
        rows = cursor.fetchall()
        conn.close()
        questions = [{"question": row[0], "answer": row[1]} for row in rows]
        return jsonify({"questions": questions})
    except Exception as e:
        print(f"Error fetching questions: {e}")
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
        db_answer, confidence = query_validated_qa(user_embedding)

        # Return the database answer for validation
        return jsonify({
            "database": db_answer or "No match found in database.",
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error in /ask route: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/add", methods=["POST"])
def add_to_database():
    """Add or update a question-answer pair in the database."""
    try:
        data = request.json
        print(f"Incoming Data: {data}")  # Debug incoming request
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            print("Validation Failed: Question or Answer is missing.")
            return jsonify({"error": "Both question and answer are required."}), 400

        # Compute embedding for the question
        embedding = compute_embedding(question)

        # Add or update the question-answer pair in the database
        conn = connect_db()
        try:
            save_or_update_db(conn, question, answer, embedding)
        finally:
            conn.close()

        print(f"Question '{question}' updated or added successfully.")
        return jsonify({"message": "Answer submitted and saved successfully!"})
    except Exception as e:
        print(f"Error in /add route: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Wrap Flask app as ASGI for Uvicorn
asgi_app = WsgiToAsgi(app)

# --- Run App ---
if __name__ == "__main__":
    # Ensure the database exists
    if not os.path.exists(DB_PATH):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = connect_db()
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
