import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = r"D:\EPR Data\Updated db'\knowledge_base.db"
model = SentenceTransformer("all-MiniLM-L6-v2")

def update_db_embeddings():
    """Regenerate embeddings in the database using SentenceTransformer."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question FROM ValidatedQA")
    rows = cursor.fetchall()

    for row in rows:
        question_id, question = row
        embedding = model.encode(question).astype(np.float32).tobytes()
        cursor.execute(
            "UPDATE ValidatedQA SET embedding = ? WHERE id = ?", (embedding, question_id)
        )

    conn.commit()
    conn.close()
    print("Database embeddings updated successfully.")

# Run the update function
update_db_embeddings()
