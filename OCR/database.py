from pymongo import MongoClient
from pymongo.errors import PyMongoError

# ---------------- LAZY MongoDB CONNECTION ---------------- #
# BUG FIX: Wrapped in try/except so the app doesn't crash on import
# if MongoDB is not running. The JSON fallback in main.py still works.

try:
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=3000)
    # Confirm the connection is actually alive
    client.admin.command("ping")
    db = client["ocr_db"]
    documents_collection = db["documents"]
    print("MongoDB connected successfully.")
except PyMongoError:
    client = None
    db = None
    documents_collection = None
    print("Warning: MongoDB unavailable. Falling back to JSON file storage.")
