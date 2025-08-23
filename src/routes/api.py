# src/routes/api.py
from __future__ import annotations

import os
from flask import Blueprint, request, jsonify, render_template, current_app

from config import Config
from services.json_ingest_service import build_vectorstore_from_json

api_bp = Blueprint("api", __name__)


@api_bp.route("/")
def index():
    return render_template("index.html")


@api_bp.route("/api/ask", methods=["POST"])
def ask_question():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"status": "error", "error": "Question is required"}), 400

    answer = current_app.rag_service.get_answer(question)
    return jsonify({"status": "success", "question": question, "answer": answer})


@api_bp.route("/api/status")
def get_status():
    rag = current_app.rag_service
    return jsonify(
        {
            "rag_initialized": rag.is_initialized(),
            "document_count": rag.get_document_count(),
        }
    )


@api_bp.route("/api/process-json", methods=["POST"])
def process_json():
    """
    Ручной пересбор индекса из JSON (например, если файл обновили).
    """
    json_path = os.path.join(Config.PDF_UPLOAD_DIR, "document.pdf.json")
    if not os.path.exists(json_path):
        return (
            jsonify(
                {"status": "error", "message": f"JSON not found: {json_path}"}
            ),
            400,
        )

    try:
        vectorstore = build_vectorstore_from_json(json_path)
        current_app.rag_service.initialize(vectorstore)
        return jsonify({"status": "success", "message": "JSON processed successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
