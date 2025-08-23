# src/app.py
from __future__ import annotations

import os
import sys
import logging
from flask import Flask

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from routes.api import api_bp
from services.rag_service import RAGService
from services.json_ingest_service import build_vectorstore_from_json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_object(Config)

    # Сервисы (единые инстансы)
    app.rag_service = RAGService()

    # Пытаемся построить индекс на старте (если JSON есть)
    json_path = os.path.join(Config.PDF_UPLOAD_DIR, "document.pdf.json")
    try:
        if os.path.exists(json_path):
            vectorstore = build_vectorstore_from_json(json_path)
            app.rag_service.initialize(vectorstore)
            logger.info("Index built at startup from %s", json_path)
        else:
            logger.warning("JSON not found at startup: %s", json_path)
    except Exception:
        logger.exception("Failed to initialize RAG at startup")

    # Роуты
    app.register_blueprint(api_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=app.config["DEBUG"])
