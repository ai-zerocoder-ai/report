# src/services/json_ingest_service.py (фрагмент: заменить функцию)
from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping, Sequence

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

from config import Config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def _sanitize_metadata(md: Mapping[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in (md or {}).items():
        # частые списковые поля -> строка
        if k in ("languages", "tags", "keywords") and isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v if x is not None)
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
    return clean

def _batched(seq, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def build_vectorstore_from_json(json_path: str) -> Chroma:
    """
    Загружает один JSON и строит Chroma-индекс (батчево добавляя документы).
    Формат JSON-элемента: {"text": str, "metadata": dict}
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        elements = json.load(f)

    # 1) Собираем LangChain Documents
    docs: list[Document] = []
    for el in elements:
        if not isinstance(el, dict):
            continue
        text = (el.get("text") or "").strip()
        if not text:
            continue
        metadata = _sanitize_metadata(el.get("metadata", {}))
        docs.append(Document(page_content=text, metadata=metadata))

    if not docs:
        raise ValueError("No documents parsed from JSON")

    # 2) Перестраховка от сложных метаданных
    docs = filter_complex_metadata(docs)

    # 3) Чанкинг
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks: Sequence[Document] = splitter.split_documents(docs)
    logger.info("Prepared %d chunks from %d docs", len(chunks), len(docs))

    # 4) Embeddings
    embeddings = OpenAIEmbeddings(
        model=Config.EMBEDDING_MODEL,
        openai_api_key=Config.OPENAI_API_KEY,
        max_retries=3,
    )

    # 5) Создаём коллекцию и добавляем чанки порциями
    persist_dir = Config.CHROMA_PERSIST_DIRECTORY
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name="default",
    )

    BATCH = 100
    added = 0
    for batch in _batched(list(chunks), BATCH):
        vectorstore.add_documents(batch)
        added += len(batch)
        logger.info("Indexed %d/%d chunks...", added, len(chunks))

    vectorstore.persist()
    logger.info("Vectorstore built OK: %d chunks, dir=%s", len(chunks), persist_dir)
    return vectorstore
