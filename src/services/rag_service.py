# src/services/rag_service.py
from __future__ import annotations

import logging
import re
from typing import Optional, Sequence

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- паттерны --------------------
SMALLTALK_PATTERNS = [
    r"^привет\b", r"^здравствуй", r"^добрый\s+(день|вечер|утро)",
    r"^как дела", r"^ты кто", r"^кто ты", r"^что ты умеешь",
]

# Вопросы, которые явно про корпоративный отчёт/Газпром
REPORT_PATTERNS = [
    r"годов(ой|ого)\s+отч[её]т", r"\bотч[её]т\b", r"\b2024\b", r"доклад",
    r"газпром", r"показател", r"выручк", r"ebitda", r"чист(ая|ой)\s+прибыл",
    r"добыч", r"транспортировк", r"переработк", r"инвестпрограм",
    r"дивиден", r"капитальн(ые|ых)\s+вложен", r"сегмент", r"проект",
]

SMALLTALK_REPLY = (
    "Привет! Я помогу найти информацию в годовой корпоративной отчетности за 2024 год. "
    "Сделайте Ваш запрос, например - производство гелия..."
)

# -------------------- промпты --------------------
# Multi-Query для отчёта: поднимаем recall за счёт синонимов
REPORT_MQ_PROMPT = PromptTemplate.from_template(
    "Вопрос по годовому отчёту ПАО «Газпром» за 2024 год:\n{question}\n\n"
    "Сгенерируй 4–6 смысловых перефраз на русском для поиска по корпоративному отчёту. "
    "Используй синонимы и смежные термины (например: «выручка/доходы/Revenue», "
    "«операционная прибыль/EBITDA», «чистая прибыль/Net income», «инвестиции/CapEx», "
    "«добыча/разработка месторождений», «транспортировка/магистральные газопроводы», "
    "«переработка/НПЗ/Downstream», «сегменты/география/проекты»). "
    "Верни по одному варианту на строку, без нумерации и комментариев."
)

# Основной QA-промпт для отчёта‑2024: строго по контексту, но «живым» языком
REPORT_QA_PROMPT = PromptTemplate.from_template(
    "Ты эксперт по годовому отчёту ПАО «Газпром» за 2024 год. Отвечай на русском языке. "
    "Используй стиль понятного объяснения: сначала дай сам факт/цифру, затем краткое пояснение.\n\n"
    "⚠️ Правила:\n"
    "- Опирайся ТОЛЬКО на контекст. Не используй внешние знания.\n"
    "- Если в контексте есть точная цифра/дата/показатель — приводи её дословно, без округлений.\n"
    "- Если данных недостаточно — так и скажи и предложи уточнить (раздел, период, сегмент, проект).\n\n"
    "Контекст:\n{context}\n\n"
    "Вопрос: {question}\n\n"
    "Формат ответа:\n"
    "Чёткий ответ (цифра/факт/краткое утверждение).\n"
    "Короткое пояснение (1–3 предложения) человеческим языком.\n\n"
    "Ответ:"
)

# -------------------- сервис --------------------
class RAGService:
    """
    RAG для годового отчёта 2024:
      - MMR + MultiQuery + контекстная компрессия,
      - строгая опора на контекст (никаких «домыслов»),
      - дружелюбный small-talk.
    """

    def __init__(self) -> None:
        self.vectorstore: Optional[Chroma] = None

        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            temperature=0.3,        # живее, но без размывания цифр
            max_tokens=800,
            openai_api_key=Config.OPENAI_API_KEY,
            timeout=60,
            max_retries=3,
        )

        # Ретриверы
        self.report_retriever = None
        self.report_mq = None
        self.report_compressed = None

        # На случай, если понадобится «общий» без фильтра (fallback)
        self.general_retriever = None
        self.general_mq = None
        self.general_compressed = None

        # Имя файла отчёта для мета‑фильтра (если есть в метаданных)
        # Если в вашей индексации вы не пишете source_filename — фильтрация отключится автоматически.
        self.report_source_filename = "document.pdf.json"

    # ---------- инициализация ----------
    def initialize(self, vectorstore: Chroma) -> None:
        self.vectorstore = vectorstore

        # Базовый ретривер (без фильтра) — пригодится как fallback
        base = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8, "fetch_k": 40, "lambda_mult": 0.6},
        )
        self.general_retriever = base
        self.general_mq = MultiQueryRetriever.from_llm(
            retriever=base,
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                "Перефразируй вопрос по корпоративному документу:\n{question}\n\n"
                "Дай 4–6 вариантов на русском без нумерации."
            ),
        )
        self.general_compressed = ContextualCompressionRetriever(
            base_retriever=self.general_mq,
            base_compressor=LLMChainExtractor.from_llm(self.llm),
        )

        # Пытаемся построить ретривер С ФИЛЬТРОМ по файлу отчёта (если метаданные есть)
        # Если потом вернёт пусто — в get_answer() автоматически используем general_compressed.
        report_base = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "fetch_k": 40,
                "lambda_mult": 0.6,
                # ВАЖНО: это сработает только если при индексации в metadata попадал source_filename
                "filter": {"source_filename": self.report_source_filename},
            },
        )
        self.report_retriever = report_base
        self.report_mq = MultiQueryRetriever.from_llm(
            retriever=report_base, llm=self.llm, prompt=REPORT_MQ_PROMPT
        )
        self.report_compressed = ContextualCompressionRetriever(
            base_retriever=self.report_mq,
            base_compressor=LLMChainExtractor.from_llm(self.llm),
        )

        logger.info("RAG initialized for Report‑2024 (with optional metadata filter)")

    def is_initialized(self) -> bool:
        return self.vectorstore is not None

    def get_document_count(self) -> int:
        try:
            if not self.vectorstore:
                return 0
            collection = getattr(self.vectorstore, "_collection", None)
            return int(collection.count()) if collection else 0
        except Exception:
            return 0

    # ---------- ответ ----------
    def get_answer(self, question: str) -> str:
        q = (question or "").strip()
        if not q:
            return "Пожалуйста, задайте вопрос."

        if any(re.search(p, q.lower()) for p in SMALLTALK_PATTERNS):
            return SMALLTALK_REPLY

        if not self.vectorstore:
            return "RAG не инициализирован. Постройте индекс из JSON."

        # Выбираем ретривер:
        # 1) если вопрос похож на «про отчёт/Газпром» — сначала пробуем report_compressed (с фильтром),
        # 2) если фильтр не сработал (0 документов) — fallback на general_compressed,
        # 3) если вопрос совсем не похож — просто general_compressed.
        is_report_q = any(re.search(p, q.lower()) for p in REPORT_PATTERNS)

        retriever_chain = []
        if is_report_q and self.report_compressed:
            retriever_chain.append(self.report_compressed)
        if self.general_compressed:
            retriever_chain.append(self.general_compressed)

        docs: list[Document] = []
        for r in retriever_chain:
            try:
                docs = r.get_relevant_documents(q)
            except Exception:
                logger.exception("Retriever failed")
                docs = []
            if docs:
                break  # нашли — достаточно

        if not docs:
            return ("Не нашёл релевантных фрагментов отчёта к Вашему вопросу. "
                    "Уточните формулировку (раздел/сегмент/период/проект).")

        context = self._compose_context(docs, max_chars=4500)

        # Генерация
        try:
            prompt_text = REPORT_QA_PROMPT.format(context=context, question=q)
            resp = self.llm.invoke(prompt_text)
            return (getattr(resp, "content", None) or str(resp)).strip()
        except Exception as e:
            logger.exception("LLM error")
            return f"Ошибка генерации ответа: {e}"

    # ---------- utils ----------
    def _compose_context(self, docs: Sequence[Document], max_chars: int = 4500) -> str:
        """
        Склеиваем контекст из документов (со страницей и, при наличии, именем файла),
        удерживая общий размер под лимитом токенов.
        """
        parts, total = [], 0
        for d in docs:
            meta = d.metadata or {}
            page = meta.get("page_number")
            src = meta.get("source_filename")
            header_bits = []
            if page:
                header_bits.append(f"стр.{page}")
            if src:
                header_bits.append(f"{src}")
            prefix = f"[{' • '.join(header_bits)}] " if header_bits else ""
            block = f"{prefix}{d.page_content.strip()}"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n\n---\n\n".join(parts)
