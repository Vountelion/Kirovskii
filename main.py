# rag_system_async.py - ПОЛНАЯ ВЕРСИЯ С COLBERT И СОХРАНЕНИЕМ ГРАФА ЛЕЙДЕНА

# Python ≥3.10

from __future__ import annotations

import asyncio, hashlib, json, logging, os, time, re

from dataclasses import dataclass

from functools import lru_cache

from typing import Any, List, Protocol, Optional

from pathlib import Path

import faiss  # pip install faiss-cpu

import numpy as np

import torch

from cachetools import TTLCache  # pip install cachetools

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader
)

from langchain_community.vectorstores import FAISS

from langchain_community.retrievers import BM25Retriever

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_ollama import OllamaLLM

from langchain_experimental.text_splitter import SemanticChunker

from langchain.text_splitter import RecursiveCharacterTextSplitter

from neo4j import AsyncGraphDatabase

# ✅ НОВЫЕ ИМПОРТЫ ДЛЯ COLBERT
try:
    from ragatouille import RAGPretrainedModel

    COLBERT_AVAILABLE = True
    print("✅ ColBERT (RAGatouille) доступен")
except ImportError:
    COLBERT_AVAILABLE = False
    print("❌ ColBERT недоступен. Установите: pip install ragatouille")


# -----------------------------------------------------------------------------
# 1. Конфигурация и базовые протоколы
# -----------------------------------------------------------------------------

@dataclass
class RAGConfig:
    folder_path: str
    faiss_path: str
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 8
    ttl_cache_sec: int = 3_600
    faiss_ivf_thr: int = 1_000
    faiss_nprobe: int = 32
    log_level: int = logging.INFO
    chunk_size: int = 1000
    chunk_overlap: int = 200
    use_semantic_chunking: bool = True
    # ✅ НОВЫЕ НАСТРОЙКИ ДЛЯ СОХРАНЕНИЯ/ЗАГРУЗКИ ГРАФА
    force_rebuild_graph: bool = True  # При первом запуске True, потом False
    save_graph_structure: bool = True  # Сохранять ли граф после создания
    graph_metadata_dir: str = "graph_data/"  # Папка для метаданных
    auto_load_existing: bool = True  # Автоматически загружать существующий граф
    # ✅ НОВЫЕ НАСТРОЙКИ ДЛЯ COLBERT
    use_colbert_reranker: bool = True
    colbert_model: str = "colbert-ir/colbertv2.0"
    # ✅ УЛУЧШЕННЫЕ НАСТРОЙКИ LLM ДЛЯ ПРЕДОТВРАЩЕНИЯ ПОВТОРЕНИЙ
    llm_temperature: float = 0.4  # Снижено для более стабильных ответов
    llm_repeat_penalty: float = 1.05  # Мягкое наказание за повторения
    llm_repeat_last_n: int = 128  # Окно для анализа повторений
    llm_top_p: float = 0.85  # Более избирательная выборка
    llm_top_k: int = 40  # Ограничение количества кандидатов
    llm_num_predict: int = 2048  # Максимум токенов
    llm_stop_sequences: List[str] = None  # Будет установлено в __post_init__

    def __post_init__(self):
        if self.llm_stop_sequences is None:
            # ✅ СТОП-ПОСЛЕДОВАТЕЛЬНОСТИ ДЛЯ ПРЕДОТВРАЩЕНИЯ ПОВТОРЕНИЙ
            self.llm_stop_sequences = [
                "\n\n\n",  # Много переносов строк
                "Повторяю",  # Русские фразы повторения
                "Еще раз",
                "Снова",
                "Опять",
                "Как я уже говорил",
                "Как уже было сказано",
                "Repeating",  # Английские фразы повторения
                "Again,",
                "As I mentioned",
                "As stated before",
            ]


class Retriever(Protocol):
    async def retrieve(self, query: str, k: int) -> List[Any]: ...


# -----------------------------------------------------------------------------
# 2. Логгер и утилиты
# -----------------------------------------------------------------------------

def init_logger(level: int) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler("rag_system.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("RAG")


logger = init_logger(logging.INFO)


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()


def compute_documents_hash(documents: List[Any]) -> str:
    """Вычисляет хэш содержимого всех документов."""
    hasher = hashlib.md5()
    for doc in sorted(documents, key=lambda x: x.metadata.get('source', '')):
        content = doc.page_content.encode('utf-8', errors='ignore')
        hasher.update(content)
    return hasher.hexdigest()


# -----------------------------------------------------------------------------
# 3. ✅ НОВЫЙ ColBERT Reranker
# -----------------------------------------------------------------------------

class ColBERTReranker:
    """ColBERT reranker для улучшения качества ранжирования документов."""

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.model: Optional[RAGPretrainedModel] = None
        self._available = COLBERT_AVAILABLE

    @property
    def colbert_model(self) -> Optional[RAGPretrainedModel]:
        """Ленивая инициализация ColBERT модели"""
        if not self._available:
            return None

        if self.model is None:
            try:
                logger.info(f"Загрузка ColBERT модели: {self.cfg.colbert_model}")
                self.model = RAGPretrainedModel.from_pretrained(self.cfg.colbert_model)
                logger.info("✅ ColBERT модель загружена успешно")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки ColBERT: {e}")
                self._available = False
                return None
        return self.model

    def rerank(self, query: str, documents: List[Any], k: int = 10) -> List[Any]:
        """Переранжирование документов с помощью ColBERT"""
        if not self._available or not documents:
            logger.debug("ColBERT недоступен или нет документов для ранжирования")
            return documents[:k]

        model = self.colbert_model
        if model is None:
            return documents[:k]

        try:
            # Извлекаем тексты для ранжирования
            doc_texts = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    doc_texts.append(doc.page_content)
                else:
                    doc_texts.append(str(doc))

            if not doc_texts:
                return documents[:k]

            # Переранжирование с ColBERT
            start_time = time.time()
            ranked_results = model.rerank(query=query, documents=doc_texts, k=k)
            rerank_time = time.time() - start_time

            logger.debug(f"ColBERT переранжирование: {len(documents)} -> {len(ranked_results)} за {rerank_time:.2f}с")

            # Сопоставляем результаты с исходными документами
            reranked_docs = []
            for result in ranked_results:
                # RAGatouille возвращает словарь с 'content' и 'score'
                if isinstance(result, dict) and 'content' in result:
                    content = result['content']
                    # Ищем соответствующий документ
                    for doc in documents:
                        doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                        if doc_content == content:
                            reranked_docs.append(doc)
                            break
                else:
                    # Если результат в другом формате, добавляем как есть
                    reranked_docs.append(result)

            return reranked_docs[:k]

        except Exception as e:
            logger.error(f"Ошибка ColBERT ранжирования: {e}")
            return documents[:k]


# -----------------------------------------------------------------------------
# 4. ✅ CONTEXT-AWARE Cache Manager
# -----------------------------------------------------------------------------

class ContextAwareCacheManager:
    def __init__(self, ttl: int):
        self.cache = TTLCache(maxsize=1_000, ttl=ttl)
        self.user_contexts = {}  # Контекст для каждого пользователя
        self.hits = 0
        self.misses = 0

    def _generate_context_aware_key(self, query: str, user_id: str = None, context_window: int = 3) -> str:
        """Создает ключ кэша с учетом контекста пользователя"""
        # Базовый хэш запроса
        base_hash = md5_text(query.lower().strip())

        if user_id:
            # Получаем последние N запросов пользователя для контекста
            user_history = self.user_contexts.get(user_id, [])
            recent_queries = user_history[-context_window:]
            context_hash = md5_text("|".join(recent_queries))
            return f"{base_hash}_{context_hash}_{user_id}"

        return base_hash

    def get(self, query: str, user_id: str = None) -> Optional[str]:
        """Получение с учетом пользовательского контекста"""
        cache_key = self._generate_context_aware_key(query, user_id)
        result = self.cache.get(cache_key)

        if result:
            self.hits += 1
            logger.debug(f"Cache hit для пользователя {user_id}: {query[:50]}...")
        else:
            self.misses += 1

        return result

    def set(self, query: str, answer: str, user_id: str = None):
        """Сохранение с учетом контекста"""
        cache_key = self._generate_context_aware_key(query, user_id)
        self.cache[cache_key] = answer

        # Обновляем историю пользователя
        if user_id:
            if user_id not in self.user_contexts:
                self.user_contexts[user_id] = []

            self.user_contexts[user_id].append(query.lower().strip())

            # Ограничиваем историю (храним только последние 10 запросов)
            if len(self.user_contexts[user_id]) > 10:
                self.user_contexts[user_id] = self.user_contexts[user_id][-10:]

    def clear_user_context(self, user_id: str):
        """Очистка контекста пользователя"""
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]

        # Удаляем связанные кэш-записи
        keys_to_remove = [k for k in self.cache.keys() if str(user_id) in str(k)]
        for key in keys_to_remove:
            del self.cache[key]

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = (self.hits / max(total, 1)) * 100
        return {
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%"
        }


# -----------------------------------------------------------------------------
# 5. ✅ DYNAMIC LLM Manager
# -----------------------------------------------------------------------------

class DynamicLLMManager:
    def __init__(self, base_llm):
        self.base_llm = base_llm
        self.response_history = {}  # user_id -> последние ответы

    def _calculate_response_similarity(self, new_response: str, user_id: str) -> float:
        """Вычисляет сходство с предыдущими ответами пользователя"""
        if user_id not in self.response_history:
            return 0.0

        # Простая метрика сходства (можно улучшить с помощью embeddings)
        previous_responses = self.response_history[user_id]
        max_similarity = 0.0

        for prev_response in previous_responses:
            # Jaccard similarity для быстрой оценки
            set1 = set(new_response.lower().split())
            set2 = set(prev_response.lower().split())
            similarity = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
            max_similarity = max(max_similarity, similarity)

        return max_similarity

    def generate_response(self, prompt: str, user_id: str = None) -> str:
        """Генерирует ответ с адаптивными параметрами"""
        # Базовые параметры
        temperature = 0.4
        top_p = 0.85
        top_k = 40

        # Адаптируем параметры для разнообразия
        if user_id and user_id in self.response_history:
            # Увеличиваем креативность для пользователей с историей
            temperature = min(0.7, 0.4 + len(self.response_history[user_id]) * 0.05)
            top_p = min(0.95, 0.85 + len(self.response_history[user_id]) * 0.02)

        # Добавляем инструкцию для разнообразия в prompt
        enhanced_prompt = f"""{prompt}

ВАЖНО: Дай уникальный, свежий ответ. Избегай повторения фраз и структур из предыдущих ответов. Используй разные формулировки и подходы к объяснению."""

        # Генерируем ответ с обновленными параметрами
        original_temp = getattr(self.base_llm, 'temperature', 0.4)
        original_top_p = getattr(self.base_llm, 'top_p', 0.85)
        original_top_k = getattr(self.base_llm, 'top_k', 40)

        self.base_llm.temperature = temperature
        self.base_llm.top_p = top_p
        self.base_llm.top_k = top_k

        response = self.base_llm.invoke(enhanced_prompt)

        # Восстанавливаем исходные параметры
        self.base_llm.temperature = original_temp
        self.base_llm.top_p = original_top_p
        self.base_llm.top_k = original_top_k

        # Проверяем на сходство и при необходимости регенерируем
        if user_id:
            similarity = self._calculate_response_similarity(response, user_id)

            if similarity > 0.7:  # Слишком похожий ответ
                logger.info(f"Высокое сходство ({similarity:.2f}), регенерирую с большей креативностью")

                # Увеличиваем креативность для регенерации
                self.base_llm.temperature = min(0.9, temperature + 0.3)
                self.base_llm.top_p = 0.95

                creative_prompt = f"""{prompt}

КРИТИЧЕСКИ ВАЖНО: Предыдущий ответ был слишком похож на уже данные. 
Дай СОВЕРШЕННО ДРУГОЙ ответ с:
- Другой структурой изложения
- Другими примерами и аналогиями  
- Другим стилем объяснения
- Другими акцентами и приоритетами"""

                response = self.base_llm.invoke(creative_prompt)

                # Восстанавливаем параметры
                self.base_llm.temperature = original_temp
                self.base_llm.top_p = original_top_p

        # Сохраняем ответ в историю
        self._update_response_history(response, user_id)

        return response.strip()

    def _update_response_history(self, response: str, user_id: str):
        """Обновляет историю ответов пользователя"""
        if not user_id:
            return

        if user_id not in self.response_history:
            self.response_history[user_id] = []

        self.response_history[user_id].append(response)

        # Храним только последние 5 ответов
        if len(self.response_history[user_id]) > 5:
            self.response_history[user_id] = self.response_history[user_id][-5:]

    def clear_user_history(self, user_id: str):
        """Очищает историю ответов пользователя"""
        if user_id in self.response_history:
            del self.response_history[user_id]


# -----------------------------------------------------------------------------
# 6. ✅ УЛУЧШЕННЫЙ Менеджер ресурсов с ColBERT
# -----------------------------------------------------------------------------

class ResourceManager:
    """Отвечает за загрузку/очистку тяжёлых моделей."""

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._emb_model: Optional[HuggingFaceEmbeddings] = None
        self._sentence_embedder: Optional[HuggingFaceEmbeddings] = None
        self._colbert_reranker: Optional[ColBERTReranker] = None
        self._llm: Optional[OllamaLLM] = None

    @property
    def emb_model(self) -> HuggingFaceEmbeddings:
        """Модель для векторизации при поиске"""
        if self._emb_model is None:
            try:
                self._emb_model = HuggingFaceEmbeddings(
                    model_name="Qwen/Qwen3-Embedding-0.6B",
                    model_kwargs={"device": self.cfg.device},
                    encode_kwargs={
                        "batch_size": self.cfg.batch_size,
                        "normalize_embeddings": True
                    }
                )
                logger.info("Embedding-модель для FAISS загружена (лениво): Qwen/Qwen3-Embedding-0.6B")
            except Exception as e:
                logger.error(f"Ошибка загрузки embedding модели: {e}")
                raise
        return self._emb_model

    @property
    def sentence_embedder(self) -> HuggingFaceEmbeddings:
        """Модель для семантического чанкинга"""
        if self._sentence_embedder is None:
            try:
                self._sentence_embedder = HuggingFaceEmbeddings(
                    model_name="sberbank-ai/sbert_large_nlu_ru",
                    model_kwargs={'device': self.cfg.device},
                    encode_kwargs={
                        'batch_size': self.cfg.batch_size,
                        'normalize_embeddings': True
                    }
                )
                logger.info("Embedding-модель для чанкинга загружена (лениво): sberbank-ai/sbert_large_nlu_ru")
            except Exception as e:
                logger.warning(
                    f"Ошибка загрузки модели для чанкинга: {e}. Будет использован RecursiveCharacterTextSplitter")
                self._sentence_embedder = None
        return self._sentence_embedder

    @property
    def colbert_reranker(self) -> ColBERTReranker:
        """ColBERT reranker для улучшенного ранжирования"""
        if self._colbert_reranker is None:
            self._colbert_reranker = ColBERTReranker(self.cfg)
        return self._colbert_reranker

    @property
    def llm(self) -> OllamaLLM:
        if self._llm is None:
            try:
                # ✅ ИСПРАВЛЕННЫЕ НАСТРОЙКИ LLM ДЛЯ ПРЕДОТВРАЩЕНИЯ ПОВТОРЕНИЙ
                self._llm = OllamaLLM(
                    model="qwen2.5vl:7b",
                    # ✅ Основные параметры для предотвращения повторений
                    num_ctx=16384,  # Большой контекст
                    num_predict=self.cfg.llm_num_predict,  # Максимум токенов
                    temperature=self.cfg.llm_temperature,  # Сбалансированная креативность
                    repeat_penalty=self.cfg.llm_repeat_penalty,  # Мягкое наказание
                    repeat_last_n=self.cfg.llm_repeat_last_n,  # Окно анализа
                    top_p=self.cfg.llm_top_p,  # Nucleus sampling
                    top_k=self.cfg.llm_top_k,  # Ограничение кандидатов
                    # ✅ Стоп-последовательности
                    stop=self.cfg.llm_stop_sequences,
                    # ✅ Дополнительные настройки
                    seed=42,  # Для воспроизводимости (можно убрать для большей случайности)
                )
                logger.info("✅ LLM загружен с улучшенными настройками против повторений")
            except Exception as e:
                logger.error(f"Ошибка загрузки LLM: {e}")
                raise
        return self._llm

    @staticmethod
    def gpu_clear():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("GPU-cache очищен")


# -----------------------------------------------------------------------------
# 7. Улучшенный загрузчик документов с поддержкой PDF, DOCX, TXT
# -----------------------------------------------------------------------------

class AdvancedDocumentLoader:
    """Загрузчик документов с поддержкой PDF, DOCX и TXT файлов."""
    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}

    def __init__(self):
        self.documents = []
        self.chunks = []

    def load_documents(self, folder_path: str) -> List[Any]:
        """Загружает документы из указанной папки."""
        documents = []
        if not os.path.exists(folder_path):
            logger.error(f"Папка не найдена: {folder_path}")
            raise FileNotFoundError(f"Папка не найдена: {folder_path}")

        logger.info(f"Начало загрузки файлов из {folder_path}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            ext = Path(filename).suffix.lower()

            if ext not in self.SUPPORTED_EXTENSIONS:
                logger.warning(f"Пропущен файл с неподдерживаемым расширением: {filename}")
                continue

            try:
                loader = self._get_loader(file_path, ext)
                loaded_docs = loader.load()

                # Добавляем метаданные
                for doc in loaded_docs:
                    doc.metadata.update({
                        'filename': filename,
                        'file_path': file_path,
                        'file_type': ext
                    })

                documents.extend(loaded_docs)
                logger.info(f"Успешно загружен: {filename} (частей: {len(loaded_docs)})")

            except Exception as e:
                logger.error(f"Ошибка загрузки {filename}: {str(e)}")
                continue

        logger.info(f"Всего загружено документов: {len(documents)}")
        self.documents = documents
        return documents

    def _get_loader(self, file_path: str, ext: str):
        """Возвращает соответствующий загрузчик для типа файла."""
        if ext == '.pdf':
            return UnstructuredPDFLoader(file_path)
        elif ext == '.docx':
            return UnstructuredWordDocumentLoader(file_path)
        elif ext == '.txt':
            return TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Неподдерживаемый тип файла: {ext}")

    def chunk_documents(self, documents: List[Any], cfg: RAGConfig, rm: ResourceManager) -> List[Any]:
        """Разбивает документы на чанки."""
        if cfg.use_semantic_chunking and rm.sentence_embedder is not None:
            chunks = self._semantic_chunking(documents, rm.sentence_embedder)
        else:
            chunks = self._recursive_chunking(documents, cfg)

        # Добавляем уникальные ID для чанков
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["original_doc_count"] = len(documents)

        self.chunks = chunks
        logger.info(f"Создано чанков: {len(chunks)}")
        return chunks

    def _semantic_chunking(self, documents: List[Any], sentence_embedder: HuggingFaceEmbeddings) -> List[Any]:
        """Семантическое разбиение на чанки."""
        try:
            chunker = SemanticChunker(
                embeddings=sentence_embedder,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=0.5
            )
            chunks = chunker.split_documents(documents)
            logger.info(f"Семантическое разбиение завершено: {len(chunks)} чанков")
            return chunks
        except Exception as e:
            logger.error(f"Ошибка семантического разбиения: {e}. Переключение на RecursiveCharacterTextSplitter")
            return self._recursive_chunking(documents, None)

    def _recursive_chunking(self, documents: List[Any], cfg: Optional[RAGConfig]) -> List[Any]:
        """Рекурсивное разбиение на чанки."""
        chunk_size = cfg.chunk_size if cfg else 1000
        chunk_overlap = cfg.chunk_overlap if cfg else 200

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Рекурсивное разбиение завершено: {len(chunks)} чанков")
        return chunks


# -----------------------------------------------------------------------------
# 8. Оптимизированное FAISS-хранилище с сохранением метаданных
# -----------------------------------------------------------------------------

class OptimizedFaissStore:
    """Улучшенное FAISS-хранилище с поддержкой метаданных и хэширования."""

    def __init__(self, cfg: RAGConfig, rm: ResourceManager):
        self.cfg = cfg
        self.rm = rm
        self.vector_store: Optional[FAISS] = None
        self.documents_hash: Optional[str] = None
        self._ready = asyncio.Event()

    async def build(self, chunks: List[Any]):
        """Строим или загружаем векторное хранилище."""
        current_hash = compute_documents_hash(chunks)
        hash_file = os.path.join(os.path.dirname(self.cfg.faiss_path), "documents_hash.json")

        force_recreate = True
        if os.path.exists(hash_file) and os.path.exists(self.cfg.faiss_path):
            try:
                with open(hash_file, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                saved_hash = saved_data.get('hash')
                if saved_hash == current_hash:
                    force_recreate = False
                    logger.info(f"Хэш документов не изменился: {current_hash}")
            except Exception as e:
                logger.warning(f"Ошибка чтения файла хэша: {e}")

        if force_recreate:
            await self._create_new_index(chunks, current_hash, hash_file)
        else:
            await self._load_existing_index()

        self._ready.set()

    async def _create_new_index(self, chunks: List[Any], current_hash: str, hash_file: str):
        """Создает новый FAISS индекс."""
        logger.info("Создание нового FAISS индекса...")
        self.rm.gpu_clear()

        try:
            self.vector_store = FAISS.from_documents(chunks, self.rm.emb_model)
            os.makedirs(os.path.dirname(self.cfg.faiss_path), exist_ok=True)
            self.vector_store.save_local(self.cfg.faiss_path)

            with open(hash_file, 'w', encoding='utf-8') as f:
                json.dump({'hash': current_hash}, f)

            self.documents_hash = current_hash
            logger.info(f"FAISS индекс создан и сохранен в {self.cfg.faiss_path}")

        except Exception as e:
            logger.error(f"Ошибка создания FAISS индекса: {e}")
            raise

    async def _load_existing_index(self):
        """Загружает существующий FAISS индекс."""
        try:
            self.rm.gpu_clear()
            self.vector_store = FAISS.load_local(
                self.cfg.faiss_path,
                self.rm.emb_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS индекс загружен из {self.cfg.faiss_path}")
        except Exception as e:
            logger.error(f"Ошибка загрузки FAISS индекса: {e}")
            raise

    async def search(self, query: str, k: int = 10) -> List[Any]:
        """Поиск по векторному хранилищу."""
        await self._ready.wait()
        if not self.vector_store:
            logger.warning("Векторное хранилище не инициализировано")
            return []

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            logger.error(f"Ошибка поиска в FAISS: {e}")
            return []


# -----------------------------------------------------------------------------
# 9. BM25-ретривер (async оболочка)
# -----------------------------------------------------------------------------

class AsyncBM25Retriever:
    def __init__(self, chunks: List[Any]):
        self.retriever = BM25Retriever.from_documents(chunks)
        self.retriever.k = 10
        logger.info("BM25Retriever инициализирован")

    async def retrieve(self, query: str, k: int = 10) -> List[Any]:
        loop = asyncio.get_running_loop()
        try:
            self.retriever.k = k
            docs = await loop.run_in_executor(None, self.retriever.invoke, query)
            return docs
        except Exception as e:
            logger.error(f"Ошибка BM25 поиска: {e}")
            return []


# -----------------------------------------------------------------------------
# 10. ✅ GraphRAGSystem С СОХРАНЕНИЕМ И ЗАГРУЗКОЙ СТРУКТУРЫ ЛЕЙДЕНА
# -----------------------------------------------------------------------------

class GraphRAGSystem:
    """✅ ПОЛНАЯ GraphRAG система с сохранением и загрузкой структуры Лейдена."""

    def __init__(self, cfg: RAGConfig, rm: ResourceManager):
        self.cfg, self.rm = cfg, rm
        self.graph_driver = None
        self.chunks: List[Any] = []
        self.community_summaries: dict = {}

        # ✅ НОВЫЕ параметры для управления сохранением/загрузкой
        self.graph_metadata_file = os.path.join(cfg.graph_metadata_dir, "graph_metadata.json")
        self.community_summaries_file = os.path.join(cfg.graph_metadata_dir, "community_summaries.json")
        self.graph_built = False

        # Создаем директорию для метаданных
        os.makedirs(cfg.graph_metadata_dir, exist_ok=True)

        self._entity_extraction_prompt = """
Ты эксперт по анализу текста. Проанализируй текст и извлеки сущности и связи.

ТЕКСТ:
{text}

ИНСТРУКЦИЯ: Верни ТОЛЬКО валидный JSON без дополнительных комментариев:

{{
  "entities": [
    {{"text": "название", "label": "тип", "description": "описание"}}
  ],
  "relations": [
    {{"source": "сущность1", "target": "сущность2", "type": "связь", "description": "описание"}}
  ]
}}

Если сущностей нет, верни: {{"entities": [], "relations": []}}
"""

        self._query_entity_extraction_prompt = """
Ты — помощник по извлечению ключевых сущностей из пользовательского запроса.
Извлеки все важные сущности, которые могут помочь в поиске релевантной информации.

Запрос: {query}

Верни JSON-массив строк с названиями сущностей:
["сущность1", "сущность2", ...]
"""

        try:
            self.graph_driver = AsyncGraphDatabase.driver(
                cfg.neo4j_uri,
                auth=(cfg.neo4j_user, cfg.neo4j_password),
                max_connection_pool_size=10,
                connection_timeout=30
            )
            logger.info("GraphRAG: Асинхронное Neo4j соединение установлено")
        except Exception as e:
            logger.error(f"GraphRAG: Ошибка подключения к Neo4j: {e}")
            self.graph_driver = None

    async def verify_connection(self) -> bool:
        """Проверяет соединение с Neo4j."""
        if not self.graph_driver:
            return False

        try:
            await self.graph_driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"GraphRAG: Ошибка проверки соединения: {e}")
            return False

    async def _force_clear_database(self):
        """Принудительная полная очистка графовой БД."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.error("GraphRAG: Нет активного event loop для очистки БД")
            return

        async with self.graph_driver.session() as session:
            try:
                await session.run("MATCH (n) DETACH DELETE n")

                # ✅ ИСПРАВЛЕННОЕ удаление ограничений для Neo4j 5.x
                try:
                    await session.run("DROP CONSTRAINT entity_name_unique IF EXISTS")
                except Exception as e:
                    logger.debug(f"GraphRAG: Ограничения уже удалены: {e}")

                logger.info("GraphRAG: База данных принудительно очищена")

            except Exception as e:
                logger.error(f"GraphRAG: Ошибка при очистке БД: {e}")
                raise

    async def initialize_graph(self, chunks: List[Any]) -> bool:
        """✅ УЛУЧШЕННАЯ инициализация с проверкой существующего графа."""
        if not await self.verify_connection():
            logger.warning("GraphRAG: Neo4j недоступен")
            return False

        self.chunks = chunks

        try:
            # ✅ НОВАЯ ЛОГИКА: Проверяем существующий граф
            if not self.cfg.force_rebuild_graph and self.cfg.auto_load_existing:
                existing_graph = await self._check_existing_complete_graph()

                if existing_graph:
                    logger.info("🎯 GraphRAG: Найден существующий граф со структурой Лейдена")
                    success = await self._load_existing_graph_structure()
                    if success:
                        logger.info("✅ GraphRAG: Существующий граф успешно загружен")
                        return True

            # Если нет существующего графа или принудительная пересборка
            if self.cfg.force_rebuild_graph:
                logger.info("GraphRAG: Принудительная пересборка графа")
                await self._force_clear_database()

            success = await self._build_graph_from_chunks(chunks)
            if success and self.cfg.save_graph_structure:
                # ✅ СОХРАНЯЕМ граф после создания
                await self._save_graph_structure()
                logger.info("GraphRAG: Граф успешно построен и сохранен")
            return success

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка инициализации: {e}")
            return False

    async def _check_existing_complete_graph(self) -> bool:
        """Проверяет существование полноценного графа со структурой Лейдена."""
        async with self.graph_driver.session() as session:
            try:
                # Проверяем наличие узлов и связей
                result = await session.run("""
                    MATCH (e:Entity) 
                    OPTIONAL MATCH ()-[r:RELATED]->()
                    RETURN count(DISTINCT e) AS nodes, count(r) AS relationships
                """)

                record = await result.single()
                if not record or record["nodes"] == 0:
                    logger.info("GraphRAG: Граф пуст")
                    return False

                nodes_count = record["nodes"]
                rels_count = record["relationships"]
                logger.info(f"GraphRAG: Найден граф: {nodes_count} узлов, {rels_count} связей")

                # Проверяем наличие хотя бы одного уровня сообществ Лейдена
                for level in range(6):  # Проверяем уровни 0-5
                    community_check = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        RETURN count(e) AS with_communities
                        LIMIT 1
                    """)

                    community_record = await community_check.single()
                    if community_record and community_record["with_communities"] > 0:
                        logger.info(f"GraphRAG: Найдены сообщества Лейдена уровня {level}")
                        return True

                logger.info("GraphRAG: Граф существует, но без структуры Лейдена")
                return False

            except Exception as e:
                logger.warning(f"GraphRAG: Ошибка проверки существующего графа: {e}")
                return False

    async def _load_existing_graph_structure(self) -> bool:
        """Загружает существующую структуру графа и метаданные."""
        try:
            # Загружаем метаданные графа
            await self._load_graph_metadata()

            # Загружаем сводки сообществ
            await self._load_community_summaries_from_file()

            # Проверяем какие уровни Лейдена доступны
            available_levels = await self._get_available_leiden_levels()
            logger.info(f"GraphRAG: Доступные уровни Лейдена: {available_levels}")

            if available_levels:
                # Вычисляем метрики для существующих уровней
                await self._compute_community_metrics_safe(available_levels)
                logger.info("GraphRAG: ✅ Существующая структура Лейдена загружена")
                return True
            else:
                logger.warning("GraphRAG: Структура Лейдена не найдена")
                return False

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка загрузки структуры: {e}")
            return False

    async def _get_available_leiden_levels(self) -> List[int]:
        """Определяет доступные уровни алгоритма Лейдена."""
        available_levels = []

        async with self.graph_driver.session() as session:
            try:
                for level in range(6):  # Проверяем уровни 0-5
                    result = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        RETURN count(e) AS count
                        LIMIT 1
                    """)

                    record = await result.single()
                    if record and record["count"] > 0:
                        available_levels.append(level)

            except Exception as e:
                logger.warning(f"GraphRAG: Ошибка определения уровней Лейдена: {e}")

        return available_levels

    async def _save_graph_structure(self):
        """Сохраняет структуру графа и метаданные."""
        try:
            # 1. Сохраняем метаданные графа
            await self._save_graph_metadata()

            # 2. Сохраняем сводки сообществ в файл
            await self._save_community_summaries_to_file()

            # 3. Результаты алгоритма Лейдена уже сохранены как свойства узлов
            logger.info("GraphRAG: ✅ Структура графа сохранена")

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка сохранения структуры: {e}")

    async def _save_graph_metadata(self):
        """Сохраняет метаданные графа в файл."""
        try:
            async with self.graph_driver.session() as session:
                # Собираем статистику графа
                stats_result = await session.run("""
                    MATCH (e:Entity) 
                    OPTIONAL MATCH ()-[r:RELATED]->()
                    RETURN count(DISTINCT e) AS nodes, count(r) AS relationships
                """)

                stats_record = await stats_result.single()

                # Собираем информацию об уровнях Лейдена
                leiden_levels = {}
                for level in range(6):
                    level_result = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        WITH e.community_{level} AS community, count(e) AS size
                        RETURN count(DISTINCT community) AS communities, 
                               avg(size) AS avg_size, 
                               max(size) AS max_size,
                               min(size) AS min_size
                    """)

                    level_record = await level_result.single()
                    if level_record and level_record["communities"]:
                        leiden_levels[level] = {
                            "communities": level_record["communities"],
                            "avg_size": float(level_record["avg_size"]),
                            "max_size": level_record["max_size"],
                            "min_size": level_record["min_size"]
                        }

                metadata = {
                    "created_timestamp": time.time(),
                    "nodes_count": stats_record["nodes"],
                    "relationships_count": stats_record["relationships"],
                    "leiden_levels": leiden_levels,
                    "chunks_processed": len(self.chunks),
                    "version": "1.0"
                }

                # Сохраняем в файл
                with open(self.graph_metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)

                logger.info(f"GraphRAG: Метаданные сохранены в {self.graph_metadata_file}")

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка сохранения метаданных: {e}")

    async def _load_graph_metadata(self):
        """Загружает метаданные графа из файла."""
        try:
            if os.path.exists(self.graph_metadata_file):
                with open(self.graph_metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                logger.info(f"GraphRAG: Метаданные загружены из {self.graph_metadata_file}")
                logger.info(
                    f"GraphRAG: Граф создан: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(metadata['created_timestamp']))}")
                logger.info(f"GraphRAG: Узлов: {metadata['nodes_count']}, связей: {metadata['relationships_count']}")
                logger.info(f"GraphRAG: Уровни Лейдена: {list(metadata['leiden_levels'].keys())}")
                return metadata
            else:
                logger.debug(f"GraphRAG: Файл метаданных {self.graph_metadata_file} не найден")
                return None

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка загрузки метаданных: {e}")
            return None

    async def _save_community_summaries_to_file(self):
        """Сохраняет сводки сообществ в файл."""
        try:
            if self.community_summaries:
                with open(self.community_summaries_file, 'w', encoding='utf-8') as f:
                    json.dump(self.community_summaries, f, indent=2, ensure_ascii=False)

                logger.info(f"GraphRAG: Сводки сообществ сохранены в {self.community_summaries_file}")
            else:
                logger.debug("GraphRAG: Нет сводок для сохранения")

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка сохранения сводок: {e}")

    async def _load_community_summaries_from_file(self):
        """Загружает сводки сообществ из файла."""
        try:
            if os.path.exists(self.community_summaries_file):
                with open(self.community_summaries_file, 'r', encoding='utf-8') as f:
                    self.community_summaries = json.load(f)

                logger.info(f"GraphRAG: Сводки сообществ загружены из {self.community_summaries_file}")
            else:
                logger.debug(f"GraphRAG: Файл сводок {self.community_summaries_file} не найден")
                self.community_summaries = {}

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка загрузки сводок: {e}")
            self.community_summaries = {}

    async def _build_graph_from_chunks(self, chunks: List[Any]) -> bool:
        """✅ МАСШТАБИРУЕМОЕ построение графа знаний из ВСЕХ чанков."""
        all_entities: dict = {}
        all_relations: List[dict] = []

        logger.info(f"GraphRAG: Обработка ВСЕХ {len(chunks)} чанков для извлечения сущностей...")

        successful_extractions = 0
        batch_size = 100  # ✅ УВЕЛИЧЕННЫЙ размер батча для эффективной обработки больших объемов

        for batch_start in range(0, len(chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(chunks))
            batch_chunks = chunks[batch_start:batch_end]

            logger.info(
                f"GraphRAG: Обработка батча {batch_start}-{batch_end}/{len(chunks)} ({((batch_end / len(chunks)) * 100):.1f}%)")

            for i, chunk in enumerate(batch_chunks):
                actual_i = batch_start + i

                try:
                    text = chunk.page_content[:1000]

                    prompt = self._entity_extraction_prompt.format(text=text)
                    raw_response = self.rm.llm.invoke(prompt)
                    cleaned_response = self._clean_json_response(raw_response)

                    if cleaned_response:
                        data = json.loads(cleaned_response)
                        successful_extractions += 1
                    else:
                        data = self._simple_entity_extraction(text, actual_i)

                    for entity in data.get("entities", []):
                        if not entity.get("text"):
                            continue

                        entity_key = entity["text"].lower().strip()
                        if entity_key not in all_entities:
                            entity["doc_ids"] = {actual_i}
                            all_entities[entity_key] = entity
                        else:
                            all_entities[entity_key]["doc_ids"].add(actual_i)

                    for relation in data.get("relations", []):
                        if not relation.get("source") or not relation.get("target"):
                            continue
                        relation["doc_id"] = actual_i
                        all_relations.append(relation)

                except json.JSONDecodeError as e:
                    logger.warning(f"GraphRAG: Ошибка парсинга JSON для чанка {actual_i}: {e}")
                    try:
                        fallback_data = self._simple_entity_extraction(chunk.page_content[:1000], actual_i)
                        for entity in fallback_data.get("entities", []):
                            entity_key = entity["text"].lower().strip()
                            if entity_key not in all_entities:
                                entity["doc_ids"] = {actual_i}
                                all_entities[entity_key] = entity
                            else:
                                all_entities[entity_key]["doc_ids"].add(actual_i)
                    except Exception:
                        pass
                    continue

                except Exception as e:
                    logger.warning(f"GraphRAG: Ошибка обработки чанка {actual_i}: {e}")
                    continue

        logger.info(f"GraphRAG: Успешных извлечений JSON: {successful_extractions}/{len(chunks)}")
        logger.info(f"GraphRAG: Извлечено {len(all_entities)} уникальных сущностей и {len(all_relations)} связей")

        if not all_entities:
            logger.warning("GraphRAG: Не извлечено ни одной сущности")
            return False

        try:
            await self._create_graph_in_neo4j_with_weights(all_entities, all_relations)
            logger.info("GraphRAG: Граф успешно создан в Neo4j")
        except Exception as e:
            logger.error(f"GraphRAG: Ошибка создания графа: {e}")
            return False

        # ✅ Используем алгоритм Лейдена для GDS 2.20
        try:
            await self._create_leiden_communities_gds_2_20()
            logger.info("GraphRAG: Алгоритм Лейдена успешно завершен")
        except Exception as e:
            logger.warning(f"GraphRAG: Не удалось создать сообщества Лейдена: {e}")

        return True

    def _simple_entity_extraction(self, text: str, doc_id: int) -> dict:
        """Простое извлечение сущностей без LLM."""
        words = re.findall(r'\b[А-ЯЁ][а-яё]+\b|\b[A-Z][a-z]+\b', text)

        entities = []
        for word in set(words[:10]):
            if len(word) > 3:
                entities.append({
                    "text": word,
                    "label": "Concept",
                    "description": f"Сущность из документа {doc_id}"
                })

        return {
            "entities": entities,
            "relations": []
        }

    def _clean_json_response(self, response: str) -> str:
        """Очищает ответ LLM для парсинга JSON."""
        if not response or not response.strip():
            logger.debug("GraphRAG: Пустой ответ от LLM")
            return ""

        text = response.strip()

        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1

        if self._debug_count <= 3:
            logger.debug(f"GraphRAG: Сырой ответ LLM #{self._debug_count}: {text[:200]}...")

        if "```" in text:
            lines = text.splitlines()
            code_lines = []
            in_code = False

            for line in lines:
                line_stripped = line.strip()
                if line_stripped.startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)

            if code_lines:
                text = "\n".join(code_lines).strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}") + 1

        if start_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx]
        else:
            start_idx = text.find("[")
            end_idx = text.rfind("]") + 1
            if start_idx != -1 and end_idx > start_idx:
                text = text[start_idx:end_idx]
            else:
                logger.debug(f"GraphRAG: Не найдена JSON структура в ответе: {text[:100]}...")
                return ""

        text = text.replace("``````", "").strip()

        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            logger.debug(f"GraphRAG: Невалидный JSON после очистки: {text[:100]}...")
            return ""

    async def _create_graph_in_neo4j_with_weights(self, entities: dict, relations: List[dict]):
        """✅ МАСШТАБИРУЕМОЕ создание графа в Neo4j с обязательными весами связей для GDS 2.20."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.error("GraphRAG: Нет активного event loop для создания графа")
            return

        if not self.graph_driver:
            logger.error("GraphRAG: Neo4j драйвер недоступен")
            return

        try:
            async with self.graph_driver.session() as session:
                # ✅ ИСПРАВЛЕННОЕ создание ограничений для Neo4j 5.x
                try:
                    await session.run(
                        "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                    logger.info("GraphRAG: Ограничение уникальности создано")
                except Exception as e:
                    logger.warning(f"GraphRAG: Не удалось создать ограничение: {e}")

                logger.info("GraphRAG: Создание сущностей в Neo4j...")
                entity_list = list(entities.values())
                batch_size = 200  # ✅ УВЕЛИЧЕННЫЙ batch_size для больших объемов

                for i in range(0, len(entity_list), batch_size):
                    batch = entity_list[i:i + batch_size]
                    entity_batch = []

                    for entity in batch:
                        entity_batch.append({
                            "name": str(entity["text"])[:100],
                            "label": str(entity.get("label", "Unknown"))[:50],
                            "description": str(entity.get("description", ""))[:200],
                            "doc_ids": list(entity["doc_ids"])
                        })

                    try:
                        await session.run(
                            "UNWIND $batch AS entity "
                            "CREATE (e:Entity {name: entity.name, label: entity.label, description: entity.description, doc_ids: entity.doc_ids})",
                            batch=entity_batch
                        )
                        logger.info(
                            f"GraphRAG: Создан батч сущностей {i // batch_size + 1}/{(len(entity_list) - 1) // batch_size + 1}")
                    except Exception as e:
                        logger.error(f"GraphRAG: Ошибка создания батча сущностей: {e}")
                        continue

                logger.info("GraphRAG: Создание связей с обязательными весами в Neo4j...")
                if relations:
                    for i in range(0, len(relations), batch_size):
                        batch = relations[i:i + batch_size]
                        relation_batch = []

                        for relation in batch:
                            if relation.get("source") and relation.get("target"):
                                relation_batch.append({
                                    "source": str(relation["source"])[:100],
                                    "target": str(relation["target"])[:100],
                                    "type": str(relation.get("type", "RELATED"))[:50],
                                    "description": str(relation.get("description", ""))[:200],
                                    "doc_id": relation["doc_id"],
                                    "weight": 1.0  # ✅ КРИТИЧЕСКИ ВАЖНО: Обязательный вес для GDS 2.20
                                })

                        if relation_batch:
                            try:
                                # ✅ ИСПРАВЛЕННЫЙ запрос с обязательными весами
                                await session.run(
                                    """
                                    UNWIND $batch AS rel
                                    MATCH (a:Entity {name: rel.source})
                                    WITH rel, a
                                    MATCH (b:Entity {name: rel.target})
                                    CREATE (a)-[:RELATED {
                                        type: rel.type, 
                                        description: rel.description, 
                                        doc_id: rel.doc_id,
                                        weight: rel.weight
                                    }]->(b)
                                    """,
                                    batch=relation_batch
                                )
                                logger.info(
                                    f"GraphRAG: Создан батч связей {i // batch_size + 1}/{(len(relations) - 1) // batch_size + 1}")
                            except Exception as e:
                                logger.warning(f"GraphRAG: Ошибка создания батча связей: {e}")
                                continue

                logger.info("GraphRAG: Граф с весами успешно создан в Neo4j")

        except Exception as e:
            logger.error(f"GraphRAG: Критическая ошибка создания графа: {e}")

    async def _create_leiden_communities_gds_2_20(self):
        """✅ МАСШТАБИРУЕМЫЙ алгоритм Лейдена для больших графов с проверкой размера."""
        async with self.graph_driver.session() as session:
            try:
                # Проверка доступности GDS
                try:
                    result = await session.run(
                        "CALL gds.debug.sysInfo() YIELD key, value WHERE key = 'gdsVersion' RETURN value")
                    record = await result.single()
                    if record:
                        gds_version = record["value"]
                        logger.info(f"GraphRAG: ✅ GDS доступен, версия: {gds_version}")
                    else:
                        raise Exception("GDS version not found")
                except Exception as e:
                    logger.warning(f"GraphRAG: GDS недоступен ({e}), используем упрощенный подход")
                    await self._create_simple_communities()
                    return

                # ✅ Проверка размера графа для алгоритма Лейдена
                node_count_result = await session.run("MATCH (e:Entity) RETURN count(e) AS node_count")
                node_count_record = await node_count_result.single()
                node_count = node_count_record["node_count"] if node_count_record else 0

                edge_count_result = await session.run("MATCH ()-[r:RELATED]->() RETURN count(r) AS edge_count")
                edge_count_record = await edge_count_result.single()
                edge_count = edge_count_record["edge_count"] if edge_count_record else 0

                logger.info(f"GraphRAG: Размер графа: {node_count} узлов, {edge_count} связей")

                # ✅ Минимальные требования для алгоритма Лейдена
                if node_count < 10 or edge_count < 15:
                    logger.warning(f"GraphRAG: Граф слишком мал для алгоритма Лейдена (минимум: 10 узлов, 15 связей). "
                                   f"Текущий размер: {node_count} узлов, {edge_count} связей. Используем fallback.")
                    await self._create_simple_communities()
                    return

                # ✅ Создание проекции графа с неориентированными связями
                try:
                    await session.run("""
                        CALL gds.graph.project('knowledge_graph', 'Entity', {
                            RELATED: {
                                orientation: 'UNDIRECTED',
                                properties: {
                                    weight: {
                                        property: 'weight',
                                        defaultValue: 1.0
                                    }
                                }
                            }
                        })
                    """)
                    logger.info("GraphRAG: ✅ Проекция графа создана с неориентированными связями")
                except Exception as e:
                    logger.warning(f"GraphRAG: Ошибка создания проекции с весами: {e}")
                    try:
                        await session.run("""
                            CALL gds.graph.project('knowledge_graph', 'Entity', {
                                RELATED: {
                                    orientation: 'UNDIRECTED'
                                }
                            })
                        """)
                        logger.info("GraphRAG: ✅ Проекция графа создана неориентированными связями (без весов)")
                    except Exception as e2:
                        logger.error(f"GraphRAG: Критическая ошибка создания проекции: {e2}")
                        await self._create_simple_communities()
                        return

                # ✅ Алгоритм Лейдена с оптимизированными параметрами для больших графов
                gamma_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5]
                successful_levels = []

                for level, gamma in enumerate(gamma_values):
                    try:
                        logger.info(f"GraphRAG: Запуск Лейдена уровень {level} (gamma={gamma})...")
                        result = await session.run("""
                            CALL gds.leiden.write('knowledge_graph', {
                                writeProperty: $property,
                                gamma: $gamma,
                                maxLevels: 10,
                                tolerance: 1e-4,
                                randomSeed: 42
                            }) 
                            YIELD communityCount, modularity
                            RETURN communityCount, modularity
                        """, property=f"community_{level}", gamma=gamma)

                        record = await result.single()
                        if record:
                            successful_levels.append(level)
                            logger.info(f"GraphRAG: ✅ Уровень {level} (gamma={gamma}): "
                                        f"{record['communityCount']} сообществ, "
                                        f"модульность={record['modularity']:.3f}")

                    except Exception as e:
                        logger.warning(f"GraphRAG: ❌ Ошибка на уровне {level} (gamma={gamma}): {e}")
                        continue

                # Удаление проекции и обработка результатов
                try:
                    await session.run("CALL gds.graph.drop('knowledge_graph')")
                    logger.info("GraphRAG: ✅ Проекция графа удалена")
                except Exception as e:
                    logger.warning(f"GraphRAG: Ошибка удаления проекции: {e}")

                if successful_levels:
                    await self._compute_community_metrics_safe(successful_levels)
                    await self._generate_hierarchical_summaries_safe(successful_levels)
                    logger.info(f"GraphRAG: ✅ Алгоритм Лейдена успешно завершен для {len(successful_levels)} уровней")
                else:
                    logger.warning("GraphRAG: ❌ Ни один уровень Лейдена не создан, используем fallback")
                    await self._create_simple_communities()

            except Exception as e:
                logger.error(f"GraphRAG: ❌ Ошибка алгоритма Лейдена: {e}")
                await self._create_simple_communities()

    async def _compute_community_metrics_safe(self, successful_levels: List[int]):
        """✅ БЕЗОПАСНОЕ вычисление метрик - только для существующих свойств."""
        async with self.graph_driver.session() as session:
            try:
                for level in successful_levels:
                    # Проверяем существование свойства
                    check_result = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        RETURN count(e) AS count
                        LIMIT 1
                    """)

                    check_record = await check_result.single()
                    if not check_record or check_record["count"] == 0:
                        logger.debug(f"GraphRAG: Свойство community_{level} не найдено, пропускаем")
                        continue

                    # Вычисляем метрики
                    result = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        WITH e.community_{level} AS community, collect(e) AS entities
                        WITH community, entities, size(entities) AS community_size
                        RETURN avg(community_size) AS avg_size, 
                               max(community_size) AS max_size,
                               min(community_size) AS min_size,
                               count(DISTINCT community) AS total_communities
                    """)

                    record = await result.single()
                    if record and record["avg_size"] is not None:
                        logger.info(f"GraphRAG: ✅ Уровень {level} метрики: "
                                    f"сообществ={record['total_communities']}, "
                                    f"ср.размер={record['avg_size']:.1f}, "
                                    f"макс={record['max_size']}, мин={record['min_size']}")

            except Exception as e:
                logger.warning(f"GraphRAG: Ошибка вычисления метрик: {e}")

    async def _generate_hierarchical_summaries_safe(self, successful_levels: List[int]):
        """✅ МАСШТАБИРУЕМАЯ генерация сводок для больших графов."""
        async with self.graph_driver.session() as session:
            try:
                for level in successful_levels:
                    # Проверяем существование свойства
                    check_result = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        RETURN count(e) AS count
                        LIMIT 1
                    """)

                    check_record = await check_result.single()
                    if not check_record or check_record["count"] == 0:
                        logger.debug(f"GraphRAG: Свойство community_{level} не найдено, пропускаем сводки")
                        continue

                    # ✅ ОПТИМИЗИРОВАННАЯ генерация сводок - только для значимых сообществ
                    result = await session.run(f"""
                        MATCH (e:Entity) 
                        WHERE e.community_{level} IS NOT NULL
                        WITH e.community_{level} AS community, collect(e.name)[0..20] AS entity_names
                        WHERE size(entity_names) >= 3 AND size(entity_names) <= 20
                        RETURN community, entity_names
                        ORDER BY size(entity_names) DESC
                        LIMIT 50
                    """)

                    level_summaries = {}
                    community_count = 0

                    async for record in result:
                        community_id = f"{level}_{record['community']}"
                        entity_names = record["entity_names"]

                        try:
                            summary_prompt = f"""
Создай краткую сводку для сообщества уровня {level} (gamma={[0.1, 0.3, 0.5, 0.8, 1.0, 1.5][level]}):

Сущности: {', '.join(entity_names)}

Опиши основную тематическую область в 2-3 предложениях.
"""
                            summary = self.rm.llm.invoke(summary_prompt)
                            level_summaries[community_id] = {
                                "level": level,
                                "summary": summary.strip(),
                                "entity_count": len(entity_names),
                                "entities": entity_names[:10]
                            }
                            community_count += 1

                        except Exception as e:
                            logger.warning(f"GraphRAG: Ошибка создания сводки для {community_id}: {e}")

                    if level_summaries:
                        self.community_summaries[f"level_{level}"] = level_summaries
                        logger.info(f"GraphRAG: ✅ Создано {len(level_summaries)} сводок для уровня {level}")

            except Exception as e:
                logger.warning(f"GraphRAG: Ошибка генерации иерархических сводок: {e}")

    async def _create_simple_communities(self):
        """Простые сообщества как fallback."""
        async with self.graph_driver.session() as session:
            try:
                await session.run(
                    "MATCH (e:Entity) "
                    "OPTIONAL MATCH (e)-[:RELATED]-(connected:Entity) "
                    "WITH e, collect(DISTINCT connected.name)[0..10] AS connections "
                    "SET e.community = size(connections)"
                )
                logger.info("GraphRAG: ✅ Созданы простые сообщества на основе связности (fallback)")
                await self._generate_community_summaries()

            except Exception as e:
                logger.warning(f"GraphRAG: Ошибка создания простых сообществ: {e}")

    async def _generate_community_summaries(self):
        """Генерирует сводки для простых сообществ."""
        async with self.graph_driver.session() as session:
            try:
                result = await session.run(
                    "MATCH (e:Entity) WHERE e.community IS NOT NULL "
                    "WITH e.community AS community, collect(e.name)[0..5] AS entity_names "
                    "WHERE size(entity_names) > 1 "
                    "RETURN community, entity_names "
                    "LIMIT 50"
                )

                async for record in result:
                    community_id = record["community"]
                    entity_names = record["entity_names"]

                    try:
                        summary_prompt = f"""
Создай краткую сводку для группы связанных сущностей:
{', '.join(entity_names)}

Опиши основную тему в 1-2 предложениях.
"""
                        summary = self.rm.llm.invoke(summary_prompt)
                        self.community_summaries[community_id] = summary.strip()

                    except Exception as e:
                        logger.warning(f"GraphRAG: Ошибка создания сводки для сообщества {community_id}: {e}")

            except Exception as e:
                logger.warning(f"GraphRAG: Ошибка генерации сводок сообществ: {e}")

    async def graph_search(self, query: str, k: int = 10) -> List[Any]:
        """✅ ИСПРАВЛЕННЫЙ поиск через граф знаний с проверкой существующих свойств (ВСЕ чанки)."""
        if not await self.verify_connection():
            return []

        try:
            prompt = self._query_entity_extraction_prompt.format(query=query)
            raw_response = self.rm.llm.invoke(prompt)
            cleaned_response = self._clean_json_response(raw_response)

            if not cleaned_response:
                query_words = [word.strip().lower() for word in query.split() if len(word.strip()) > 2]
                query_entities = query_words[:5]
            else:
                try:
                    query_entities = json.loads(cleaned_response)
                except:
                    query_entities = []

            if not query_entities:
                logger.debug("GraphRAG: Не удалось извлечь сущности из запроса")
                return []

            async with self.graph_driver.session() as session:
                # ✅ БЕЗОПАСНЫЙ поиск с проверкой существующих свойств
                result = await session.run(
                    """
                    MATCH (e:Entity) 
                    WHERE toLower(e.name) IN [entity IN $entities | toLower(entity)]
                    WITH e

                    // Ищем связанные сущности на разных уровнях иерархии (если они существуют)
                    OPTIONAL MATCH (related:Entity)
                    WHERE (
                        // Прямые связи
                        (e)-[:RELATED*1..2]-(related) OR
                        // Сообщества уровня 0 (если существует)
                        (e.community_0 IS NOT NULL AND e.community_0 = related.community_0) OR
                        // Сообщества уровня 1 (если существует)
                        (e.community_1 IS NOT NULL AND e.community_1 = related.community_1) OR
                        // Сообщества уровня 2 (если существует)
                        (e.community_2 IS NOT NULL AND e.community_2 = related.community_2) OR
                        // Простые сообщества (fallback)
                        (e.community IS NOT NULL AND e.community = related.community)
                    )

                    WITH DISTINCT coalesce(related, e) AS found_entity
                    WHERE found_entity.doc_ids IS NOT NULL
                    UNWIND found_entity.doc_ids AS doc_id
                    RETURN DISTINCT doc_id
                    LIMIT $limit
                    """,
                    entities=query_entities,
                    limit=k * 3
                )

                doc_ids = []
                async for record in result:
                    doc_ids.append(record["doc_id"])

            # ✅ ВАЖНО: Теперь возвращаем чанки из ВСЕГО набора
            relevant_chunks = []
            for doc_id in doc_ids[:k]:
                if 0 <= doc_id < len(self.chunks):
                    relevant_chunks.append(self.chunks[doc_id])

            logger.debug(
                f"GraphRAG: Найдено {len(relevant_chunks)} документов через граф для сущностей: {query_entities[:3]}...")
            return relevant_chunks

        except Exception as e:
            logger.error(f"GraphRAG: Ошибка поиска в графе: {e}")
            return []

    async def close(self):
        """Закрывает соединение с Neo4j."""
        if self.graph_driver:
            await self.graph_driver.close()
            logger.info("GraphRAG: Neo4j соединение закрыто")


# -----------------------------------------------------------------------------
# 11. ✅ ОБНОВЛЕННАЯ Основная RAG система с ColBERT и улучшениями
# -----------------------------------------------------------------------------

class EnhancedAsyncRAGSystem:
    """Расширенная RAG-система с поддержкой ColBERT и GraphRAG."""

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.rm = ResourceManager(cfg)
        self.cache = ContextAwareCacheManager(cfg.ttl_cache_sec)  # ✅ Новый кэш
        self.loader = AdvancedDocumentLoader()
        self.faiss = OptimizedFaissStore(cfg, self.rm)
        self.bm25: Optional[AsyncBM25Retriever] = None
        self.graph_rag = GraphRAGSystem(cfg, self.rm)
        self.use_graph = True
        self.llm_manager = None  # ✅ Инициализируем позже

        # ✅ Счетчик для предотвращения зацикливания запросов
        self._query_history = []
        self._max_history = 10

        try:
            self.graph_driver = AsyncGraphDatabase.driver(
                cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)
            )
        except Exception:
            self.graph_driver = None

        self._init_lock = asyncio.Lock()
        self._initialized = False
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_response_time": 0.0,
            "colbert_usage": 0
        }

    def _add_to_query_history(self, query: str):
        """Добавляет запрос в историю для предотвращения зацикливания"""
        query_hash = md5_text(query)
        self._query_history.append({
            'hash': query_hash,
            'timestamp': time.time()
        })

        # Сохраняем только последние запросы
        if len(self._query_history) > self._max_history:
            self._query_history.pop(0)

    def _is_repeated_query(self, query: str, threshold: float = 0.8) -> bool:
        """Проверяет, является ли запрос повторяющимся"""
        current_hash = md5_text(query)
        current_time = time.time()

        for hist_query in self._query_history:
            # Проверяем запросы за последние 5 минут
            if current_time - hist_query['timestamp'] < 300:
                if hist_query['hash'] == current_hash:
                    logger.warning("Обнаружен повторяющийся запрос")
                    return True
        return False

    async def initialize(self):
        """Инициализация всех компонентов системы."""
        async with self._init_lock:
            if self._initialized:
                return

            try:
                logger.info("Начало инициализации расширенной RAG-системы...")
                documents = self.loader.load_documents(self.cfg.folder_path)
                if not documents:
                    raise RuntimeError("Не загружено ни одного документа")

                chunks = self.loader.chunk_documents(documents, self.cfg, self.rm)
                if not chunks:
                    raise RuntimeError("Не создано ни одного чанка")

                # ✅ ВСЕ компоненты используют ВСЕ чанки
                logger.info(f"📚 FAISS, BM25, ColBERT и GraphRAG: Используется полная коллекция из {len(chunks)} чанков")

                await self.faiss.build(chunks)
                self.bm25 = AsyncBM25Retriever(chunks)

                # ✅ Инициализируем динамический LLM менеджер
                self.llm_manager = DynamicLLMManager(self.rm.llm)

                # ✅ Проверяем доступность ColBERT
                if self.cfg.use_colbert_reranker and COLBERT_AVAILABLE:
                    test_reranker = self.rm.colbert_reranker
                    if test_reranker.colbert_model is not None:
                        logger.info("✅ ColBERT reranker готов к использованию")
                    else:
                        logger.warning("❌ ColBERT недоступен, используем базовое ранжирование")
                        self.cfg.use_colbert_reranker = False

                # ✅ GraphRAG теперь использует ВСЕ чанки с сохранением/загрузкой
                if self.use_graph:
                    logger.info(f"🎯 GraphRAG: Обработка ВСЕХ {len(chunks)} чанков для полноценного графа знаний")
                    graph_success = await self.graph_rag.initialize_graph(chunks)
                    if graph_success:
                        logger.info("✅ GraphRAG успешно инициализирован на всех чанках")
                    else:
                        logger.warning("❌ GraphRAG не удалось инициализировать, используем только FAISS+BM25+ColBERT")
                        self.use_graph = False

                self._initialized = True
                logger.info("✅ Расширенная RAG-система успешно инициализирована")
                logger.info(f"📊 Итоговая статистика:")
                logger.info(f"   • FAISS + BM25 + ColBERT + GraphRAG: {len(chunks)} чанков")
                logger.info(f"   • ColBERT reranker: {'✅ Включен' if self.cfg.use_colbert_reranker else '❌ Отключен'}")
                logger.info(
                    f"   • GraphRAG сохранение: {'✅ Включено' if self.cfg.save_graph_structure else '❌ Отключено'}")

            except Exception as e:
                logger.error(f"Ошибка инициализации расширенной RAG-системы: {e}")
                raise

    def _clean_and_deduplicate_response(self, response: str) -> str:
        """Очищает ответ от повторений и нежелательных элементов"""
        if not response:
            return response

        # Удаляем избыточные переносы строк
        response = re.sub(r'\n{3,}', '\n\n', response)

        # Удаляем повторяющиеся фразы
        lines = response.split('\n')
        cleaned_lines = []
        seen_lines = set()

        for line in lines:
            line = line.strip()
            if line and line not in seen_lines:
                cleaned_lines.append(line)
                seen_lines.add(line)
            elif not line:  # Пустые строки оставляем
                cleaned_lines.append(line)

        # Объединяем обратно
        cleaned_response = '\n'.join(cleaned_lines)

        # Удаляем фразы-индикаторы повторений
        repetition_patterns = [
            r'(Как я уже говорил,?\s*)',
            r'(Повторяю,?\s*)',
            r'(Еще раз,?\s*)',
            r'(Снова,?\s*)',
            r'(As I mentioned,?\s*)',
            r'(Again,?\s*)',
        ]

        for pattern in repetition_patterns:
            cleaned_response = re.sub(pattern, '', cleaned_response, flags=re.IGNORECASE)

        return cleaned_response.strip()

    async def process_query(self, query: str, k: int = 5, include_time: bool = True, user_id: str = None) -> str:
        """Обработка запроса с учетом пользователя и контекста."""
        start_time = time.time()
        self.metrics["total_queries"] += 1

        try:
            await self.initialize()

            # ✅ Проверяем на повторяющиеся запросы
            if self._is_repeated_query(query):
                logger.warning(f"Повторяющийся запрос обнаружен: {query[:50]}...")
                return "Этот вопрос уже был недавно задан. Попробуйте сформулировать его по-другому."

            # Добавляем в историю
            self._add_to_query_history(query)

            # ✅ Проверяем кэш с учетом пользователя
            cached = self.cache.get(query, user_id)
            if cached:
                logger.debug(f"Cache hit для пользователя {user_id}: {query[:50]}...")
                response_time = time.time() - start_time
                if include_time:
                    return f"{cached}\n⏱ Время ответа: {response_time:.2f} сек (из кэша)"
                return cached

            logger.info(f"Обработка запроса для пользователя {user_id}: {query[:100]}...")

            search_tasks = [
                self.faiss.search(query, k=15),
                self.bm25.retrieve(query, k=15)
            ]

            if self.use_graph:
                search_tasks.append(self.graph_rag.graph_search(query, k))

            results = await asyncio.gather(*search_tasks, return_exceptions=True)

            faiss_results = results[0] if not isinstance(results[0], Exception) else []
            bm25_results = results[1] if not isinstance(results[1], Exception) else []
            graph_results = results[2] if self.use_graph and len(results) > 2 and not isinstance(results[2],
                                                                                                 Exception) else []

            # ✅ ЛОГИРОВАНИЕ результатов поиска
            logger.info(
                f"🔍 Результаты поиска: FAISS={len(faiss_results)}, BM25={len(bm25_results)}, Graph={len(graph_results)}")

            if not faiss_results and not bm25_results and not graph_results:
                logger.warning("Не найдено документов ни одним методом поиска")
                return "Извините, не удалось найти релевантную информацию по вашему запросу."

            all_result_lists = [faiss_results, bm25_results]
            if graph_results:
                all_result_lists.append(graph_results)

            fused = self._rrf_fusion(all_result_lists, k_rrf=60)
            unique_docs = self._remove_duplicates(fused)

            if len(unique_docs) > k:
                # ✅ ИСПОЛЬЗУЕМ COLBERT ДЛЯ ПЕРЕРАНЖИРОВАНИЯ
                if self.cfg.use_colbert_reranker:
                    reranked = self.rm.colbert_reranker.rerank(query, unique_docs[:15], k=k)
                    self.metrics["colbert_usage"] += 1
                    logger.debug(f"ColBERT переранжирование: {len(unique_docs)} -> {len(reranked)}")
                else:
                    reranked = unique_docs[:k]
            else:
                reranked = unique_docs

            if not reranked:
                logger.warning("После переранжировки не осталось документов")
                return "Не найдено релевантных документов после анализа."

            # ✅ Используем динамический LLM менеджер
            answer = self._generate_answer_with_diversity(query, reranked, user_id)

            # ✅ ОЧИСТКА ОТВЕТА ОТ ПОВТОРЕНИЙ
            cleaned_answer = self._clean_and_deduplicate_response(answer)

            # ✅ Кэшируем с учетом пользователя
            self.cache.set(query, cleaned_answer, user_id)

            response_time = time.time() - start_time
            self.metrics["successful_queries"] += 1
            self.metrics["total_response_time"] += response_time

            logger.info(f"Запрос обработан за {response_time:.2f} сек для пользователя {user_id}")

            if include_time:
                return f"{cleaned_answer}\n⏱ Время ответа: {response_time:.2f} сек"
            return cleaned_answer

        except Exception as e:
            self.metrics["failed_queries"] += 1
            logger.error(f"Ошибка обработки запроса: {e}")
            return f"Извините, произошла ошибка при обработке запроса: {str(e)}"

    def _generate_answer_with_diversity(self, query: str, docs: List[Any], user_id: str = None) -> str:
        """Генерирует ответ с учетом разнообразия для пользователя."""
        try:
            contexts = []
            for doc in docs:
                if hasattr(doc, 'page_content'):
                    contexts.append(doc.page_content)
                else:
                    contexts.append(str(doc))

            context = "\n---\n".join(contexts)

            # ✅ УЛУЧШЕННЫЙ ПРОМТ ДЛЯ ПРЕДОТВРАЩЕНИЯ ПОВТОРЕНИЙ
            prompt = f"""Ты — интеллектуальный ассистент, который предоставляет точные, ясные и краткие ответы на русском языке.

        КРИТИЧЕСКИ ВАЖНО:
        - Отвечай ТОЛЬКО на основе предоставленного контекста
        - НЕ повторяй одну и ту же информацию несколько раз
        - Избегай фраз типа "как я уже говорил", "повторяю", "еще раз"
        - Структурируй ответ логично и последовательно
        - Будь конкретным и избегай общих рассуждений

        Инструкции:
        1. Дай прямой ответ на вопрос
        2. Используй конкретные факты из контекста
        3. Если контекста недостаточно, скажи об этом честно
        4. Расшифровывай сокращения при первом упоминании
        5. Заканчивай ответ, когда информация исчерпана

        Контекст:
        {context}

        Вопрос: {query}

        Ответ:"""

            # ✅ Используем динамический LLM менеджер
            answer = self.llm_manager.generate_response(prompt, user_id)
            self.rm.gpu_clear()

            return answer

        except Exception as e:
            logger.error(f"Ошибка генерации ответа: {e}")
            return "Извините, произошла ошибка при генерации ответа."

    def clear_user_context(self, user_id: str):
        """Очистка контекста пользователя"""
        if hasattr(self.cache, 'clear_user_context'):
            self.cache.clear_user_context(user_id)
        if hasattr(self.llm_manager, 'clear_user_history'):
            self.llm_manager.clear_user_history(user_id)
        logger.info(f"Контекст пользователя {user_id} очищен")

    def _rrf_fusion(self, result_lists: List[List[Any]], k_rrf: int = 60) -> List[Any]:
        """Reciprocal Rank Fusion для объединения результатов поиска."""
        doc_scores = {}
        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_hash = md5_text(doc.page_content if hasattr(doc, 'page_content') else str(doc))
                doc_scores[doc_hash] = doc_scores.get(doc_hash, 0.0) + 1.0 / (rank + 1 + k_rrf)

        docs_dict = {}
        for result_list in result_lists:
            for doc in result_list:
                doc_hash = md5_text(doc.page_content if hasattr(doc, 'page_content') else str(doc))
                docs_dict.setdefault(doc_hash, doc)

        sorted_docs = []
        for doc_hash, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_hash in docs_dict:
                sorted_docs.append(docs_dict[doc_hash])

        return sorted_docs

    def _remove_duplicates(self, docs: List[Any]) -> List[Any]:
        """Удаляет дубликаты документов по содержимому."""
        seen = set()
        unique = []

        for doc in docs:
            content_hash = md5_text(doc.page_content if hasattr(doc, 'page_content') else str(doc))
            if content_hash not in seen:
                seen.add(content_hash)
                unique.append(doc)

        logger.debug(f"Удалены дубликаты: {len(docs)} -> {len(unique)}")
        return unique

    def get_metrics(self) -> dict:
        """Возвращает метрики производительности системы."""
        total_queries = max(self.metrics["total_queries"], 1)
        success_rate = (self.metrics["successful_queries"] / total_queries) * 100
        avg_response_time = self.metrics["total_response_time"] / max(self.metrics["successful_queries"], 1)

        return {
            **self.metrics,
            "success_rate": f"{success_rate:.2f}%",
            "average_response_time": f"{avg_response_time:.2f}s",
            "graph_enabled": self.use_graph,
            "colbert_enabled": self.cfg.use_colbert_reranker,
            **self.cache.get_stats()
        }

    async def close(self):
        """Закрывает соединения и освобождает ресурсы."""
        try:
            logger.info("🔄 Закрытие RAG-системы...")

            if self.use_graph and hasattr(self, 'graph_rag'):
                await self.graph_rag.close()

            if hasattr(self, 'graph_driver') and self.graph_driver:
                await self.graph_driver.close()

            self.rm.gpu_clear()

            logger.info("✅ RAG-система корректно закрыта")

        except Exception as e:
            logger.error(f"❌ Ошибка при закрытии RAG-системы: {e}")

    # -----------------------------------------------------------------------------
    # 12. Главная асинхронная RAG-система (обратная совместимость)
    # -----------------------------------------------------------------------------

class AsyncRAGSystem(EnhancedAsyncRAGSystem):
    """Обратная совместимость с оригинальным классом."""
    pass

    # -----------------------------------------------------------------------------
    # 13. ✅ ОБНОВЛЕННЫЙ пример использования с ColBERT и улучшениями
    # -----------------------------------------------------------------------------

async def main():
        """Интерактивный режим работы с расширенной RAG-системой."""
        cfg = RAGConfig(
            folder_path=os.getenv("RAG_DOCS_PATH", "docs/"),
            faiss_path=os.getenv("RAG_FAISS_PATH", "vector_store/faiss_index"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "neo4jneo4j"),
            use_semantic_chunking=True,
            # ✅ НАСТРОЙКИ ДЛЯ ПЕРВОГО/ПОВТОРНОГО ЗАПУСКА
            force_rebuild_graph=True,  # При первом запуске True, потом False
            save_graph_structure=True,  # Сохранять граф
            auto_load_existing=True,  # Автозагрузка при повторных запусках
            graph_metadata_dir="graph_data/",  # Папка для метаданных
            # ✅ ВКЛЮЧАЕМ COLBERT
            use_colbert_reranker=True,
            colbert_model="colbert-ir/colbertv2.0",
            # ✅ ОПТИМАЛЬНЫЕ НАСТРОЙКИ LLM
            llm_temperature=0.4,
            llm_repeat_penalty=1.05,
            llm_repeat_last_n=128,
            llm_top_p=0.85,
            llm_top_k=40,
        )

        rag = EnhancedAsyncRAGSystem(cfg)

        print("\n🚀 Добро пожаловать в расширенную RAG-систему с ColBERT, GraphRAG и сохранением структуры!")
        print("📁 Поддерживаемые форматы: PDF, DOCX, TXT")
        print("⚡ Возможности:")
        print("   • Векторный поиск FAISS с Qwen3-Embedding-0.6B")
        print("   • BM25 лексический поиск")
        print("   • ColBERT переранжирование для максимальной точности")
        print("   • GraphRAG с извлечением сущностей и построением графа знаний")
        print("   • Алгоритм Лейдена для иерархических сообществ (как в Microsoft GraphRAG)")
        print("   • 🆕 СОХРАНЕНИЕ и ЗАГРУЗКА структуры графа и сообществ Лейдена")
        print("   • Защита от повторений и контекстная память для каждого пользователя")
        print("   • Многоуровневое кэширование с учетом пользовательского контекста")
        print("✅ Исправления:")
        print("   • Neo4j GDS 2.20 совместимость (gamma вместо resolution)")
        print("   • Неориентированные связи для алгоритма Лейдена")
        print("   • Оптимизированные настройки LLM против повторений")
        print("   • Динамическая адаптация параметров генерации")
        print("\n💾 Сохранение графа:")
        print(f"   • При первом запуске: граф будет создан и сохранен")
        print(f"   • При повторных запусках: граф будет загружен за ~30-60 секунд")
        print(f"   • Метаданные сохраняются в: {cfg.graph_metadata_dir}")
        print("\n📚 Система работает с ПОЛНОЙ коллекцией документов")
        print("Команды:")
        print("  • Введите запрос для поиска")
        print("  • 'clear' - очистить контекст пользователя")
        print("  • 'metrics' - показать статистику системы")
        print("  • 'exit' - выход\n")

        try:
            user_id = "default_user"  # В реальном приложении берется из Telegram

            while True:
                query = input("🔍 Ваш запрос: ").strip()

                if not query:
                    continue

                if query.lower() == 'exit':
                    break

                if query.lower() == 'clear':
                    rag.clear_user_context(user_id)
                    print("🧹 Контекст пользователя очищен!\n")
                    continue

                if query.lower() == 'metrics':
                    metrics = rag.get_metrics()
                    print("\n📊 Метрики системы:")
                    print("=" * 50)
                    for key, value in metrics.items():
                        if key == "colbert_usage":
                            print(f"  📈 ColBERT использований: {value}")
                        elif key == "graph_enabled":
                            print(f"  🕸️ GraphRAG: {'✅ Включен' if value else '❌ Отключен'}")
                        elif key == "colbert_enabled":
                            print(f"  🎯 ColBERT: {'✅ Включен' if value else '❌ Отключен'}")
                        else:
                            print(f"  {key}: {value}")
                    print("=" * 50)
                    print()
                    continue

                try:
                    print("⏳ Обрабатываю запрос...")
                    answer = await rag.process_query(query, user_id=user_id, include_time=True)
                    print(f"\n💡 {answer}\n")
                    print("-" * 80)
                except Exception as e:
                    print(f"\n❌ Ошибка: {e}\n")

        except KeyboardInterrupt:
            print("\n\n👋 Завершение работы...")
        finally:
            await rag.close()
            print("✅ Система корректно завершена")

    # -----------------------------------------------------------------------------
    # 14. ✅ ТОЧКА ВХОДА
    # -----------------------------------------------------------------------------

if __name__ == "__main__":
        # Проверяем доступность зависимостей
        print("🔍 Проверка зависимостей...")

        required_packages = {
            "faiss": "faiss-cpu",
            "torch": "torch",
            "langchain_community": "langchain-community",
            "langchain_huggingface": "langchain-huggingface",
            "langchain_ollama": "langchain-ollama",
            "neo4j": "neo4j",
            "cachetools": "cachetools",
        }

        missing_packages = []
        for package, install_name in required_packages.items():
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (установите: pip install {install_name})")
                missing_packages.append(install_name)

        if not COLBERT_AVAILABLE:
            print("❌ ragatouille (установите: pip install ragatouille)")
            missing_packages.append("ragatouille")
        else:
            print("✅ ragatouille (ColBERT)")

        if missing_packages:
            print(f"\n🚨 Установите недостающие пакеты:")
            print(f"pip install {' '.join(missing_packages)}")
            print("\nТакже убедитесь что:")
            print("• Neo4j запущен и доступен")
            print("• Ollama запущен с моделью qwen2.5vl:7b")
            print("• Папка 'docs/' содержит ваши документы")
            exit(1)

        print("✅ Все зависимости установлены!")
        print("\n🚀 Запуск расширенной RAG-системы...\n")

        # Запуск основного цикла
        asyncio.run(main())

