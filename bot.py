# telegram_bot.py
import logging
import asyncio
import yaml
import os
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    CallbackQueryHandler
)
from telegram.error import TelegramError
from typing import Dict, Any, Optional
from datetime import datetime

# Импортируем новую асинхронную RAG-систему
from main import AsyncRAGSystem, RAGConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s - ChatID: %(chat_id)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные
_config = {}
_user_responses = {}  # {user_id: {"query": str, "response": str, "message_id": int, "timestamp": datetime}}
_rag_system: Optional[AsyncRAGSystem] = None
_user_sessions = {}  # Для отслеживания активных сессий пользователей


def load_config() -> Dict[str, Any]:
    """Загружает конфигурацию из файла settings.yaml."""
    try:
        with open("settings.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info("Конфигурация успешно загружена из settings.yaml")
        return config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {str(e)}")
        # Возвращаем конфигурацию по умолчанию
        return {
            "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
            "folder_path": os.getenv("RAG_DOCS_PATH", "docs/"),
            "faiss_path": os.getenv("RAG_FAISS_PATH", "vector_store/faiss_index"),
            "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
            "neo4j_password": os.getenv("NEO4J_PASSWORD", "neo4jneo4j"),
        }


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    try:
        if not update.effective_user or not update.message or not update.effective_chat:
            logger.error("Некорректные данные в обработчике start")
            return

        user = update.effective_user
        chat_id = update.effective_chat.id
        logger.info(
            f"Пользователь {user.id} ({user.username or 'без имени'}) запустил бота",
            extra={"chat_id": chat_id}
        )

        # Инициализируем сессию пользователя
        _user_sessions[user.id] = {
            "start_time": datetime.now(),
            "queries_count": 0,
            "last_activity": datetime.now()
        }

        welcome_message = (
            f"👋 Здравствуйте, {user.first_name or 'пользователь'}!\n\n"
            "🤖 Я — интеллектуальный ассистент с улучшенной RAG-системой.\n"
            "📚 Работаю с базой знаний и использую:\n"
            "  • Асинхронную обработку запросов\n"
            "  • Кэширование для быстрых ответов\n"
            "  • Оптимизированный векторный поиск\n"
            "  • Гибридную агрегацию результатов\n\n"
            "💬 Задайте мне вопрос, и я найду ответ в документах!\n\n"
            "ℹ️ Для получения справки введите /help\n"
            "📊 Для просмотра статистики введите /stats"
        )
        await update.message.reply_text(welcome_message)

    except Exception as e:
        logger.error(f"Ошибка в обработчике start: {str(e)}", exc_info=True)
        if update.message:
            await update.message.reply_text(
                "❌ Произошла ошибка при обработке команды. Попробуйте позже."
            )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    try:
        if not update.effective_chat or not update.message:
            logger.error("Некорректные данные в обработчике help_command")
            return

        chat_id = update.effective_chat.id
        logger.info("Запрошена справка", extra={"chat_id": chat_id})

        help_text = (
            "🔍 *Как пользоваться ботом:*\n\n"
            "1️⃣ Задайте вопрос в чате (поддерживаются PDF, DOCX, TXT документы)\n"
            "2️⃣ Бот найдет информацию с помощью улучшенного поиска:\n"
            "   • Векторный поиск (FAISS)\n"
            "   • BM25 алгоритм\n"
            "   • Графовая база знаний (Neo4j)\n"
            "3️⃣ Получите точный ответ с источниками\n"
            "4️⃣ Оцените ответ кнопками 👍/👎\n"
            "5️⃣ Посмотрите источники информации\n\n"
            "*🛠 Доступные команды:*\n"
            "• `/start` — Начать работу с ботом\n"
            "• `/help` — Показать эту справку\n"
            "• `/stats` — Статистика работы системы\n"
            "• `/clear` — Очистить историю запросов\n\n"
            "*⚡ Особенности:*\n"
            "• Быстрые ответы благодаря кэшированию\n"
            "• Высокая точность благодаря переранжировке\n"
            "• Поддержка больших документов\n"
            "• Мониторинг производительности"
        )
        await update.message.reply_text(help_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Ошибка в обработчике help_command: {str(e)}", exc_info=True)
        if update.message:
            await update.message.reply_text(
                "❌ Произошла ошибка при обработке команды. Попробуйте позже."
            )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /stats - показывает статистику системы."""
    global _rag_system

    try:
        if not update.effective_chat or not update.message:
            return

        chat_id = update.effective_chat.id
        user_id = update.effective_user.id if update.effective_user else 0

        logger.info("Запрошена статистика", extra={"chat_id": chat_id})

        # Отправляем индикатор обработки
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        if not _rag_system:
            await update.message.reply_text("❌ RAG-система не инициализирована")
            return

        # Получаем метрики системы
        try:
            metrics = _rag_system.get_metrics()

            # Получаем статистику пользователя
            user_session = _user_sessions.get(user_id, {})
            user_queries = user_session.get("queries_count", 0)

            stats_text = (
                "📊 *Статистика RAG-системы:*\n\n"
                f"🔢 Всего запросов: `{metrics.get('total_queries', 0)}`\n"
                f"✅ Успешных: `{metrics.get('successful_queries', 0)}`\n"
                f"❌ Неудачных: `{metrics.get('failed_queries', 0)}`\n"
                f"📈 Успешность: `{metrics.get('success_rate', '0%')}`\n"
                f"⏱ Среднее время ответа: `{metrics.get('average_response_time', '0s')}`\n\n"

                "💾 *Кэширование:*\n"
                f"🎯 Попаданий в кэш: `{metrics.get('cache_hits', 0)}`\n"
                f"💨 Промахов кэша: `{metrics.get('cache_misses', 0)}`\n"
                f"📊 Эффективность кэша: `{metrics.get('hit_rate', '0%')}`\n\n"

                "👤 *Ваша статистика:*\n"
                f"📝 Ваших запросов: `{user_queries}`\n"
                f"🕐 Начало сессии: `{user_session.get('start_time', 'N/A')}`"
            )

            await update.message.reply_text(stats_text, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Ошибка получения метрик: {e}")
            await update.message.reply_text(
                "❌ Не удалось получить статистику системы"
            )

    except Exception as e:
        logger.error(f"Ошибка в обработчике stats_command: {str(e)}", exc_info=True)
        if update.message:
            await update.message.reply_text(
                "❌ Произошла ошибка при получении статистики"
            )


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /clear - очищает историю пользователя."""
    try:
        if not update.effective_user or not update.message:
            return

        user_id = update.effective_user.id

        # Очищаем данные пользователя
        if user_id in _user_responses:
            del _user_responses[user_id]

        if user_id in _user_sessions:
            _user_sessions[user_id]["queries_count"] = 0
            _user_sessions[user_id]["last_activity"] = datetime.now()

        await update.message.reply_text(
            "🧹 История ваших запросов очищена!"
        )

    except Exception as e:
        logger.error(f"Ошибка в обработчике clear_command: {str(e)}", exc_info=True)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик текстовых сообщений с поддержкой асинхронной RAG-системы."""
    global _rag_system

    try:
        if not update.effective_user or not update.message or not update.effective_chat:
            logger.error("Некорректные данные в обработчике handle_message")
            return

        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        query = update.message.text.strip() if update.message.text else ""

        if not query:
            return

        # Обновляем статистику пользователя
        if user_id in _user_sessions:
            _user_sessions[user_id]["queries_count"] += 1
            _user_sessions[user_id]["last_activity"] = datetime.now()

        logger.info(f"Получен запрос: {query[:100]}...", extra={"chat_id": chat_id})

        # Проверяем инициализацию RAG-системы
        if not _rag_system:
            await update.message.reply_text(
                "❌ Система не готова. Попробуйте позже или обратитесь к администратору."
            )
            return

        # Отправка индикатора набора текста
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")

        # Сообщение о начале обработки
        processing_message = await update.message.reply_text("⏳ Анализирую запрос...")

        try:
            # Асинхронная обработка запроса
            response = await _rag_system.process_query(query, include_time=True)

            # Проверяем длину ответа (Telegram ограничивает 4096 символов)
            if len(response) > 4000:
                # Разбиваем длинный ответ на части
                parts = [response[i:i + 4000] for i in range(0, len(response), 4000)]

                # Отправляем первую часть с кнопками
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("👍", callback_data="feedback_good"),
                        InlineKeyboardButton("👎", callback_data="feedback_bad"),
                        InlineKeyboardButton("📚 Источники", callback_data="show_sources")
                    ],
                    [
                        InlineKeyboardButton("📊 Метрики", callback_data="show_metrics")
                    ]
                ])

                sent_message = await update.message.reply_text(
                    parts[0] + "\n\n*(продолжение следует...)*",
                    reply_markup=keyboard,
                    parse_mode="Markdown"
                )

                # Отправляем остальные части
                for part in parts[1:]:
                    await update.message.reply_text(part, parse_mode="Markdown")

            else:
                # Формирование клавиатуры с кнопками
                keyboard = InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("👍", callback_data="feedback_good"),
                        InlineKeyboardButton("👎", callback_data="feedback_bad"),
                        InlineKeyboardButton("📚 Источники", callback_data="show_sources")
                    ],
                    [
                        InlineKeyboardButton("📊 Метрики", callback_data="show_metrics")
                    ]
                ])

                # Отправка полного ответа
                sent_message = await update.message.reply_text(
                    response,
                    reply_markup=keyboard,
                    parse_mode="Markdown"
                )

            # Удаляем сообщение о обработке
            await processing_message.delete()

            # Сохраняем ответ для возможности показа источников
            _user_responses[user_id] = {
                "query": query,
                "response": response,
                "message_id": sent_message.message_id,
                "timestamp": datetime.now()
            }

            logger.info(f"Запрос успешно обработан для пользователя {user_id}", extra={"chat_id": chat_id})

        except Exception as e:
            logger.error(f"Ошибка обработки запроса: {str(e)}", exc_info=True, extra={"chat_id": chat_id})

            # Удаляем сообщение о обработке
            try:
                await processing_message.delete()
            except:
                pass

            error_message = (
                "❌ *Ошибка при обработке запроса*\n\n"
                f"Описание: `{str(e)[:200]}...`\n\n"
                "🔄 Попробуйте:\n"
                "• Переформулировать вопрос\n"
                "• Повторить запрос через некоторое время\n"
                "• Использовать команду /help для справки"
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Критическая ошибка в обработчике сообщений: {str(e)}", exc_info=True)
        try:
            if update.message:
                await update.message.reply_text(
                    "❌ Произошла критическая ошибка. Обратитесь к администратору."
                )
        except:
            pass


async def handle_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатий на кнопки обратной связи."""
    global _rag_system

    try:
        if not update.callback_query or not update.effective_user or not update.effective_chat:
            logger.error("Некорректные данные в обработчике handle_feedback")
            return

        query = update.callback_query
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id

        # Проверка наличия данных пользователя
        if user_id not in _user_responses:
            await query.answer("ℹ️ Информация о вашем запросе не найдена.")
            return

        user_data = _user_responses[user_id]

        # Обработка различных типов callback'ов
        if query.data == "feedback_good":
            await query.answer("✅ Спасибо за положительный отзыв!")
            logger.info(f"Положительный отзыв от пользователя {user_id}", extra={"chat_id": chat_id})

        elif query.data == "feedback_bad":
            await query.answer("📝 Спасибо за отзыв! Мы работаем над улучшением.")
            logger.info(f"Отрицательный отзыв от пользователя {user_id}", extra={"chat_id": chat_id})

        elif query.data == "show_sources":
            await query.answer("📚 Показываю источники информации")

            sources_text = (
                "📚 *Источники информации:*\n\n"
                f"🔍 Запрос: `{user_data.get('query', '')[:100]}...`\n"
                f"🕐 Время обработки: `{user_data.get('timestamp', 'N/A')}`\n\n"
                "📄 *Использованные методы поиска:*\n"
                "• 🔍 Векторный поиск (FAISS)\n"
                "• 📊 BM25 алгоритм\n"
                "• 🌐 Графовая база знаний (Neo4j)\n"
                "• 🔄 Гибридная агрегация (RRF)\n"
                "• 🎯 Переранжировка (CrossEncoder)\n\n"
                "ℹ️ *Примечание:* Детальная информация об источниках "
                "доступна в логах системы."
            )

            await context.bot.send_message(
                chat_id=chat_id,
                text=sources_text,
                reply_to_message_id=user_data.get("message_id"),
                parse_mode="Markdown"
            )

        elif query.data == "show_metrics":
            await query.answer("📊 Получаю метрики...")

            if _rag_system:
                try:
                    metrics = _rag_system.get_metrics()
                    metrics_text = (
                        "📊 *Метрики текущего ответа:*\n\n"
                        f"⚡ Всего запросов в системе: `{metrics.get('total_queries', 0)}`\n"
                        f"✅ Успешность: `{metrics.get('success_rate', '0%')}`\n"
                        f"💾 Попаданий в кэш: `{metrics.get('cache_hits', 0)}`\n"
                        f"📈 Эффективность кэша: `{metrics.get('hit_rate', '0%')}`\n"
                        f"⏱ Среднее время: `{metrics.get('average_response_time', '0s')}`\n\n"
                        f"🕐 Время вашего запроса: `{user_data.get('timestamp', 'N/A')}`"
                    )

                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=metrics_text,
                        reply_to_message_id=user_data.get("message_id"),
                        parse_mode="Markdown"
                    )
                except Exception as e:
                    await context.bot.send_message(
                        chat_id=chat_id,
                        text="❌ Не удалось получить метрики",
                        reply_to_message_id=user_data.get("message_id")
                    )

        logger.info(f"Обработан callback: {query.data} от пользователя {user_id}", extra={"chat_id": chat_id})

    except Exception as e:
        logger.error(f"Ошибка в обработчике обратной связи: {str(e)}", exc_info=True)
        if update.callback_query:
            await update.callback_query.answer("❌ Произошла ошибка при обработке запроса.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Глобальный обработчик ошибок."""
    logger.error(f"Необработанная ошибка: {context.error}", exc_info=context.error)

    # Попытаемся отправить сообщение пользователю об ошибке
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ Произошла неожиданная ошибка. Администратор уведомлен."
            )
        except:
            pass


async def initialize_rag_system() -> bool:
    """Инициализация RAG-системы."""
    global _rag_system, _config

    try:
        logger.info("🔄 Инициализация RAG-системы...")

        # Создаем конфигурацию для RAG-системы
        rag_config = RAGConfig(
            folder_path=_config.get('folder_path', 'docs/'),
            faiss_path=_config.get('faiss_path', 'vector_store/faiss_index'),
            neo4j_uri=_config.get('neo4j_uri', 'bolt://localhost:7687'),
            neo4j_user=_config.get('neo4j_user', 'neo4j'),
            neo4j_password=_config.get('neo4j_password', 'neo4jneo4j'),
            use_semantic_chunking=_config.get('use_semantic_chunking', True)
        )

        # Создаем экземпляр RAG-системы
        _rag_system = AsyncRAGSystem(rag_config)

        # Инициализируем систему
        await _rag_system.initialize()

        logger.info("✅ RAG-система успешно инициализирована")
        return True

    except Exception as e:
        logger.error(f"❌ Ошибка инициализации RAG-системы: {e}", exc_info=True)
        return False


def main_bot() -> int:
    """Основная функция запуска бота."""
    global _config

    try:
        # Загрузка конфигурации
        logger.info("📥 Загрузка конфигурации...")
        _config = load_config()

        # Получение токена бота
        telegram_token = _config.get("telegram_token") or "7811702895:AAFOpxQUXUbkWrx6ksdXEM_LRRaSjnKur8M"

        if not telegram_token or telegram_token == "YOUR_TELEGRAM_BOT_TOKEN":
            logger.error("❌ Токен Telegram-бота не найден в конфигурации")
            return 1

        # Создание приложения бота
        logger.info("🤖 Создание Telegram-бота...")
        application = Application.builder().token(telegram_token).build()

        # Регистрация обработчиков команд
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("stats", stats_command))
        application.add_handler(CommandHandler("clear", clear_command))

        # Регистрация обработчиков сообщений и callback'ов
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
        application.add_handler(CallbackQueryHandler(handle_feedback))

        # Регистрация глобального обработчика ошибок
        application.add_error_handler(error_handler)

        # Асинхронная инициализация RAG-системы
        async def post_init(application):
            success = await initialize_rag_system()
            if not success:
                logger.error("❌ Не удалось инициализировать RAG-систему")
                raise RuntimeError("RAG система не инициализирована")

        # Добавляем пост-инициализацию
        application.post_init = post_init

        # Запуск бота
        logger.info("🚀 Запуск Telegram-бота...")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )

        return 0

    except Exception as e:
        logger.critical(f"💥 Критическая ошибка при запуске бота: {e}", exc_info=True)
        return 1

    finally:
        # Корректное закрытие RAG-системы
        async def cleanup():
            global _rag_system
            if _rag_system:
                logger.info("🔄 Закрытие RAG-системы...")
                await _rag_system.close()
                logger.info("✅ RAG-система закрыта")

        try:
            asyncio.run(cleanup())
        except Exception as e:
            logger.error(f"Ошибка при закрытии системы: {e}")


if __name__ == "__main__":
    exit_code = main_bot()
    exit(exit_code)
