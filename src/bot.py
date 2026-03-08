"""
Telegram bot front-end for the MCP + Ollama pipeline.

Each user message is routed through the MCP client (client.py),
which queries Ollama with web-search tool support via DuckDuckGo.

Setup:
  export TELEGRAM_BOT_TOKEN="<your BotFather token>"
  uv run python src/bot.py

The bot responds to any text message with the LLM answer.
"""
import os
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, ContextTypes

from client import run

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable is not set.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.message.text.strip()
    logger.info("Received message from %s: %r", update.effective_user.username, query)

    # Show typing indicator while we process
    await update.message.chat.send_action("typing")

    try:
        answer = await run(query)
    except Exception as exc:
        logger.exception("Error processing query")
        answer = f"Sorry, something went wrong: {exc}"

    await update.message.reply_text(answer)


def main() -> None:
    app = Application.builder().token(TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    logger.info("Bot started. Polling for updates...")
    app.run_polling()


if __name__ == "__main__":
    main()
