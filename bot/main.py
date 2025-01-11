from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import os
from bot.model_loader import loader_model, predict_text
from config import TELEGRAM_BOT_TOKEN

# Загрузка модели и токенизатора
model, tokenizer = loader_model()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Отправьте текст, чтобы проверить, написан ли он искусственным интеллектом."
    )


async def analyze_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text

    if len(user_text) < 10:
        await update.message.reply_text(
            "Текст слишком короткий. Пожалуйста, отправьте текст длиной хотя бы 10 символов.")
        return

    probability = predict_text(user_text, model, tokenizer)
    if probability > 50:
        response = f"Текст сгенерирован ИИ"
    else:
        response = f"Текст написан человеком"
    await update.message.reply_text(response)


def main():
    # Загрузка токена из переменной окружения
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, analyze_text))

    print("Бот запущен. Ожидание сообщений...")
    app.run_polling()


if __name__ == "__main__":
    main()
