import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os




# Load environment variables from .env
load_dotenv()

# Set up logging
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load LLM
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Function to process user input with LLM
def process_with_llm(user_message):
    try:
        # Prepare the input for the model
        formatted_input = f"User: {user_message}\nAssistant:"
        inputs = tokenizer(formatted_input, return_tensors="pt")
        
        # Generate a response
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response text
        response_text = response.split("Assistant:")[-1].strip()
        return response_text
    except Exception as e:
        logger.error(f"Error in LLM processing: {e}")
        return "I'm sorry, I couldn't process your request."

# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the bot is started."""
    await update.message.reply_text("Hi! I'm WiseWhisper, your intelligent assistant. Ask me anything!")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming messages and respond using the LLM."""
    user_message = update.message.text  # Get the user's input message
    logger.info(f"Received message: {user_message}")

    # Process the message with the LLM
    response = process_with_llm(user_message)

    # Send the response back to the user
    await update.message.reply_text(response)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log errors caused by updates."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

# Main function to start the bot
def main():
    """Start the bot."""
    # bot token from Telegram loaded from the .env file
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

    # Ensure the token is set
    if not TELEGRAM_TOKEN:
        raise ValueError("TELEGRAM_TOKEN is not set in the environment variables.")

    # Create the application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # command and message handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # error handler
    application.add_error_handler(error_handler)

    # Run the bot
    logger.info("Bot is starting...")
    application.run_polling()

if __name__ == "__main__":
    main()
