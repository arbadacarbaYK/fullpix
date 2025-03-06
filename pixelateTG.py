import os
import cv2
import numpy as np
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import Updater, MessageHandler, CommandHandler, CallbackContext
from telegram.ext.filters import Filters

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename='bot_log.txt'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Pixelation levels
PIXELATION_LEVELS = {
    'very_fine': 0.2,
    'fine': 0.15,
    'rough': 0.09,
    'very_rough': 0.08,
    'distorted': 0.06
}

def log_event(message: str):
    """Log events with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"{timestamp}: {message}")

def process_image(photo_path: str, output_path: str, pixelation_factor: float) -> bool:
    try:
        log_event(f"Processing image with pixelation factor: {pixelation_factor}")
        image = cv2.imread(photo_path)
        if image is None:
            log_event("Failed to read image")
            return False
        
        h, w = image.shape[:2]
        log_event(f"Image dimensions: {w}x{h}")
        
        small = cv2.resize(image, (int(w * pixelation_factor), int(h * pixelation_factor)), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        cv2.imwrite(output_path, pixelated)
        log_event("Image processed successfully")
        return True
    except Exception as e:
        log_event(f"Error processing image: {str(e)}")
        return False

def start(update: Update, context: CallbackContext) -> None:
    """Send welcome message and instructions"""
    if update.effective_chat.type != "private":
        return
    
    welcome_text = (
        "Welcome to the Pixelation Bot! 🎨\n\n"
        "Send me any photo and I'll offer you different pixelation levels:\n"
        "• Very Fine (0.2)\n"
        "• Fine (0.15)\n"
        "• Rough (0.09)\n"
        "• Very Rough (0.08)\n"
        "• Distorted (0.06)\n\n"
        "Just send a photo to begin!"
    )
    update.message.reply_text(welcome_text)
    log_event(f"Start command received from user {update.effective_user.id}")

def handle_photo(update: Update, context: CallbackContext) -> None:
    """Handle incoming photos"""
    if update.effective_chat.type != "private":
        log_event(f"Ignored message from non-private chat: {update.effective_chat.id}")
        return

    user_id = update.effective_user.id
    log_event(f"Received photo from user {user_id}")
    
    photo = update.message.photo[-1]
    file = context.bot.get_file(photo.file_id)
    
    # Create user-specific directory if it doesn't exist
    user_dir = f"user_{user_id}"
    os.makedirs(user_dir, exist_ok=True)
    
    input_path = f"{user_dir}/original.jpg"
    file.download(input_path)
    log_event(f"Downloaded photo to {input_path}")
    
    keyboard = [[level.replace('_', ' ').title()] for level in PIXELATION_LEVELS.keys()]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True)
    
    context.user_data['photo_path'] = input_path
    update.message.reply_text(
        "Choose your pixelation level:",
        reply_markup=reply_markup
    )
    log_event(f"Sent pixelation options to user {user_id}")

def handle_text(update: Update, context: CallbackContext) -> None:
    """Handle text responses for pixelation level selection"""
    if update.effective_chat.type != "private":
        return
    
    user_id = update.effective_user.id
    text = update.message.text.lower().replace(' ', '_')
    
    if text not in PIXELATION_LEVELS:
        log_event(f"Invalid pixelation level received from user {user_id}: {text}")
        update.message.reply_text("Please select a valid pixelation level from the keyboard.")
        return
    
    if 'photo_path' not in context.user_data:
        log_event(f"No photo found for user {user_id}")
        update.message.reply_text("Please send a photo first.")
        return
    
    input_path = context.user_data['photo_path']
    output_path = f"user_{user_id}/pixelated_{text}.jpg"
    pixelation_factor = PIXELATION_LEVELS[text]
    
    log_event(f"Processing image for user {user_id} with level {text}")
    if process_image(input_path, output_path, pixelation_factor):
        with open(output_path, 'rb') as f:
            update.message.reply_photo(photo=f)
            log_event(f"Sent pixelated image to user {user_id}")
    else:
        update.message.reply_text("Sorry, failed to process the image. Please try again.")
        log_event(f"Failed to process image for user {user_id}")

def main():
    """Start the bot"""
    try:
        log_event("Bot starting up")
        updater = Updater(TOKEN, use_context=True)
        dispatcher = updater.dispatcher
        
        # Add handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(MessageHandler(Filters.photo & Filters.private, handle_photo))
        dispatcher.add_handler(MessageHandler(Filters.text & Filters.private, handle_text))
        
        updater.start_polling()
        log_event("Bot started successfully")
        updater.idle()
    except Exception as e:
        log_event(f"Critical error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
