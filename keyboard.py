import os
import cv2
import numpy as np
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DEFAULT_PIXELATION_FACTOR = float(os.getenv('PIXELATION_FACTOR', 0.1))  # Default value if not in .env

# Store pending photos in a dictionary {chat_id: file_path}
pending_photos = {}

def process_image(photo_path, output_path, pixelation_factor):
    try:
        image = cv2.imread(photo_path)
        if image is None:
            logger.error(f"Failed to read image: {photo_path}")
            return False
        
        h, w = image.shape[:2]
        small = cv2.resize(image, (max(1, int(w * pixelation_factor)), max(1, int(h * pixelation_factor))), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(output_path, pixelated)
        return True
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False

def build_keyboard():
    keyboard = [
        [InlineKeyboardButton("Yvettes Setting", callback_data=f"pixelate_{DEFAULT_PIXELATION_FACTOR}")],
        [
            InlineKeyboardButton("Fine", callback_data="pixelate_0.3"),
            InlineKeyboardButton("Finer", callback_data="pixelate_0.5"),
            InlineKeyboardButton("Finest", callback_data="pixelate_0.7")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def handle_photo(update: Update, context: CallbackContext) -> None:
    if update.effective_chat.type != "private":
        return
    
    photo = update.message.photo[-1]
    file = context.bot.get_file(photo.file_id)
    
    input_path = f"photo_{update.effective_chat.id}.jpg"
    file.download(input_path)
    
    # Store the photo path
    pending_photos[update.effective_chat.id] = input_path
    
    # Send the keyboard
    reply_markup = build_keyboard()
    update.message.reply_text("Choose pixelation level:", reply_markup=reply_markup)

def handle_button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    chat_id = query.message.chat_id
    
    if chat_id not in pending_photos:
        query.edit_message_text("No photo to process. Please send a new photo.")
        return
    
    # Extract pixelation factor from callback data
    pixelation_factor = float(query.data.split('_')[1])
    
    input_path = pending_photos[chat_id]
    output_path = f"pixelated_{chat_id}.jpg"
    
    if process_image(input_path, output_path, pixelation_factor):
        with open(output_path, 'rb') as f:
            context.bot.send_photo(chat_id=chat_id, photo=f)
        
        # Clean up
        os.remove(input_path)
        os.remove(output_path)
        del pending_photos[chat_id]
        
        # Remove the keyboard
        query.edit_message_text("Image processed!")
    else:
        query.edit_message_text("Failed to process image.")
        if os.path.exists(input_path):
            os.remove(input_path)
        del pending_photos[chat_id]

def main():
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
    dispatcher.add_handler(CallbackQueryHandler(handle_button, pattern='^pixelate_'))
    
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
