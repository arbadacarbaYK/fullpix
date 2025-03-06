import os
import cv2
import numpy as np
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
PIXELATION_FACTOR = 0.1  # Finer pixelation for artistic effect

def process_image(photo_path, output_path):
    try:
        image = cv2.imread(photo_path)
        if image is None:
            logger.error(f"Failed to read image: {photo_path}")
            return False
        
        # Apply pixelation to the full image
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * PIXELATION_FACTOR), int(h * PIXELATION_FACTOR)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(output_path, pixelated)
        return True
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False

def handle_photo(update: Update, context: CallbackContext) -> None:
    if update.effective_chat.type != "private":
        return  # Ignore groups
    
    photo = update.message.photo[-1]
    file = context.bot.get_file(photo.file_id)
    
    input_path = "photo.jpg"
    output_path = "pixelated.jpg"
    file.download(input_path)
    
    if process_image(input_path, output_path):
        with open(output_path, 'rb') as f:
            update.message.reply_photo(photo=f)
    else:
        update.message.reply_text("Failed to process image.")

def main():
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
    
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
