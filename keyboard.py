import os
import cv2
import numpy as np
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, MessageHandler, Filters, CallbackContext, CallbackQueryHandler

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Store pending photos in a dictionary {chat_id: file_path}
pending_photos = {}

def process_image(photo_path, output_path, pixelation_factor):
    try:
        image = cv2.imread(photo_path)
        if image is None:
            print(f"Failed to read image: {photo_path}")
            return False
        
        h, w = image.shape[:2]
        small = cv2.resize(image, (max(1, int(w * pixelation_factor)), max(1, int(h * pixelation_factor))), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        success = cv2.imwrite(output_path, pixelated)
        print(f"Image write success: {success}")
        return success
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return False

def build_keyboard():
    keyboard = [
        [InlineKeyboardButton("very fine", callback_data="pixelate_0.2")],
        [InlineKeyboardButton("fine", callback_data="pixelate_0.15")],
        [InlineKeyboardButton("rough", callback_data="pixelate_0.09")],
        [InlineKeyboardButton("very rough", callback_data="pixelate_0.08")],
        [InlineKeyboardButton("distorted", callback_data="pixelate_0.06")]
    ]
    return InlineKeyboardMarkup(keyboard)

def handle_photo(update: Update, context: CallbackContext) -> None:
    if update.effective_chat.type != "private":
        print("Ignoring non-private chat")
        return
    
    chat_id = update.effective_chat.id
    print(f"Received photo from chat {chat_id}")
    
    photo = update.message.photo[-1]
    file = context.bot.get_file(photo.file_id)
    
    input_path = f"photo_{chat_id}.jpg"
    file.download(input_path)
    print(f"Photo downloaded to {input_path}")
    
    # Store the photo path
    pending_photos[chat_id] = input_path
    
    # Send the keyboard
    reply_markup = build_keyboard()
    update.message.reply_text("Choose pixelation level:", reply_markup=reply_markup)
    print("Keyboard sent")

def handle_button(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    chat_id = query.message.chat_id
    print(f"Button pressed in chat {chat_id}, data: {query.data}")
    
    query.answer()  # Acknowledge the button press
    
    if chat_id not in pending_photos:
        print(f"No pending photo for chat {chat_id}")
        query.edit_message_text("No photo to process. Please send a new photo.")
        return
    
    # Extract pixelation factor from callback data
    try:
        pixelation_factor = float(query.data.split('_')[1])
        print(f"Parsed pixelation factor: {pixelation_factor}")
    except (IndexError, ValueError) as e:
        print(f"Error parsing callback data: {str(e)}")
        query.edit_message_text("Invalid pixelation factor.")
        return
    
    input_path = pending_photos[chat_id]
    output_path = f"pixelated_{chat_id}.jpg"
    
    if process_image(input_path, output_path, pixelation_factor):
        print(f"Image processed, sending from {output_path}")
        try:
            with open(output_path, 'rb') as f:
                context.bot.send_photo(chat_id=chat_id, photo=f)
            print("Photo sent successfully")
            
            # Clean up
            if os.path.exists(input_path):
                os.remove(input_path)
                print(f"Removed input file: {input_path}")
            if os.path.exists(output_path):
                os.remove(output_path)
                print(f"Removed output file: {output_path}")
            del pending_photos[chat_id]
            print(f"Removed chat {chat_id} from pending_photos")
            
            query.edit_message_text("Image processed!")
        except Exception as e:
            print(f"Error sending photo: {str(e)}")
            query.edit_message_text(f"Failed to send processed image: {str(e)}")
    else:
        print("Image processing failed")
        query.edit_message_text("Failed to process image.")
    
    # Clean up in case of failure
    if chat_id in pending_photos:
        if os.path.exists(input_path):
            os.remove(input_path)
            print(f"Cleaned up input file: {input_path}")
        del pending_photos[chat_id]
        print(f"Removed chat {chat_id} from pending_photos after failure")

def main():
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env")
        return
    
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    
    dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
    dispatcher.add_handler(CallbackQueryHandler(handle_button, pattern='^pixelate_'))
    
    print("Bot starting...")
    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
