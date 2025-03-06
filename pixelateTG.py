import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
from dotenv import load_dotenv
import cv2
import random
import imageio
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Updater, CallbackContext, CommandHandler, CallbackQueryHandler, MessageHandler, Filters
from concurrent.futures import ThreadPoolExecutor, wait
from uuid import uuid4
import time
import logging
import traceback
import socket
import urllib3
from telegram.utils.request import Request
import psutil
import glob
import logging.handlers
from gif_processor import process_telegram_gif
from constants import PIXELATION_FACTOR
from gif_processor import GifProcessor

# Configure DNS settings
socket.setdefaulttimeout(20)
urllib3.disable_warnings()

# Configure logging
logging.basicConfig(
    filename='pixelbot_debug.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logging.getLogger('telegram').setLevel(logging.INFO)
logging.getLogger('telegram.ext.dispatcher').setLevel(logging.INFO)
logging.getLogger('telegram.bot').setLevel(logging.INFO)

load_dotenv()

TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
MAX_THREADS = 15
RESIZE_FACTOR = 2.0
executor = ThreadPoolExecutor(max_workers=MAX_THREADS)

# Dictionary to store active sessions
active_sessions = {}

def verify_permissions():
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Verifying permissions for directories...")
    for directory in ['processed', 'downloads']:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Test write permissions
            test_file = os.path.join(directory, 'test.txt')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            
            logger.info(f"Verified write permissions for {directory}")
            
        except Exception as e:
            logger.error(f"Permission error for directory {directory}: {str(e)}")
            return False
    return True

def get_file_path(directory, id_prefix, session_id, suffix):
    """Generate a file path with the given parameters"""
    # Make sure we don't add .jpg to GIF files
    if suffix.endswith('.gif.jpg'):
        suffix = suffix.replace('.gif.jpg', '.gif')
    elif suffix.endswith('.gif'):
        # Keep as is
        pass
    elif not suffix.endswith('.jpg'):
        suffix = f"{suffix}.jpg"
        
    return os.path.join(directory, f"{id_prefix}_{session_id}_{suffix}")

def cleanup_temp_files():
    """Clean up temporary files in downloads and processed directories"""
    for directory in ['downloads', 'processed']:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                os.remove(os.path.join(directory, f))
            logger.info(f"Cleaned up {directory} directory")

def start(update: Update, context: CallbackContext) -> None:
    """Handle /start command"""
    update.message.reply_text("Send me a photo to get started!")

def get_id_prefix(update):
    """Generate a consistent ID prefix for a user"""
    return f"user_{update.effective_user.id}"

def process_image(photo_path, output_path):
    try:
        image = cv2.imread(photo_path)
        if image is None:
            logger.error(f"Failed to read image: {photo_path}")
            return False
        
        # Resize down and back up to pixelate
        h, w = image.shape[:2]
        small = cv2.resize(image, (int(w * PIXELATION_FACTOR), int(h * PIXELATION_FACTOR)), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(output_path, pixelated)
        return True

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False
           
def handle_message(update: Update, context: CallbackContext, photo=None) -> None:
    try:
        logger.debug("Function called")
        message = photo if photo else update.message
        chat_id = message.chat_id
        
        session_id = str(uuid4())
        id_prefix = f"user_{chat_id}"

        session_data = {
            'chat_id': chat_id,
            'id_prefix': id_prefix,
            'session_id': session_id,
            'is_gif': False
        }

        # Check if this is a GIF
        is_gif = False
        file_id = None
        
        # Handle photos
        if message.photo:
            file_id = message.photo[-1].file_id
        # Handle GIFs as documents
        elif message.document and message.document.mime_type == 'image/gif':
            file_id = message.document.file_id
            is_gif = True
        # Handle GIFs as animations
        elif message.animation:
            file_id = message.animation.file_id
            is_gif = True
            
        if not file_id:
            return
            
        # Download the file
        file_extension = 'gif.jpg' if is_gif else 'original.jpg'
        file_path = get_file_path('downloads', id_prefix, session_id, file_extension)
        
        file = context.bot.get_file(file_id)
        file.download(file_path)
        logger.info(f"Downloaded {'GIF' if is_gif else 'photo'} to {file_path}")
        
        # Store session data
        session_data['input_path'] = file_path
        session_data['is_gif'] = is_gif
        context.user_data[session_id] = session_data
        logger.debug(f"Created new session: {session_id}")
        
        # Create keyboard with effect options
        keyboard = [
            [
                InlineKeyboardButton("🧩 Pixelate", callback_data=f"pixelate:{session_id}")
            ],
            [
                InlineKeyboardButton("🤡 Clown", callback_data=f"clown:{session_id}"),
                InlineKeyboardButton("😎 Liotta", callback_data=f"liotta:{session_id}"),
                InlineKeyboardButton("💀 Skull", callback_data=f"skull:{session_id}")
            ],
            [
                InlineKeyboardButton("🐱 Cat", callback_data=f"cat:{session_id}"),
                InlineKeyboardButton("🐸 Pepe", callback_data=f"pepe:{session_id}"),
                InlineKeyboardButton("👨 Chad", callback_data=f"chad:{session_id}")
            ],
            [
                InlineKeyboardButton("❌ Close", callback_data=f"close:{session_id}")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send the keyboard
        message.reply_text('Choose an effect:', reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {str(e)}")
        logger.error(traceback.format_exc())

def cleanup_before_start(bot):
    """Clean up before starting the bot"""
    logger.info("Cleanup before start completed")
    return True

def error_handler(update: Update, context: CallbackContext) -> None:
    """Log Errors caused by Updates."""
    logger.error(f'Update "{update}" caused error "{context.error}"')

def button_callback(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    
    try:
        # Parse the callback data
        if ':' in query.data:
            action, session_id = query.data.split(':')
        else:
            # Handle the old format for backward compatibility
            for effect in ['pixelate', 'clown', 'liotta', 'skull', 'cat', 'pepe', 'chad', 'close']:
                if query.data.startswith(f"{effect}_"):
                    action = effect
                    session_id = query.data[len(effect)+1:]
                    break
            else:
                query.answer("Invalid action")
                return
        
        # Handle close action separately
        if action == 'close':
            # Simply delete the message with the keyboard
            query.delete_message()
            return
            
        # Get session data
        session_data = context.user_data.get(session_id)
        if not session_data:
            logger.error(f"No session data found for {action}")
            query.answer("Session expired, please try again")
            return
        
        # Process the image with the selected effect
        chat_id = session_data.get('chat_id')
        input_path = session_data.get('input_path')
        is_gif = session_data.get('is_gif', False)
        id_prefix = session_data.get('id_prefix')
        
        if not input_path or not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            query.answer("Original file not found, please send a new one")
            return
            
        # Show processing message
        query.answer(f"Processing with {action} effect...")
        query.edit_message_text(f"Processing with {action} effect...")
        
        else:
            # Process photo
            output_path = get_file_path('processed', id_prefix, session_id, action)
            if process_image(input_path, output_path, action):
                # Send the processed photo without caption
                with open(output_path, 'rb') as f:
                    context.bot.send_photo(
                        chat_id=chat_id,
                        photo=f
                    )
                # Clean up
                os.remove(output_path)
            else:
                context.bot.send_message(
                    chat_id=chat_id,
                    text="Failed to process image. Please try again."
                )
                
def get_last_update_id() -> int:
    try:
        with open('pixelbot_last_update.txt', 'r') as f:
            return int(f.read().strip())
    except:
        return 0

def save_last_update_id(update_id: int) -> None:
    with open('pixelbot_last_update.txt', 'w') as f:
        f.write(str(update_id))

def cleanup_old_files():
    """Cleanup files older than 24 hours"""
    current_time = time.time()
    for directory in ['processed', 'downloads']:
        if os.path.exists(directory):
            for f in os.listdir(directory):
                filepath = os.path.join(directory, f)
                if os.path.getmtime(filepath) < (current_time - 86400):  # 24 hours
                    try:
                        os.remove(filepath)
                        logger.debug(f"Removed old file: {filepath}")
                    except Exception as e:
                        logger.error(f"Failed to remove {filepath}: {e}")

def photo_command(update: Update, context: CallbackContext) -> None:
    # Check if this is a reply to a photo
    if not update.message.reply_to_message or not update.message.reply_to_message.photo:
        return
    # Process the replied-to photo
    handle_message(update, context, photo=update.message.reply_to_message)

def handle_photo(update: Update, context: CallbackContext) -> None:
    """Handle photos sent to the bot"""
    # Check if this is a direct message or a group
    chat_type = update.effective_chat.type
    
    # In groups, only respond to explicit /pixel commands as replies
    if chat_type in ['group', 'supergroup']:
        return
    
    # In direct messages, process automatically
    process_media(update, context)

def handle_pixel_command(update: Update, context: CallbackContext) -> None:
    """Handle /pixel command - must be a reply to a photo/GIF in groups"""
    try:
        # Check if this is a reply to a message
        if not update.message.reply_to_message:
            update.message.reply_text("Please use this command as a reply to a photo or GIF.")
            return
        
        # Check if the replied message contains a photo or document (GIF)
        replied_msg = update.message.reply_to_message
        
        # Check for photos
        has_photo = bool(replied_msg.photo)
        
        # Check for GIFs - they can be in document or animation field
        has_gif = (replied_msg.document and 
                  (replied_msg.document.mime_type == 'image/gif' or 
                   replied_msg.document.file_name.lower().endswith('.gif'))) or bool(replied_msg.animation)
        
        if not (has_photo or has_gif):
            # Send message without reply_to_message_id to avoid errors
            update.message.chat.send_message("Please reply to a photo or GIF.")
            return
        
        # Process the media from the replied message
        process_media(update, context, replied_msg)
        
    except Exception as e:
        logger.error(f"Error in handle_pixel_command: {str(e)}")
        logger.error(traceback.format_exc())
        # Send error message without reply_to_message_id
        try:
            update.message.chat.send_message("An error occurred while processing your request.")
        except:
            pass

def process_media(update: Update, context: CallbackContext, replied_msg=None) -> None:
    """Process media (photos or GIFs) and show the effect keyboard"""
    try:
        # Use either the replied message or the current message
        message = replied_msg if replied_msg else update.message
        
        # Handle the message with our existing function
        handle_message(update, context, photo=message)
        
    except Exception as e:
        logger.error(f"Error in process_media: {str(e)}")
        logger.error(traceback.format_exc())

def main() -> None:
    try:
        # Kill any existing instances
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['pid'] != current_pid:
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline and 'pixelateTG.py' in cmdline[0]:
                        proc.kill()
                        logger.info(f"Killed existing bot instance with PID {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

        # Get environment from ENV var, default to 'development'
        env = os.getenv('BOT_ENV', 'development')
        logger.info(f"Starting bot in {env} environment")
        
        # Initialize as before
        if not verify_permissions():
            logger.error("Failed to verify directory permissions")
            return
            
        for directory in ['processed', 'downloads']:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
                
        cleanup_temp_files()
        socket.setdefaulttimeout(20)
        urllib3.disable_warnings()
        
        # Use token from environment
        if not TOKEN:
            logger.error("No TELEGRAM_BOT_TOKEN found in environment")
            return
        token = TOKEN
        
        # Initialize the updater with proper timeouts
        updater = Updater(
            token=token,
            use_context=True,
            request_kwargs={
                'connect_timeout': 20,
                'read_timeout': 20
            }
        )
        
        cleanup_before_start(updater.bot)
        
        # Register handlers
        logger.info("Registering handlers...")
        dispatcher = updater.dispatcher
        
        # Register error handler
        dispatcher.add_error_handler(error_handler)
        
        # Command handlers - register pixel command with explicit filters for ALL chat types
        dispatcher.add_handler(CommandHandler(
            "pixel", 
            handle_pixel_command,
            filters=Filters.command & (Filters.chat_type.groups | Filters.chat_type.private)
        ))
        
        # Other command handlers
        dispatcher.add_handler(CommandHandler("start", start))
        dispatcher.add_handler(CommandHandler("help", help_command))
        
        # Media handlers - photos and GIFs
        dispatcher.add_handler(MessageHandler(Filters.photo, handle_photo))
        dispatcher.add_handler(MessageHandler(Filters.document.category("image/gif") | 
                                            Filters.animation, handle_photo))
        
        # Button callback handler
        dispatcher.add_handler(CallbackQueryHandler(button_callback))
        
        # Start the Bot with clean state
        logger.info(f"Starting bot in {env} mode...")
        updater.start_polling(drop_pending_updates=True)
        
        # Run the bot until the user presses Ctrl-C
        updater.idle()
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
