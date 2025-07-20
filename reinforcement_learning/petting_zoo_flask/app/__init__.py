from flask import Flask
from flask_socketio import SocketIO
from app.config.env_config import FLASK_KEY
from app.config.logging_config import setup_logger
import threading

app = Flask(__name__)
logger = setup_logger()
app.logger = logger
app.secret_key = FLASK_KEY

socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
client_sessions_lock = threading.Lock()
client_sessions = {}
