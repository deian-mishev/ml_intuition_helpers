from flask import Flask
from flask_socketio import SocketIO
from app.config.env_config import FLASK_KEY
import threading

app = Flask(__name__)
app.secret_key = FLASK_KEY

socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
global_lock = threading.Lock()
client_sessions_lock = threading.Lock()
client_sessions = {}
