from flask import Flask
from flask_socketio import SocketIO
import threading

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
global_lock = threading.Lock()
client_sessions = {}