import app
import app.routes.api

socketio = app.socketio
flask_app = app.app
if __name__ == '__main__':
    socketio.run(flask_app, host='0.0.0.0', port=5000)