import app
from app.config.env_config import FLASK_RUN_PORT
import app.config.oauth2_config
import app.routes.api

if __name__ == '__main__':
    app.socketio.run(app.app, host='0.0.0.0', port=FLASK_RUN_PORT)