from functools import wraps
from flask import Flask, jsonify, redirect, request, session, url_for
from flask_socketio import SocketIO
from authlib.integrations.flask_client import OAuth
import threading
from urllib.parse import quote
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")

KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")
KEYCLOAK_CLIENT_SECRET = os.getenv("KEYCLOAK_CLIENT_SECRET")
KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL")
KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM")

oauth = OAuth(app)
oauth.register(
    name='keycloak',
    client_id=KEYCLOAK_CLIENT_ID,
    client_secret=KEYCLOAK_CLIENT_SECRET,
    server_url=KEYCLOAK_SERVER_URL,
    server_metadata_url=f'{KEYCLOAK_SERVER_URL}realms/{KEYCLOAK_REALM}/.well-known/openid-configuration',
    access_token_url=f'{KEYCLOAK_SERVER_URL}realms/{KEYCLOAK_REALM}/protocol/openid-connect/token',
    authorize_url=f'{KEYCLOAK_SERVER_URL}realms/{KEYCLOAK_REALM}/protocol/openid-connect/auth',
    userinfo_endpoint=f'{KEYCLOAK_SERVER_URL}realms/{KEYCLOAK_REALM}/protocol/openid-connect/userinfo',
    client_kwargs={'scope': 'openid profile email'},
)


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


def roles_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user_roles = session.get('roles', [])
            if not any(role in user_roles for role in roles):
                return jsonify({"error": "Forbidden"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


@app.route('/auth')
def auth():
    token = oauth.keycloak.authorize_access_token()
    userinfo = oauth.keycloak.userinfo(token=token)

    roles = []
    if 'realm_access' in userinfo:
        roles.extend(userinfo['realm_access'].get('roles', []))
    if 'resource_access' in userinfo:
        for client_roles in userinfo['resource_access'].values():
            roles.extend(client_roles.get('roles', []))
    session['id_token'] = token.get('id_token')
    session['user'] = userinfo
    session['roles'] = roles

    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    id_token = session.get('id_token')
    session.clear()

    redirect_uri = url_for('index', _external=True)
    logout_url = (
        f"{KEYCLOAK_SERVER_URL.rstrip('/')}/realms/{KEYCLOAK_REALM}/protocol/openid-connect/logout"
        f"?post_logout_redirect_uri={quote(redirect_uri, safe='')}"
    )
    if id_token:
        logout_url += f"&id_token_hint={quote(id_token)}"
    return redirect(logout_url)


@app.route('/login')
def login():
    return oauth.keycloak.authorize_redirect(redirect_uri=(url_for('auth', _external=True)))


socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")
global_lock = threading.Lock()
client_sessions_lock = threading.Lock()
client_sessions = {}
