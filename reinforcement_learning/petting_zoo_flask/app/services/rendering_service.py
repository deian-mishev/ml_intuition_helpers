from app.config.session_state import SessionState
import base64
import cv2
import numpy as np

def render_frame(session: SessionState) -> str:
    frame = session.env.render()
    _, buffer = cv2.imencode('.png', frame)
    return base64.b64encode(buffer).decode('utf-8')