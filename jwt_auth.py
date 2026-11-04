"""
=============================================================================
AI-Enhanced Learning Assistant - JWT (JSON Web Token) Authentication Engine
=============================================================================
Pure-Python, zero-dependency, RFC 7519 compliant JWT generator & validator.
Implements HMAC-SHA256 (HS256) signature verification, base64url encoding,
expiration enforcement, and Flask authentication decorators.
=============================================================================
"""

import os
import json
import hmac
import hashlib
import base64
import time
from functools import wraps
from flask import request, redirect, url_for, session, g, jsonify
from dotenv import load_dotenv

# Load JWT secret configuration from .env file
_env_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(_env_path):
    load_dotenv(dotenv_path=_env_path, override=True)
else:
    load_dotenv(override=True)

JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'ai_learning_assistant_jwt_super_secret_2026_key_#889!')
JWT_ALGORITHM = 'HS256'
DEFAULT_EXPIRATION_SECONDS = 86400  # 24 Hours


def _base64url_encode(data: bytes) -> str:
    """Encodes bytes to base64url string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')


def _base64url_decode(data_str: str) -> bytes:
    """Decodes base64url string with proper padding."""
    padding_needed = 4 - (len(data_str) % 4)
    if padding_needed and padding_needed != 4:
        data_str += '=' * padding_needed
    return base64.urlsafe_b64decode(data_str.encode('utf-8'))


def generate_jwt_token(user_id, username, email=None, extra_claims=None, expires_in=DEFAULT_EXPIRATION_SECONDS):
    """
    Generates a cryptographically signed JWT token (HS256).
    """
    now = int(time.time())
    header = {
        "alg": JWT_ALGORITHM,
        "typ": "JWT"
    }

    payload = {
        "sub": str(user_id),
        "user_id": user_id,
        "username": username,
        "email": email or "",
        "iat": now,
        "exp": now + expires_in
    }

    if extra_claims and isinstance(extra_claims, dict):
        payload.update(extra_claims)

    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')

    encoded_header = _base64url_encode(header_bytes)
    encoded_payload = _base64url_encode(payload_bytes)

    signing_input = f"{encoded_header}.{encoded_payload}".encode('utf-8')
    signature = hmac.new(JWT_SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    encoded_signature = _base64url_encode(signature)

    return f"{encoded_header}.{encoded_payload}.{encoded_signature}"


def decode_jwt_token(token: str):
    """
    Decodes and validates a JWT token.
    Returns (payload_dict, None) on success or (None, error_message) on failure.
    """
    if not token or not isinstance(token, str):
        return None, "Token missing or invalid format."

    parts = token.strip().split('.')
    if len(parts) != 3:
        return None, "Malformed JWT token structure."

    encoded_header, encoded_payload, encoded_signature = parts

    try:
        signing_input = f"{encoded_header}.{encoded_payload}".encode('utf-8')
        expected_sig = hmac.new(JWT_SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = _base64url_decode(encoded_signature)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None, "Invalid token cryptographic signature."

        payload_bytes = _base64url_decode(encoded_payload)
        payload = json.loads(payload_bytes.decode('utf-8'))

        # Check expiration
        now = int(time.time())
        if 'exp' in payload and now > payload['exp']:
            return None, "Token has expired."

        return payload, None

    except Exception as e:
        return None, f"Token decode exception: {str(e)}"


def get_token_from_request():
    """
    Retrieves JWT token from Authorization Header, Cookies, or Form/Session.
    """
    # 1. Check HTTP Authorization Header: Bearer <token>
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header[7:].strip()

    # 2. Check HTTP-only Cookie
    cookie_token = request.cookies.get('jwt_token')
    if cookie_token:
        return cookie_token

    # 3. Check Session
    if session.get('jwt_token'):
        return session.get('jwt_token')

    return None


def jwt_required(f):
    """
    Flask route decorator ensuring valid JWT authentication before execution.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = get_token_from_request()
        if not token:
            if request.is_json:
                return jsonify({"error": "Unauthorized: JWT token required."}), 401
            return redirect(url_for('login', next=request.path))

        payload, error = decode_jwt_token(token)
        if error or not payload:
            session.clear()
            if request.is_json:
                return jsonify({"error": f"Unauthorized: {error}"}), 401
            return redirect(url_for('login', error="Session expired. Please login again."))

        # Attach authenticated user to Flask global context and session
        g.current_user = payload
        session['loggedin'] = True
        session['user_id'] = payload.get('user_id')
        session['username'] = payload.get('username')
        session['email'] = payload.get('email')

        return f(*args, **kwargs)
    return decorated_function


def set_jwt_auth_cookie(response, token):
    """Sets HTTP-only JWT cookie on response object."""
    response.set_cookie(
        'jwt_token',
        token,
        max_age=DEFAULT_EXPIRATION_SECONDS,
        httponly=True,
        samesite='Lax'
    )
    return response


def clear_jwt_auth_cookie(response):
    """Clears JWT cookie on response object."""
    response.delete_cookie('jwt_token')
    return response
