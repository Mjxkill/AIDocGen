import os
import json
import time
import base64
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

# Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "73f1e8a9c2b4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 1 week

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/auth/login")

# Base path relative to this file
BASE_DIR = Path(__file__).parent.parent.resolve()
USERS_FILE = BASE_DIR / "data" / "users.json"

def _get_users():
    if not USERS_FILE.exists():
        print(f"Creating default admin at {USERS_FILE}")
        # Create default admin user: admin / admin
        admin_user = {
            "username": "admin",
            "hashed_password": pwd_context.hash("admin"),
            "role": "admin",
            "id": "u-admin",
            "must_change_password": True,
        }
        USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(USERS_FILE, "w") as f:
            json.dump([admin_user], f)
    return json.loads(USERS_FILE.read_text())

def _save_users(users):
    USERS_FILE.write_text(json.dumps(users))

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    users = _get_users()
    user = next((u for u in users if u["username"] == username), None)
    if user is None:
        raise credentials_exception
    return user

async def get_admin_user(current_user: dict = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user


# ── short-lived download tokens ──
# Used to authenticate native browser downloads (where the Authorization
# header cannot be set). Token = HMAC over (user_id|scope|exp) with the
# JWT secret. Scope is a string the server picks per-resource (eg.
# "audiobook:JOBID:m4b") so a token can only download one specific file.

def create_download_token(user_id: str, scope: str, expires_in: int = 600) -> str:
    """Returns a URL-safe token good for `expires_in` seconds, scoped to
    one specific download. Embed in `?token=` query params."""
    exp = int(time.time()) + max(60, int(expires_in))
    payload = f"{user_id}|{scope}|{exp}"
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    raw = f"{payload}|{sig}".encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def verify_download_token(token: str, expected_scope: str) -> Optional[str]:
    """Returns the user_id if the token is valid for the given scope,
    else None. Constant-time signature compare."""
    if not token:
        return None
    try:
        pad = "=" * ((4 - len(token) % 4) % 4)
        raw = base64.urlsafe_b64decode((token + pad).encode()).decode()
        user_id, scope, exp_str, sig = raw.split("|", 3)
    except Exception:
        return None
    if scope != expected_scope:
        return None
    try:
        exp = int(exp_str)
    except ValueError:
        return None
    if exp < int(time.time()):
        return None
    expected_sig = hmac.new(SECRET_KEY.encode(),
                            f"{user_id}|{scope}|{exp}".encode(),
                            hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return None
    return user_id
