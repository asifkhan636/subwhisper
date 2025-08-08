import yaml
from pathlib import Path
from typing import Dict, Callable
from fastapi import Depends, HTTPException, Header

AUTH_FILE = Path(__file__).with_name("auth.yaml")

def _load_tokens() -> Dict[str, str]:
    if AUTH_FILE.exists():
        data = yaml.safe_load(AUTH_FILE.read_text()) or {}
        return data.get("tokens", {})
    return {}

TOKENS = _load_tokens()

def get_role(authorization: str | None = Header(None)) -> str:
    prefix = "Bearer "
    if not authorization or not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization[len(prefix):]
    role = TOKENS.get(token)
    if not role:
        raise HTTPException(status_code=403, detail="Invalid token")
    return role

def require_role(*allowed: str) -> Callable:
    def dependency(role: str = Depends(get_role)) -> None:
        if role not in allowed:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
    return dependency
