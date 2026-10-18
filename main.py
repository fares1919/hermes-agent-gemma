import os
import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient

APP_NAME = "Hermes Agent"

DEFAULT_SYSTEM = (
    "Tu es Hermes Agent, un assistant expert en automatisation. "
    "Reponds en francais de facon concise et utile."
)

# --------------------------------------------------------------------------
# Fournisseurs LLM (tous compatibles OpenAI : seule la base_url change)
# --------------------------------------------------------------------------
PROVIDER_ORDER = [p.strip() for p in os.getenv("PROVIDER_ORDER", "river,ollama").split(",") if p.strip()]
COOLDOWN_SECONDS = float(os.getenv("PROVIDER_COOLDOWN", "300"))
REQUEST_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))

PROVIDERS = {
    "river": {
        "base_url": os.getenv("RIVER_BASE_URL", "https://api.river.ai/v1"),
        "api_key": os.getenv("RIVER_API_KEY"),
        "model": os.getenv("RIVER_MODEL", "deepseek-ai/DeepSeek-V4.1-Flash"),
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "https://ollama.com/v1"),
        "api_key": os.getenv("OLLAMA_API_KEY"),
        "model": os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud"),
    },
}

_clients = {}
_state = {
    name: {"cooldown_until": 0.0, "last_error": None, "last_ok": None}
    for name in PROVIDERS
}

# --------------------------------------------------------------------------
# Memoire (MongoDB Atlas)
# --------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "10"))
MONGO_DB = os.getenv("MONGO_DB", "hermes")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "messages")

_mongo_client = None
_collection = None

app = FastAPI(title="Hermes Agent - multi-provider + memory")


def _mask(secret):
    if not secret:
        return None
    return secret[:6] + "****" + secret[-4:] if len(secret) > 12 else "****"


def get_collection():
    global _mongo_client, _collection
    if not MONGO_URI:
        return None
    if _collection is None:
        _mongo_client = MongoClient(
            MONGO_URI, serverSelectionTimeoutMS=5000, appname="hermes-agent-gemma"
        )
        _collection = _mongo_client[MONGO_DB][MONGO_COLLECTION]
        _collection.create_index([("user_id", 1), ("session_id", 1), ("created_at", -1)])
    return _collection


def get_client(name):
    cfg = PROVIDERS[name]
    if not cfg["api_key"]:
        raise RuntimeError(f"{name}: cle API manquante")
    if name not in _clients:
        _clients[name] = OpenAI(
            base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=REQUEST_TIMEOUT
        )
    return _clients[name]


def call_llm(messages):
    """Essaie les fournisseurs dans l'ordre, avec mise en quarantaine des
    fournisseurs en echec (circuit breaker en memoire)."""
    now = time.time()
    order = [p for p in PROVIDER_ORDER if p in PROVIDERS]
    ready = [p for p in order if _state[p]["cooldown_until"] <= now]
    cooling = sorted(
        [p for p in order if _state[p]["cooldown_until"] > now],
        key=lambda p: _state[p]["cooldown_until"],
    )
    errors = []
    for name in ready + cooling:
        cfg = PROVIDERS[name]
        try:
            client = get_client(name)
            completion = client.chat.completions.create(
                model=cfg["model"], messages=messages, temperature=0.7
            )
            _state[name].update(
                cooldown_until=0.0,
                last_error=None,
                last_ok=datetime.now(timezone.utc).isoformat(),
            )
            return completion.choices[0].message.content, name, cfg["model"]
        except Exception as exc:  # noqa: BLE001 - on veut basculer sur tout echec
            _state[name].update(
                cooldown_until=time.time() + COOLDOWN_SECONDS,
                last_error=str(exc)[:300],
            )
            errors.append(f"{name}: {exc}")
    raise HTTPException(
        status_code=502,
        detail="Aucun fournisseur disponible -> " + " | ".join(errors),
    )


class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "default"
    session_id: str = "default"
    system: str = DEFAULT_SYSTEM


@app.get("/")
def home():
    return {
        "status": "live",
        "app": APP_NAME,
        "memory": bool(MONGO_URI),
        "provider_order": [p for p in PROVIDER_ORDER if p in PROVIDERS],
        "models": {n: PROVIDERS[n]["model"] for n in PROVIDER_ORDER if n in PROVIDERS},
    }


@app.get("/providers")
def providers():
    now = time.time()
    out = []
    for name in PROVIDER_ORDER:
        if name not in PROVIDERS:
            continue
        cfg, st = PROVIDERS[name], _state[name]
        out.append(
            {
                "name": name,
                "model": cfg["model"],
                "configured": bool(cfg["api_key"]),
                "key": _mask(cfg["api_key"]),
                "status": "cooldown" if st["cooldown_until"] > now else "ready",
                "cooldown_remaining_s": max(0, round(st["cooldown_until"] - now)),
                "last_ok": st["last_ok"],
                "last_error": st["last_error"],
            }
        )
    return {"order": PROVIDER_ORDER, "providers": out}


@app.get("/health/db")
def health_db():
    col = get_collection()
    if col is None:
        return {"mongo": False, "detail": "MONGO_URI non configuree"}
    try:
        col.database.client.admin.command("ping")
        return {
            "mongo": True,
            "database": MONGO_DB,
            "collection": MONGO_COLLECTION,
            "documents": col.estimated_document_count(),
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"MongoDB: {exc}")


@app.get("/history/{user_id}")
def history(user_id: str, limit: int = 10):
    col = get_collection()
    if col is None:
        raise HTTPException(status_code=503, detail="Memoire non configuree (MONGO_URI)")
    docs = list(
        col.find({"user_id": user_id}, {"_id": 0})
        .sort("created_at", -1)
        .limit(max(1, min(limit, 100)))
    )
    return {"user_id": user_id, "count": len(docs), "messages": docs}


@app.post("/chat")
def chat(req: ChatRequest):
    col = get_collection()
    history_docs = []
    if col is not None:
        try:
            history_docs = list(
                col.find({"user_id": req.user_id, "session_id": req.session_id}, {"_id": 0})
                .sort("created_at", -1)
                .limit(MEMORY_TURNS)
            )
            history_docs.reverse()
        except Exception:  # noqa: BLE001 - la memoire ne doit pas casser le chat
            history_docs = []

    messages = [{"role": "system", "content": req.system}]
    for doc in history_docs:
        messages.append({"role": "user", "content": doc.get("prompt", "")})
        messages.append({"role": "assistant", "content": doc.get("response", "")})
    messages.append({"role": "user", "content": req.prompt})

    answer, provider_name, model_name = call_llm(messages)

    if col is not None:
        try:
            col.insert_one(
                {
                    "user_id": req.user_id,
                    "session_id": req.session_id,
                    "prompt": req.prompt,
                    "response": answer,
                    "provider": provider_name,
                    "model": model_name,
                    "created_at": datetime.now(timezone.utc),
                }
            )
        except Exception:  # noqa: BLE001
            pass

    return {
        "provider": provider_name,
        "model": model_name,
        "fallback_used": provider_name != (PROVIDER_ORDER[0] if PROVIDER_ORDER else None),
        "remembered_turns": len(history_docs),
        "response": answer,
    }
