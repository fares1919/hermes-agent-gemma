import ast
import ipaddress
import json
import math
import operator
import os
import re
import socket
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient

APP_NAME = "Hermes Agent"

DEFAULT_SYSTEM = (
    "Tu es Hermes Agent, un assistant expert en automatisation. "
    "Tu disposes d'outils : utilise-les quand c'est utile (heure, calculs, notes, lecture de page web). "
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
# Memoire (MongoDB Atlas) - degrade proprement si indisponible
# --------------------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "10"))
MONGO_DB = os.getenv("MONGO_DB", "hermes")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "messages")
MONGO_NOTES_COLLECTION = os.getenv("MONGO_NOTES_COLLECTION", "notes")
MONGO_TIMEOUT_MS = int(os.getenv("MONGO_TIMEOUT_MS", "4000"))
MONGO_RETRY_SECONDS = float(os.getenv("MONGO_RETRY_SECONDS", "60"))

_mongo_client = None
_collection = None
_index_ready = False
_mongo_error = None
_mongo_retry_after = 0.0

# --------------------------------------------------------------------------
# Skills (function calling)
# --------------------------------------------------------------------------
TOOLS_ENABLED = os.getenv("TOOLS_ENABLED", "true").lower() in ("1", "true", "yes", "on")
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "4"))
WEB_FETCH_MAX_CHARS = int(os.getenv("WEB_FETCH_MAX_CHARS", "4000"))

app = FastAPI(title="Hermes Agent - multi-provider + memory + skills")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _mask(secret):
    if not secret:
        return None
    return secret[:6] + "****" + secret[-4:] if len(secret) > 12 else "****"


# ------------------------------- Mongo ------------------------------------

def get_collection():
    """Retourne la collection Mongo, ou None si indisponible.
    Ne leve jamais : la memoire ne doit pas casser l'API."""
    global _mongo_client, _collection, _index_ready, _mongo_error, _mongo_retry_after

    if not MONGO_URI:
        return None

    if _collection is None:
        try:
            _mongo_client = MongoClient(
                MONGO_URI,
                serverSelectionTimeoutMS=MONGO_TIMEOUT_MS,
                appname="hermes-agent-gemma",
            )
            _collection = _mongo_client[MONGO_DB][MONGO_COLLECTION]
        except Exception as exc:  # noqa: BLE001
            _mongo_error = str(exc)[:300]
            _mongo_retry_after = time.time() + MONGO_RETRY_SECONDS
            return None

    if not _index_ready:
        if time.time() < _mongo_retry_after:
            return None
        try:
            _collection.create_index([("user_id", 1), ("session_id", 1), ("created_at", -1)])
            _index_ready = True
            _mongo_error = None
        except Exception as exc:  # noqa: BLE001
            _mongo_error = str(exc)[:300]
            _mongo_retry_after = time.time() + MONGO_RETRY_SECONDS
            return None

    return _collection


def get_notes_collection():
    col = get_collection()
    if col is None:
        return None
    return col.database[MONGO_NOTES_COLLECTION]


# ----------------------------- Providers ----------------------------------

def get_client(name):
    cfg = PROVIDERS[name]
    if not cfg["api_key"]:
        raise RuntimeError(f"{name}: cle API manquante")
    if name not in _clients:
        _clients[name] = OpenAI(
            base_url=cfg["base_url"], api_key=cfg["api_key"], timeout=REQUEST_TIMEOUT
        )
    return _clients[name]


def call_llm(messages, tools=None):
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
        kwargs = {"model": cfg["model"], "messages": messages, "temperature": 0.7}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            client = get_client(name)
            completion = client.chat.completions.create(**kwargs)
            _state[name].update(
                cooldown_until=0.0,
                last_error=None,
                last_ok=datetime.now(timezone.utc).isoformat(),
            )
            return completion.choices[0].message, name, cfg["model"]
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


# ------------------------------- Skills -----------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_MATH_NAMES = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
        return _ALLOWED_UNARY[type(node.op)](_eval_ast(node.operand))
    if isinstance(node, ast.Name) and node.id in _MATH_NAMES:
        return _MATH_NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _MATH_NAMES.get(node.func.id)
        if callable(fn):
            return fn(*[_eval_ast(a) for a in node.args])
    raise ValueError("expression non autorisee")


def _is_public_host(host):
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:  # noqa: BLE001
        return False
    for _fam, _type, _proto, _canon, sockaddr in infos:
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return False
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast:
            return False
    return True


def tool_get_datetime(args):
    offset = float(args.get("timezone_offset_hours", 0) or 0)
    now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=offset)))
    return json.dumps(
        {
            "iso": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "heure": now.strftime("%H:%M:%S"),
            "jour": now.strftime("%A"),
            "timezone": now.tzname(),
            "utc_offset_hours": offset,
        },
        ensure_ascii=False,
    )


def tool_calculate(args):
    expr = str(args.get("expression", ""))
    if len(expr) > 200:
        return "erreur: expression trop longue"
    try:
        value = _eval_ast(ast.parse(expr, mode="eval"))
    except Exception as exc:  # noqa: BLE001
        return f"erreur: {exc}"
    return json.dumps({"expression": expr, "resultat": value})


def tool_save_note(args, user_id):
    notes = get_notes_collection()
    if notes is None:
        return "erreur: memoire indisponible"
    text = str(args.get("note", "")).strip()
    if not text:
        return "erreur: note vide"
    notes.insert_one(
        {"user_id": user_id, "note": text[:2000], "created_at": datetime.now(timezone.utc)}
    )
    total = notes.count_documents({"user_id": user_id})
    return json.dumps({"saved": text[:2000], "total_notes": total}, ensure_ascii=False)


def tool_list_notes(args, user_id):
    notes = get_notes_collection()
    if notes is None:
        return "erreur: memoire indisponible"
    docs = list(notes.find({"user_id": user_id}, {"_id": 0, "note": 1}).sort("created_at", -1).limit(20))
    return json.dumps([d["note"] for d in docs], ensure_ascii=False)


def tool_web_fetch(args):
    url = str(args.get("url", "")).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "erreur: seules les URL http(s) sont autorisees"
    if not parsed.hostname or not _is_public_host(parsed.hostname):
        return "erreur: hote non autorise"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
            raw = resp.read(400000).decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return f"erreur: {exc}"
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:WEB_FETCH_MAX_CHARS]


def execute_tool(name, args, user_id):
    try:
        if name == "get_datetime":
            return tool_get_datetime(args)
        if name == "calculate":
            return tool_calculate(args)
        if name == "save_note":
            return tool_save_note(args, user_id)
        if name == "list_notes":
            return tool_list_notes(args, user_id)
        if name == "web_fetch":
            return tool_web_fetch(args)
        return f"erreur: outil inconnu '{name}'"
    except Exception as exc:  # noqa: BLE001
        return f"erreur: {exc}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Renvoie la date et l'heure actuelles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_offset_hours": {
                        "type": "number",
                        "description": "Decalage horaire par rapport a UTC (ex: 1 pour Alger).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Calcule une expression mathematique (+, -, *, /, **, sqrt, log...).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Ex: (1200*0.19)+45"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Enregistre une note durable pour l'utilisateur courant.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string"}},
                "required": ["note"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "Liste les notes enregistrees pour l'utilisateur courant.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Recupere le texte d'une page web publique (http/https).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


def run_agent(messages, user_id, use_tools=True):
    """Boucle d'appel d'outils : tant que le modele demande un outil, on l'execute."""
    used_tools = []
    provider_name = model_name = None
    active_tools = TOOL_SCHEMAS if (TOOLS_ENABLED and use_tools) else None

    for _ in range(MAX_TOOL_ITERATIONS):
        msg, provider_name, model_name = call_llm(messages, tools=active_tools)
        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return msg.content or "", provider_name, model_name, used_tools

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            result = execute_tool(tc.function.name, args, user_id)
            used_tools.append({"tool": tc.function.name, "arguments": args})
            messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": str(result)[:6000]}
            )

    msg, provider_name, model_name = call_llm(messages, tools=None)
    return msg.content or "", provider_name, model_name, used_tools


# ------------------------------ Endpoints ---------------------------------

class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "default"
    session_id: str = "default"
    system: str = DEFAULT_SYSTEM
    use_tools: bool = True


@app.get("/")
def home():
    return {
        "status": "live",
        "app": APP_NAME,
        "memory": "ready" if _index_ready else ("configured" if MONGO_URI else "off"),
        "storage_configured": bool(MONGO_URI),
        "tools_enabled": TOOLS_ENABLED,
        "tools": TOOL_NAMES,
        "provider_order": [p for p in PROVIDER_ORDER if p in PROVIDERS],
        "models": {n: PROVIDERS[n]["model"] for n in PROVIDER_ORDER if n in PROVIDERS},
    }


@app.get("/tools")
def tools():
    return {"enabled": TOOLS_ENABLED, "tools": TOOL_SCHEMAS}


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
    info = {
        "mongo": col is not None,
        "configured": bool(MONGO_URI),
        "database": MONGO_DB,
        "collection": MONGO_COLLECTION,
        "error": _mongo_error,
        "retry_in_s": max(0, round(_mongo_retry_after - time.time())),
    }
    if col is None:
        return info
    try:
        col.database.client.admin.command("ping")
        info["documents"] = col.estimated_document_count()
    except Exception as exc:  # noqa: BLE001
        info["mongo"] = False
        info["error"] = str(exc)[:300]
    return info


@app.get("/history/{user_id}")
def history(user_id: str, limit: int = 10):
    col = get_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    docs = list(
        col.find({"user_id": user_id}, {"_id": 0})
        .sort("created_at", -1)
        .limit(max(1, min(limit, 100)))
    )
    return {"user_id": user_id, "count": len(docs), "messages": docs}


@app.delete("/history/{user_id}")
def delete_history(user_id: str, session_id: str | None = None):
    col = get_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    query = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    deleted = col.delete_many(query).deleted_count
    notes = get_notes_collection()
    notes_deleted = 0
    if notes is not None:
        notes_deleted = notes.delete_many({"user_id": user_id}).deleted_count
    return {"user_id": user_id, "deleted_messages": deleted, "deleted_notes": notes_deleted}


@app.get("/notes/{user_id}")
def notes(user_id: str, limit: int = 50):
    notes_col = get_notes_collection()
    if notes_col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    docs = list(
        notes_col.find({"user_id": user_id}, {"_id": 0, "note": 1, "created_at": 1})
        .sort("created_at", -1)
        .limit(max(1, min(limit, 200)))
    )
    return {"user_id": user_id, "count": len(docs), "notes": docs}


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

    answer, provider_name, model_name, used_tools = run_agent(
        messages, req.user_id, use_tools=req.use_tools
    )

    stored = False
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
                    "tools": [t["tool"] for t in used_tools],
                    "created_at": datetime.now(timezone.utc),
                }
            )
            stored = True
        except Exception:  # noqa: BLE001
            stored = False

    return {
        "provider": provider_name,
        "model": model_name,
        "fallback_used": provider_name != (PROVIDER_ORDER[0] if PROVIDER_ORDER else None),
        "memory_enabled": col is not None,
        "stored": stored,
        "remembered_turns": len(history_docs),
        "tools_used": used_tools,
        "response": answer,
    }
