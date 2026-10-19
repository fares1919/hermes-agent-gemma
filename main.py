import ast
import hmac
import html as htmlmod
import io
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
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from pymongo import MongoClient

APP_NAME = "Hermes Agent"
CHAT_UI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat.html")

DEFAULT_SYSTEM = (
    "Tu es Hermes Agent, un assistant expert en automatisation. "
    "Tu disposes d'outils : utilise-les quand c'est utile (heure, calculs, notes, rappels, "
    "recherche web, lecture de page, documents joints). "
    "Ne pretends jamais avoir fait une recherche, enregistre une note ou lu un document "
    "sans avoir appele l'outil correspondant. "
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
MONGO_REMINDERS_COLLECTION = os.getenv("MONGO_REMINDERS_COLLECTION", "reminders")
MONGO_DOCS_COLLECTION = os.getenv("MONGO_DOCS_COLLECTION", "documents")
MONGO_SESSIONS_COLLECTION = os.getenv("MONGO_SESSIONS_COLLECTION", "sessions")
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
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "5"))
WEB_FETCH_MAX_CHARS = int(os.getenv("WEB_FETCH_MAX_CHARS", "4000"))
# Documents joints
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
DOC_TEXT_LIMIT = int(os.getenv("DOC_TEXT_LIMIT", "300000"))
DOC_READ_CHARS = int(os.getenv("DOC_READ_CHARS", "6000"))

# Recherche web : un fournisseur a cle si disponible, sinon DuckDuckGo (sans cle).
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "auto").strip().lower()
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
SEARXNG_URL = os.getenv("SEARXNG_URL")
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "5"))
SEARCH_TIMEOUT = float(os.getenv("SEARCH_TIMEOUT", "20"))
SEARCH_LANG = os.getenv("SEARCH_LANG", "fr").strip()
SEARCH_COUNTRY = os.getenv("SEARCH_COUNTRY", "").strip()
SEARCH_EXTRA_SNIPPETS = os.getenv("SEARCH_EXTRA_SNIPPETS", "true").lower() in ("1", "true", "yes", "on")
# Garde-fou : au-dela de ce nombre d'appels payants par jour, on repasse
# automatiquement sur le moteur gratuit. 30/jour ~ 900/mois, sous le credit gratuit.
SEARCH_DAILY_LIMIT = int(os.getenv("SEARCH_DAILY_LIMIT", "30"))
PAID_BACKENDS = ("brave", "tavily", "serper")
_search_usage = {"day": "", "count": 0}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

app = FastAPI(title="Hermes Agent - multi-provider + memory + skills")

# --------------------------------------------------------------------------
# Securite : cle d'acces, anti-abus, cloisonnement CORS
# --------------------------------------------------------------------------
# HERMES_TOKEN : si defini, toutes les routes de donnees l'exigent
# (en-tete X-Hermes-Key ou Authorization: Bearer). Si absent, l'API reste
# ouverte comme avant -> aucun risque de blocage total par erreur de config,
# mais l'etat attendu en production est : token defini.
HERMES_TOKEN = os.getenv("HERMES_TOKEN", "").strip()
AUTH_ENABLED = bool(HERMES_TOKEN)

# L'interface est servie par l'API elle-meme (meme origine) : par defaut on
# n'autorise donc AUCUNE origine tierce. A renseigner seulement si vous
# hebergez l'interface ailleurs.
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

# Plafonds d'usage (par IP) pour eviter l'epuisement du quota LLM.
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
RATE_LIMIT_CHAT_PER_MIN = int(os.getenv("RATE_LIMIT_CHAT_PER_MIN", "20"))
# Plafonds de stockage par utilisateur (le MongoDB gratuit plafonne a 512 Mo).
MAX_DOCS_PER_USER = int(os.getenv("MAX_DOCS_PER_USER", "50"))
MAX_DOCS_BYTES_PER_USER = int(os.getenv("MAX_DOCS_BYTES_PER_USER", str(40 * 1024 * 1024)))
# Render place le vrai client en dernier dans X-Forwarded-For.
TRUST_PROXY = os.getenv("TRUST_PROXY", "true").lower() in ("1", "true", "yes", "on")

PUBLIC_PATHS = {"/", "/ui"}

_SECRETS_RE = re.compile(
    r"(rv_[A-Za-z0-9_\-]{6,}|rnd_[A-Za-z0-9]{6,}|sk-[A-Za-z0-9_\-]{6,}"
    r"|mongodb\+srv://[^\s\"']+|(?i:bearer)\s+[A-Za-z0-9_\-\.]{8,})"
)


def _scrub(text):
    """Retire tout secret des messages destines a sortir de l'API."""
    if not text:
        return text
    return _SECRETS_RE.sub("[masque]", str(text))


def _token_ok(request):
    provided = (request.headers.get("x-hermes-key") or "").strip()
    if not provided:
        parts = (request.headers.get("authorization") or "").split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            provided = parts[1].strip()
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), HERMES_TOKEN.encode("utf-8"))


def _client_ip(request):
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            # La derniere entree est celle ajoutee par le proxy de confiance :
            # les valeurs fournies par le client sont a gauche, donc falsifiables.
            return fwd.split(",")[-1].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "inconnu"


_rate_windows = {}


def _rate_ok(key, limit):
    """Fenetre glissante de 60 s. Ne leve jamais."""
    if limit <= 0:
        return True
    now = time.time()
    cutoff = now - 60.0
    window = _rate_windows.get(key)
    if window is None:
        window = _rate_windows[key] = []
    while window and window[0] <= cutoff:
        window.pop(0)
    if len(window) >= limit:
        return False
    window.append(now)
    if len(_rate_windows) > 4000:
        for stale in [k for k, w in _rate_windows.items() if not w or w[-1] <= cutoff]:
            _rate_windows.pop(stale, None)
    return True


@app.middleware("http")
async def security_guard(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path.rstrip("/") or "/"
    ip = _client_ip(request)
    is_chat = path.startswith("/chat")
    limit = RATE_LIMIT_CHAT_PER_MIN if is_chat else RATE_LIMIT_PER_MIN
    if not _rate_ok(f"{ip}:{'chat' if is_chat else 'api'}", limit):
        return JSONResponse(
            {"detail": "Trop de requetes. Reessayez dans une minute."},
            status_code=429,
            headers={"Retry-After": "60"},
        )

    if AUTH_ENABLED and path not in PUBLIC_PATHS and not _token_ok(request):
        return JSONResponse(
            {"detail": "Cle d'acces requise ou invalide."},
            status_code=401,
            headers={"WWW-Authenticate": 'Bearer realm="hermes"'},
        )

    return await call_next(request)


@app.on_event("startup")
def _startup_notice():
    print(
        f"[hermes] authentification={'ACTIVE' if AUTH_ENABLED else 'DESACTIVEE'}"
        f" | origines autorisees={ALLOWED_ORIGINS or 'aucune (meme origine seulement)'}"
        f" | limite chat={RATE_LIMIT_CHAT_PER_MIN}/min, api={RATE_LIMIT_PER_MIN}/min",
        flush=True,
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Hermes-Key", "Authorization"],
)


def _mask(secret):
    if not secret:
        return None
    return secret[:6] + "****" + secret[-4:] if len(secret) > 12 else "****"


def _iso(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return None if value is None else str(value)


def _as_utc(value):
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return None


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


def _sibling_collection(name):
    col = get_collection()
    if col is None:
        return None
    return col.database[name]


def get_notes_collection():
    return _sibling_collection(MONGO_NOTES_COLLECTION)


def get_reminders_collection():
    return _sibling_collection(MONGO_REMINDERS_COLLECTION)


def get_docs_collection():
    return _sibling_collection(MONGO_DOCS_COLLECTION)


def get_sessions_collection():
    return _sibling_collection(MONGO_SESSIONS_COLLECTION)


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


def _order():
    """Fournisseurs disponibles, les sains d'abord, puis ceux en quarantaine
    (du plus proche de la sortie au plus lointain) : on tente toujours tout."""
    now = time.time()
    listed = [p for p in PROVIDER_ORDER if p in PROVIDERS]
    ready = [p for p in listed if _state[p]["cooldown_until"] <= now]
    cooling = sorted(
        [p for p in listed if _state[p]["cooldown_until"] > now],
        key=lambda p: _state[p]["cooldown_until"],
    )
    return ready + cooling


def _ok(name):
    _state[name].update(
        cooldown_until=0.0, last_error=None, last_ok=datetime.now(timezone.utc).isoformat()
    )


def _fail(name, exc):
    _state[name].update(
        cooldown_until=time.time() + COOLDOWN_SECONDS, last_error=_scrub(str(exc))[:300]
    )


def _kwargs(name, tools, stream):
    cfg = PROVIDERS[name]
    kwargs = {
        "model": cfg["model"],
        "temperature": 0.7,
    }
    if stream:
        kwargs["stream"] = True
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return kwargs


def call_llm(messages, tools=None):
    """Version non streamee : essaie les fournisseurs dans l'ordre."""
    errors = []
    for name in _order():
        kwargs = _kwargs(name, tools, stream=False)
        kwargs["messages"] = messages
        try:
            completion = get_client(name).chat.completions.create(**kwargs)
            _ok(name)
            return completion.choices[0].message, name, PROVIDERS[name]["model"]
        except Exception as exc:  # noqa: BLE001 - on veut basculer sur tout echec
            _fail(name, exc)
            errors.append(f"{name}: {exc}")
    raise HTTPException(
        status_code=502, detail="Aucun fournisseur disponible -> " + " | ".join(_scrub(e) for e in errors)
    )


def stream_once(messages, tools):
    """Ouvre un stream chez le premier fournisseur disponible, diffuse chaque
    fragment au client, et retourne (provider, model, contenu, appels_outils).
    Tant que rien n'a ete diffuse, un echec permet de basculer de fournisseur."""
    last_exc = None
    for name in _order():
        kwargs = _kwargs(name, tools, stream=True)
        kwargs["messages"] = messages
        emitted = False
        content = ""
        calls = {}
        try:
            stream = get_client(name).chat.completions.create(**kwargs)
            for chunk in stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if text:
                        content += text
                        emitted = True
                        yield ("delta", {"text": text})
                    for tc in getattr(delta, "tool_calls", None) or []:
                        slot = calls.setdefault(
                            getattr(tc, "index", 0) or 0,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        if getattr(tc, "id", None):
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] = slot["name"] + fn.name if slot["name"] else fn.name
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments
            _ok(name)
            return name, PROVIDERS[name]["model"], content, calls
        except Exception as exc:  # noqa: BLE001
            _fail(name, exc)
            last_exc = exc
            if emitted:
                raise  # deja diffuse au client : plus de bascule possible
    raise RuntimeError(f"Aucun fournisseur disponible -> {last_exc}")


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
    "sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
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
    """Refuse localhost et les plages privees (garde-fou SSRF basique)."""
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


def _strip_html(text):
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", htmlmod.unescape(text)).strip()


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
    return json.dumps(
        {"saved": text[:2000], "total_notes": notes.count_documents({"user_id": user_id})},
        ensure_ascii=False,
    )


def tool_list_notes(args, user_id):
    notes = get_notes_collection()
    if notes is None:
        return "erreur: memoire indisponible"
    docs = list(
        notes.find({"user_id": user_id}, {"_id": 0, "note": 1})
        .sort("created_at", -1)
        .limit(20)
    )
    return json.dumps([d["note"] for d in docs], ensure_ascii=False)


def tool_web_fetch(args):
    url = str(args.get("url", "")).strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "erreur: seules les URL http(s) sont autorisees"
    if not parsed.hostname or not _is_public_host(parsed.hostname):
        return "erreur: hote non autorise"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            raw = resp.read(400000)
            charset = resp.headers.get_content_charset() or "utf-8"
    except Exception as exc:  # noqa: BLE001
        return f"erreur: {exc}"
    try:
        text = raw.decode(charset, "replace")
    except LookupError:
        text = raw.decode("utf-8", "replace")
    return _strip_html(text)[:WEB_FETCH_MAX_CHARS]


# ------------------------------ Recherche web -----------------------------

DDG_LINK_RE = re.compile(
    r"<a[^>]*href=\"([^\"]+)\"[^>]*class=['\"]result-link['\"][^>]*>(.*?)</a>", re.S | re.I
)
DDG_SNIP_RE = re.compile(r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>", re.S | re.I)


def _http_get(url, headers=None, timeout=None):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=timeout or SEARCH_TIMEOUT) as resp:  # noqa: S310
        raw = resp.read(600000)
        charset = resp.headers.get_content_charset() or "utf-8"
    try:
        return raw.decode(charset)
    except (UnicodeDecodeError, LookupError):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("cp1252", "replace")


def _http_post_json(url, payload, headers=None):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", "replace"))


def _clean_ddg_url(href):
    href = htmlmod.unescape(href)
    if href.startswith("//"):
        href = "https:" + href
    match = re.search(r"[?&]uddg=([^&]+)", href)
    if match:
        return urllib.parse.unquote(match.group(1))
    return href


def search_duckduckgo(query, limit):
    doc = _http_get("https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote_plus(query))
    links = DDG_LINK_RE.findall(doc)
    snippets = DDG_SNIP_RE.findall(doc)
    out = []
    for index, (href, title) in enumerate(links[:limit]):
        out.append(
            {
                "titre": _strip_html(title),
                "url": _clean_ddg_url(href),
                "extrait": _strip_html(snippets[index]) if index < len(snippets) else "",
            }
        )
    return out


def search_brave(query, limit):
    params = {"q": query, "count": limit}
    if SEARCH_COUNTRY:
        params["country"] = SEARCH_COUNTRY
    if SEARCH_LANG:
        params["search_lang"] = SEARCH_LANG
    if SEARCH_EXTRA_SNIPPETS:
        params["extra_snippets"] = "true"
    data = json.loads(
        _http_get(
            "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode(params),
            headers={"X-Subscription-Token": SEARCH_API_KEY or "", "Accept": "application/json"},
        )
    )
    out = []
    for item in (data.get("web", {}) or {}).get("results", [])[:limit]:
        snippet = _strip_html(item.get("description", ""))
        extras = [_strip_html(e) for e in (item.get("extra_snippets") or [])[:3]]
        if extras:
            snippet = (snippet + " " + " ".join(extras)).strip()
        out.append(
            {"titre": item.get("title", ""), "url": item.get("url", ""), "extrait": snippet[:600]}
        )
    return out


def search_tavily(query, limit):
    data = _http_post_json(
        "https://api.tavily.com/search",
        {
            "api_key": SEARCH_API_KEY,
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
        },
        headers={"Authorization": f"Bearer {SEARCH_API_KEY}"},
    )
    return [
        {"titre": r.get("title", ""), "url": r.get("url", ""), "extrait": r.get("content", "")}
        for r in (data.get("results") or [])[:limit]
    ]


def search_serper(query, limit):
    data = _http_post_json(
        "https://google.serper.dev/search",
        {"q": query, "num": limit},
        headers={"X-API-KEY": SEARCH_API_KEY or ""},
    )
    return [
        {
            "titre": r.get("title", ""),
            "url": r.get("link", ""),
            "extrait": r.get("snippet", ""),
        }
        for r in (data.get("organic") or [])[:limit]
    ]


def search_searxng(query, limit):
    data = json.loads(
        _http_get(
            (SEARXNG_URL or "").rstrip("/")
            + "/search?"
            + urllib.parse.urlencode({"q": query, "format": "json"}),
            headers={"Accept": "application/json"},
        )
    )
    return [
        {"titre": r.get("title", ""), "url": r.get("url", ""), "extrait": r.get("content", "")}
        for r in (data.get("results") or [])[:limit]
    ]


SEARCH_BACKENDS = {
    "brave": search_brave,
    "tavily": search_tavily,
    "serper": search_serper,
    "searxng": search_searxng,
    "duckduckgo": search_duckduckgo,
}


def _search_day_rollover():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if _search_usage["day"] != today:
        _search_usage["day"] = today
        _search_usage["count"] = 0


def search_budget():
    """Compteur d'appels payants du jour (remis a zero chaque jour UTC)."""
    _search_day_rollover()
    used = _search_usage["count"]
    unlimited = SEARCH_DAILY_LIMIT <= 0
    return {
        "limite_jour": None if unlimited else SEARCH_DAILY_LIMIT,
        "utilises_aujourd_hui": used,
        "restants_aujourd_hui": None if unlimited else max(0, SEARCH_DAILY_LIMIT - used),
    }


def _paid_allowed():
    _search_day_rollover()
    return SEARCH_DAILY_LIMIT <= 0 or _search_usage["count"] < SEARCH_DAILY_LIMIT


def _search_use():
    _search_day_rollover()
    _search_usage["count"] += 1


def _search_chain():
    """Ordre des moteurs : celui demande s'il y a une cle, sinon les moteurs
    sans cle (SearXNG si configure, puis DuckDuckGo).
    Des que le quota payant du jour est epuise, on repasse au gratuit."""
    if SEARCH_PROVIDER in SEARCH_BACKENDS and SEARCH_PROVIDER != "auto":
        usable = SEARCH_PROVIDER not in PAID_BACKENDS or SEARCH_API_KEY
        chain = [SEARCH_PROVIDER] if usable else []
    elif SEARCH_API_KEY:
        chain = ["brave", "tavily", "serper"]
    else:
        chain = []
    if not _paid_allowed():
        chain = [b for b in chain if b not in PAID_BACKENDS]
    if SEARXNG_URL:
        chain.append("searxng")
    chain.append("duckduckgo")
    seen = []
    for name in chain:
        if name not in seen:
            seen.append(name)
    return seen


def active_search_backend():
    return _search_chain()[0]


def tool_web_search(args):
    query = str(args.get("query", "")).strip()
    if not query:
        return "erreur: requete vide"
    limit = int(args.get("max_results") or SEARCH_MAX_RESULTS)
    limit = max(1, min(limit, 10))
    errors = []
    for backend in _search_chain():
        try:
            results = SEARCH_BACKENDS[backend](query, limit)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{backend}: {exc}")
            continue
        if results:
            if backend in PAID_BACKENDS:
                _search_use()
            return json.dumps(
                {
                    "requete": query,
                    "moteur": backend,
                    "resultats": results,
                },
                ensure_ascii=False,
            )
        errors.append(f"{backend}: aucun resultat")
    return json.dumps({"requete": query, "erreur": " | ".join(errors)}, ensure_ascii=False)


# ------------------------------- Rappels ----------------------------------


def _parse_when(raw):
    text = str(raw or "").strip()
    if not text:
        raise ValueError("date manquante")
    candidate = text.replace("Z", "+00:00").replace("z", "+00:00")
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%dT%H:%M", "%d/%m/%Y %H:%M", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            raise ValueError(f"format de date non reconnu: {text}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _reminder_out(doc, now):
    due = _as_utc(doc.get("due_at"))
    return {
        "id": doc.get("short_id"),
        "text": doc.get("text"),
        "due_at": _iso(due),
        "dans_s": round((due - now).total_seconds()) if due else None,
        "notifie": bool(doc.get("notified")),
        "cree_le": _iso(_as_utc(doc.get("created_at"))),
    }


def tool_set_reminder(args, user_id):
    col = get_reminders_collection()
    if col is None:
        return "erreur: memoire indisponible"
    text = str(args.get("text", "")).strip()
    if not text:
        return "erreur: texte du rappel vide"
    now = datetime.now(timezone.utc)
    try:
        if args.get("in_minutes") is not None:
            due = now + timedelta(minutes=float(args["in_minutes"]))
        else:
            due = _parse_when(args.get("when"))
    except (ValueError, TypeError) as exc:
        return f"erreur: {exc}"
    if due < now - timedelta(minutes=5):
        return json.dumps(
            {"erreur": "cette date est deja passee", "maintenant_utc": now.isoformat()},
            ensure_ascii=False,
        )
    short_id = uuid.uuid4().hex[:8]
    col.insert_one(
        {
            "short_id": short_id,
            "user_id": user_id,
            "text": text[:1000],
            "due_at": due,
            "notified": False,
            "created_at": now,
        }
    )
    return json.dumps(
        {
            "id": short_id,
            "text": text[:1000],
            "due_at_utc": due.isoformat(),
            "dans_minutes": round((due - now).total_seconds() / 60),
            "note": "le rappel sera signale a l'ouverture de l'interface",
        },
        ensure_ascii=False,
    )


def tool_list_reminders(args, user_id):
    col = get_reminders_collection()
    if col is None:
        return "erreur: memoire indisponible"
    now = datetime.now(timezone.utc)
    docs = list(
        col.find({"user_id": user_id, "notified": False}).sort("due_at", 1).limit(30)
    )
    return json.dumps([_reminder_out(d, now) for d in docs], ensure_ascii=False)


def tool_cancel_reminder(args, user_id):
    col = get_reminders_collection()
    if col is None:
        return "erreur: memoire indisponible"
    short_id = str(args.get("id", "")).strip()
    if not short_id:
        return "erreur: identifiant manquant"
    deleted = col.delete_many({"user_id": user_id, "short_id": short_id}).deleted_count
    return json.dumps({"annule": deleted}, ensure_ascii=False)


# ------------------------------ Documents ---------------------------------

TEXT_EXTENSIONS = (
    ".txt", ".md", ".log", ".csv", ".tsv", ".json", ".yml", ".yaml",
    ".py", ".js", ".ts", ".sql", ".ini", ".cfg", ".conf", ".srt", ".vtt",
)
HTML_EXTENSIONS = (".html", ".htm")


def extract_document_text(filename, raw):
    """Retourne (texte, type). Leve ValueError si le format n'est pas exploitable."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader  # import tardif : optionnel
        except Exception as exc:  # noqa: BLE001
            raise ValueError("lecture PDF indisponible sur ce serveur") from exc
        try:
            reader = PdfReader(io.BytesIO(raw))
            pages = [(page.extract_text() or "") for page in reader.pages[:300]]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"PDF illisible: {exc}") from exc
        return "\n\n".join(pages), "pdf"
    if ext in HTML_EXTENSIONS:
        return _strip_html(raw.decode("utf-8", "replace")), "html"
    if ext in TEXT_EXTENSIONS or not ext:
        return raw.decode("utf-8", "replace"), "texte"
    try:
        return raw.decode("utf-8"), "texte"
    except UnicodeDecodeError as exc:
        raise ValueError(f"format non pris en charge: {ext or 'inconnu'}") from exc


def _doc_out(doc):
    return {
        "id": doc.get("doc_id"),
        "nom": doc.get("name"),
        "type": doc.get("kind"),
        "caracteres": doc.get("chars"),
        "ajoute_le": _iso(_as_utc(doc.get("created_at"))),
    }


def _find_document(docs, user_id, reference):
    ref = str(reference or "").strip()
    if not ref:
        return None
    doc = docs.find_one({"user_id": user_id, "$or": [{"doc_id": ref}, {"name": ref}]})
    if doc is None:
        doc = docs.find_one({"user_id": user_id, "name": {"$regex": re.escape(ref), "$options": "i"}})
    if doc is None and not ref:
        doc = docs.find_one({"user_id": user_id}, sort=[("created_at", -1)])
    return doc


def tool_list_documents(args, user_id):
    docs = get_docs_collection()
    if docs is None:
        return "erreur: memoire indisponible"
    rows = list(
        docs.find({"user_id": user_id}, {"_id": 0, "text": 0}).sort("created_at", -1).limit(30)
    )
    return json.dumps([_doc_out(d) for d in rows], ensure_ascii=False)


def tool_read_document(args, user_id):
    docs = get_docs_collection()
    if docs is None:
        return "erreur: memoire indisponible"
    doc = _find_document(docs, user_id, args.get("document"))
    if doc is None:
        available = [
            d.get("name")
            for d in docs.find({"user_id": user_id}, {"_id": 0, "name": 1}).sort("created_at", -1).limit(20)
        ]
        return json.dumps(
            {"erreur": "document introuvable", "documents_disponibles": available}, ensure_ascii=False
        )
    text = doc.get("text") or ""
    start = max(0, int(args.get("debut") or 0))
    length = max(200, min(int(args.get("longueur") or DOC_READ_CHARS), DOC_READ_CHARS * 3))
    chunk = text[start:start + length]
    return json.dumps(
        {
            "document": doc.get("name"),
            "id": doc.get("doc_id"),
            "caracteres_total": len(text),
            "plage": f"{start}-{start + len(chunk)}",
            "suite_disponible": start + len(chunk) < len(text),
            "texte": chunk,
        },
        ensure_ascii=False,
    )


def tool_search_documents(args, user_id):
    docs = get_docs_collection()
    if docs is None:
        return "erreur: memoire indisponible"
    needle = str(args.get("query", "")).strip()
    if not needle:
        return "erreur: requete vide"
    limit = max(1, min(int(args.get("max_results") or 5), 20))
    pattern = re.compile(re.escape(needle), re.I)
    results = []
    for doc in docs.find(
        {"user_id": user_id}, {"_id": 0, "doc_id": 1, "name": 1, "text": 1}
    ).sort("created_at", -1).limit(20):
        text = doc.get("text") or ""
        extraits = []
        for match in pattern.finditer(text):
            extraits.append(
                text[max(0, match.start() - 120): match.end() + 120].replace("\n", " ").strip()
            )
            if len(extraits) >= 3:
                break
        if extraits:
            results.append(
                {
                    "id": doc.get("doc_id"),
                    "nom": doc.get("name"),
                    "occurrences": len(pattern.findall(text)),
                    "extraits": extraits,
                }
            )
        if len(results) >= limit:
            break
    return json.dumps({"requete": needle, "documents": results}, ensure_ascii=False)


def execute_tool(name, args, user_id):
    try:
        if name == "get_datetime":
            return tool_get_datetime(args)
        if name == "calculate":
            return tool_calculate(args)
        if name == "web_search":
            return tool_web_search(args)
        if name == "web_fetch":
            return tool_web_fetch(args)
        if name == "list_documents":
            return tool_list_documents(args, user_id)
        if name == "read_document":
            return tool_read_document(args, user_id)
        if name == "search_documents":
            return tool_search_documents(args, user_id)
        if name == "save_note":
            return tool_save_note(args, user_id)
        if name == "list_notes":
            return tool_list_notes(args, user_id)
        if name == "set_reminder":
            return tool_set_reminder(args, user_id)
        if name == "list_reminders":
            return tool_list_reminders(args, user_id)
        if name == "cancel_reminder":
            return tool_cancel_reminder(args, user_id)
        return f"erreur: outil inconnu '{name}'"
    except Exception as exc:  # noqa: BLE001
        return f"erreur: {exc}"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_datetime",
            "description": "Renvoie la date et l'heure actuelles. A appeler avant tout calcul de date.",
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
                "properties": {
                    "expression": {"type": "string", "description": "Ex: (1200*0.19)+45"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Recherche des informations actuelles sur le web et renvoie des titres, URL et extraits.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Mots-cles de recherche."},
                    "max_results": {"type": "integer", "description": "Nombre de resultats (1-10, defaut 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Recupere le texte complet d'une page web publique (http/https).",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "Liste les documents joints par l'utilisateur (PDF, texte, CSV, JSON...). "
                "A appeler des que l'utilisateur parle d'un document, d'un fichier ou d'une piece jointe."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Lit le contenu d'un document joint. Si le document est long, "
                "enchaines les appels en augmentant 'debut' jusqu'a 'suite_disponible' = false."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "document": {"type": "string", "description": "Nom ou identifiant du document."},
                    "debut": {"type": "integer", "description": "Position de depart en caracteres (defaut 0)."},
                    "longueur": {"type": "integer", "description": "Nombre de caracteres a lire (defaut 6000)."},
                },
                "required": ["document"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Cherche un mot ou une phrase dans les documents joints de l'utilisateur.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "description": "Nombre de documents examines (defaut 5)."},
                },
                "required": ["query"],
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
            "name": "set_reminder",
            "description": (
                "Programme un rappel pour l'utilisateur courant. "
                "Appelle d'abord get_datetime pour connaitre l'heure courante, "
                "puis donne 'when' en ISO 8601 AVEC decalage (ex: 2026-10-19T09:00:00+01:00) "
                "ou utilise 'in_minutes' pour un delai relatif."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Ce qu'il faut rappeler."},
                    "when": {"type": "string", "description": "Date/heure ISO 8601 avec decalage horaire."},
                    "in_minutes": {"type": "number", "description": "Delai relatif en minutes."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reminders",
            "description": "Liste les rappels a venir de l'utilisateur courant.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reminder",
            "description": "Annule un rappel a partir de son identifiant.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
                "required": ["id"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOL_SCHEMAS]


# ---------------------- Boucle d'agent (fonction calling) -----------------


def _tool_call_payload(tc):
    return {
        "id": tc.id,
        "type": "function",
        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
    }


def run_agent(messages, user_id, use_tools=True):
    """Boucle synchrone : le modele demande un outil, on l'execute, on recommence."""
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
                "tool_calls": [_tool_call_payload(tc) for tc in tool_calls],
            }
        )
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            result = execute_tool(tc.function.name, args, user_id)
            used_tools.append({"tool": tc.function.name, "arguments": args})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)[:6000]})

    msg, provider_name, model_name = call_llm(messages, tools=None)
    return msg.content or "", provider_name, model_name, used_tools


def agent_stream(messages, user_id, use_tools=True):
    """Variante streamee. Emet ('delta', ...), ('tool', ...), ('done', ...)."""
    active_tools = TOOL_SCHEMAS if (TOOLS_ENABLED and use_tools) else None
    used_tools = []
    parts = []
    provider_name = model_name = None

    for _ in range(MAX_TOOL_ITERATIONS):
        name, model, content, calls = yield from stream_once(messages, active_tools)
        provider_name, model_name = name, model
        if content:
            parts.append(content.strip())
        if not calls:
            break

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": calls[i]["id"] or f"call_{i}",
                        "type": "function",
                        "function": {"name": calls[i]["name"], "arguments": calls[i]["arguments"] or "{}"},
                    }
                    for i in sorted(calls)
                ],
            }
        )
        for i in sorted(calls):
            tool_name = calls[i]["name"]
            try:
                args = json.loads(calls[i]["arguments"] or "{}")
            except Exception:  # noqa: BLE001
                args = {}
            yield ("tool", {"tool": tool_name, "arguments": args})
            result = execute_tool(tool_name, args, user_id)
            used_tools.append({"tool": tool_name, "arguments": args})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": calls[i]["id"] or f"call_{i}",
                    "content": str(result)[:6000],
                }
            )
    else:
        # Budget d'outils epuise : on force une reponse finale sans outil.
        name, model, content, _calls = yield from stream_once(messages, None)
        provider_name, model_name = name, model
        if content:
            parts.append(content.strip())

    yield (
        "done",
        {
            "answer": "\n\n".join(p for p in parts if p),
            "provider": provider_name,
            "model": model_name,
            "tools_used": used_tools,
        },
    )


# ------------------------------ Endpoints ---------------------------------


class ChatRequest(BaseModel):
    prompt: str
    user_id: str = "default"
    session_id: str = "default"
    system: str = DEFAULT_SYSTEM
    use_tools: bool = True


class AckRequest(BaseModel):
    user_id: str
    ids: list[str] = []


_chat_ui_cache = None


@app.get("/ui", response_class=HTMLResponse)
def chat_ui():
    """Sert l'interface de chat (meme origine que l'API : aucune config)."""
    global _chat_ui_cache
    if _chat_ui_cache is None:
        try:
            with open(CHAT_UI_PATH, encoding="utf-8") as handle:
                _chat_ui_cache = handle.read()
        except OSError as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail=f"interface introuvable: {exc}") from exc
    return HTMLResponse(_chat_ui_cache)


@app.get("/")
def home(request: Request):
    authenticated = _token_ok(request) if AUTH_ENABLED else True
    public = {
        "status": "live",
        "app": APP_NAME,
        "ui": "/ui",
        "auth_required": AUTH_ENABLED,
    }
    if not authenticated:
        # Sans cle : aucune information sur la configuration interne.
        return public
    public.update(
        {
            "memory": "ready" if _index_ready else ("configured" if MONGO_URI else "off"),
            "storage_configured": bool(MONGO_URI),
            "tools_enabled": TOOLS_ENABLED,
            "tools": TOOL_NAMES,
            "search_backend": active_search_backend(),
            "search_configured": bool(SEARCH_API_KEY or SEARXNG_URL),
            "search_budget": search_budget(),
            "provider_order": [p for p in PROVIDER_ORDER if p in PROVIDERS],
            "models": {n: PROVIDERS[n]["model"] for n in PROVIDER_ORDER if n in PROVIDERS},
            "limits": {
                "documents_max": MAX_DOCS_PER_USER,
                "stockage_max_mo": round(MAX_DOCS_BYTES_PER_USER / (1024 * 1024), 1),
                "requetes_par_minute": RATE_LIMIT_PER_MIN,
                "messages_par_minute": RATE_LIMIT_CHAT_PER_MIN,
            },
        }
    )
    return public


@app.get("/auth/check")
def auth_check():
    """Route protegee : sert a l'interface pour valider la cle saisie."""
    return {"ok": True, "auth_required": AUTH_ENABLED}


@app.get("/tools")
def tools_list():
    return {
        "enabled": TOOLS_ENABLED,
        "search_backend": active_search_backend(),
        "search_budget": search_budget(),
        "tools": TOOL_SCHEMAS,
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


def _need_memory():
    col = get_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    return col


def _need_notes():
    col = get_notes_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    return col


def _need_reminders():
    col = get_reminders_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    return col


def _need_docs():
    col = get_docs_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    return col


def _need_sessions():
    col = get_sessions_collection()
    if col is None:
        raise HTTPException(
            status_code=503, detail={"message": "Memoire indisponible", "error": _mongo_error}
        )
    return col


@app.get("/history/{user_id}")
def history(user_id: str, limit: int = 10, session_id: str | None = None):
    col = _need_memory()
    query = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    docs = list(
        col.find(query, {"_id": 0}).sort("created_at", -1).limit(max(1, min(limit, 100)))
    )
    return {"user_id": user_id, "session_id": session_id, "count": len(docs), "messages": docs}


@app.get("/sessions/{user_id}")
def sessions(user_id: str, limit: int = 50):
    """Fils de discussion de l'utilisateur, du plus recent au plus ancien."""
    col = _need_memory()
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$sort": {"created_at": 1}},
        {
            "$group": {
                "_id": "$session_id",
                "messages": {"$sum": 1},
                "first_prompt": {"$first": "$prompt"},
                "last_prompt": {"$last": "$prompt"},
                "last_at": {"$max": "$created_at"},
            }
        },
        {"$sort": {"last_at": -1}},
        {"$limit": max(1, min(limit, 200))},
    ]
    rows = list(col.aggregate(pipeline))
    titles = {}
    meta = get_sessions_collection()
    if meta is not None:
        try:
            titles = {
                d.get("session_id"): (d.get("title") or "").strip()
                for d in meta.find(
                    {"user_id": user_id}, {"_id": 0, "session_id": 1, "title": 1}
                )
            }
        except Exception:  # noqa: BLE001
            titles = {}

    out = []
    for row in rows:
        sid = row.get("_id") or "default"
        custom = titles.get(sid) or ""
        out.append(
            {
                "session_id": sid,
                "messages": row.get("messages", 0),
                "titre": custom or (row.get("first_prompt") or row.get("last_prompt") or "")[:90],
                "titre_personnalise": custom or None,
                "last_prompt": (row.get("last_prompt") or "")[:90],
                "last_at": _iso(_as_utc(row.get("last_at"))),
            }
        )
    return {"user_id": user_id, "count": len(out), "sessions": out}


class RenameRequest(BaseModel):
    title: str


@app.patch("/sessions/{user_id}/{session_id}")
def rename_session(user_id: str, session_id: str, req: RenameRequest):
    """Donne un nom a un fil (titre vide = retour au titre automatique)."""
    col = _need_sessions()
    title = (req.title or "").strip()[:120]
    now = datetime.now(timezone.utc)
    if not title:
        col.delete_many({"user_id": user_id, "session_id": session_id})
        return {"user_id": user_id, "session_id": session_id, "titre": None}
    col.update_one(
        {"user_id": user_id, "session_id": session_id},
        {"$set": {"title": title, "updated_at": now}, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    return {"user_id": user_id, "session_id": session_id, "titre": title}


@app.delete("/sessions/{user_id}/{session_id}")
def delete_session(user_id: str, session_id: str):
    col = _need_memory()
    deleted = col.delete_many({"user_id": user_id, "session_id": session_id}).deleted_count
    meta = get_sessions_collection()
    if meta is not None:
        meta.delete_many({"user_id": user_id, "session_id": session_id})
    return {"user_id": user_id, "session_id": session_id, "deleted_messages": deleted}


@app.get("/search/{user_id}")
def search_history(user_id: str, q: str, limit: int = 20):
    """Recherche dans l'historique des messages de l'utilisateur."""
    col = _need_memory()
    needle = (q or "").strip()
    if len(needle) < 2:
        raise HTTPException(status_code=400, detail="requete trop courte (2 caracteres minimum)")
    pattern = re.compile(re.escape(needle[:120]), re.I)
    query = {"user_id": user_id, "$or": [{"prompt": pattern}, {"response": pattern}]}
    docs = list(
        col.find(query, {"_id": 0}).sort("created_at", -1).limit(max(1, min(limit, 50)))
    )
    results = []
    for doc in docs:
        prompt = doc.get("prompt") or ""
        response = doc.get("response") or ""
        haystack = prompt if pattern.search(prompt) else response
        match = pattern.search(haystack or "")
        start = max(0, (match.start() if match else 0) - 80)
        results.append(
            {
                "session_id": doc.get("session_id"),
                "extrait": (haystack[start:start + 220] or "").replace("\n", " ").strip(),
                "created_at": _iso(_as_utc(doc.get("created_at"))),
                "prompt": prompt[:160],
            }
        )
    return {"user_id": user_id, "query": needle, "count": len(results), "results": results}


@app.post("/upload/{user_id}")
async def upload_document(
    user_id: str,
    file: UploadFile = File(...),
    session_id: str = Form("default"),
):
    """Recoit un fichier (PDF, texte, CSV...), en extrait le texte et le range."""
    docs = _need_docs()
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="fichier vide")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"fichier trop volumineux (max {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo)",
        )

    # Plafond de stockage : le MongoDB gratuit est limite, on refuse avant
    # d'ecrire plutot que de saturer la base.
    stats = list(
        docs.aggregate(
            [
                {"$match": {"user_id": user_id}},
                {"$group": {"_id": None, "n": {"$sum": 1}, "octets": {"$sum": "$size"}}},
            ]
        )
    )
    count = int(stats[0]["n"]) if stats else 0
    used = int(stats[0].get("octets") or 0) if stats else 0
    if count >= MAX_DOCS_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"limite de {MAX_DOCS_PER_USER} documents atteinte : supprimez-en un avant d'en ajouter",
        )
    if used + len(raw) > MAX_DOCS_BYTES_PER_USER:
        raise HTTPException(
            status_code=429,
            detail=f"espace de stockage epuise ({used // (1024 * 1024)} Mo utilises sur "
            f"{MAX_DOCS_BYTES_PER_USER // (1024 * 1024)} Mo) : supprimez des documents",
        )
    try:
        text, kind = extract_document_text(file.filename or "document", raw)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    text = text.strip()
    if not text:
        raise HTTPException(
            status_code=422,
            detail="aucun texte exploitable dans ce fichier (PDF scanne sans couche texte ?)",
        )
    truncated = len(text) > DOC_TEXT_LIMIT
    text = text[:DOC_TEXT_LIMIT]
    doc = {
        "doc_id": uuid.uuid4().hex[:8],
        "user_id": user_id,
        "session_id": session_id,
        "name": (file.filename or "document")[:200],
        "kind": kind,
        "size": len(raw),
        "chars": len(text),
        "tronque": truncated,
        "text": text,
        "created_at": datetime.now(timezone.utc),
    }
    docs.insert_one(doc)
    return {
        **_doc_out(doc),
        "taille_octets": len(raw),
        "tronque": truncated,
        "apercu": text[:400],
        "message": "Document ajoute. L'agent peut le lire avec read_document.",
    }


@app.get("/documents/{user_id}")
def documents(user_id: str, limit: int = 50):
    docs = _need_docs()
    rows = list(
        docs.find({"user_id": user_id}, {"_id": 0, "text": 0})
        .sort("created_at", -1)
        .limit(max(1, min(limit, 200)))
    )
    return {"user_id": user_id, "count": len(rows), "documents": list(map(_doc_out, rows))}


@app.delete("/documents/{user_id}/{doc_id}")
def delete_document(user_id: str, doc_id: str):
    docs = _need_docs()
    deleted = docs.delete_many({"user_id": user_id, "doc_id": doc_id}).deleted_count
    return {"deleted": deleted}


@app.delete("/history/{user_id}")
def delete_history(user_id: str, session_id: str | None = None):
    col = _need_memory()
    query = {"user_id": user_id}
    if session_id:
        query["session_id"] = session_id
    deleted = col.delete_many(query).deleted_count

    meta = get_sessions_collection()
    sessions_deleted = meta.delete_many(query).deleted_count if meta is not None else 0

    # Notes, rappels et documents ne sont pas rattaches a un fil : on ne les
    # purge que si l'on efface tout l'utilisateur.
    notes_deleted = reminders_deleted = docs_deleted = 0
    if not session_id:
        notes = get_notes_collection()
        notes_deleted = notes.delete_many({"user_id": user_id}).deleted_count if notes is not None else 0
        reminders = get_reminders_collection()
        reminders_deleted = (
            reminders.delete_many({"user_id": user_id}).deleted_count if reminders is not None else 0
        )
        docs = get_docs_collection()
        docs_deleted = docs.delete_many({"user_id": user_id}).deleted_count if docs is not None else 0

    return {
        "user_id": user_id,
        "session_id": session_id,
        "deleted_messages": deleted,
        "deleted_notes": notes_deleted,
        "deleted_reminders": reminders_deleted,
        "deleted_documents": docs_deleted,
        "deleted_sessions": sessions_deleted,
    }


@app.get("/notes/{user_id}")
def notes(user_id: str, limit: int = 50):
    col = _need_notes()
    docs = list(
        col.find({"user_id": user_id}, {"_id": 0, "note": 1, "created_at": 1})
        .sort("created_at", -1)
        .limit(max(1, min(limit, 200)))
    )
    return {"user_id": user_id, "count": len(docs), "notes": docs}


@app.get("/reminders/{user_id}")
def reminders(user_id: str, limit: int = 50):
    col = _need_reminders()
    now = datetime.now(timezone.utc)
    docs = list(
        col.find({"user_id": user_id}).sort("due_at", 1).limit(max(1, min(limit, 200)))
    )
    return {"user_id": user_id, "count": len(docs), "reminders": [_reminder_out(d, now) for d in docs]}


@app.get("/reminders/{user_id}/due")
def reminders_due(user_id: str, limit: int = 20):
    """Rappels echus et pas encore signales : l'interface les affiche puis les acquitte."""
    col = _need_reminders()
    now = datetime.now(timezone.utc)
    docs = list(
        col.find({"user_id": user_id, "notified": False, "due_at": {"$lte": now}})
        .sort("due_at", 1)
        .limit(max(1, min(limit, 50)))
    )
    return {"user_id": user_id, "count": len(docs), "due": [_reminder_out(d, now) for d in docs]}


@app.post("/reminders/ack")
def reminders_ack(req: AckRequest):
    col = _need_reminders()
    result = col.update_many(
        {"user_id": req.user_id, "short_id": {"$in": req.ids}},
        {"$set": {"notified": True, "notified_at": datetime.now(timezone.utc)}},
    )
    return {"acked": result.modified_count}


@app.delete("/reminders/{user_id}/{reminder_id}")
def delete_reminder(user_id: str, reminder_id: str):
    col = _need_reminders()
    deleted = col.delete_many({"user_id": user_id, "short_id": reminder_id}).deleted_count
    return {"deleted": deleted}


# ------------------------- Chat (non streame / streame) -------------------


def _load_history(col, user_id, session_id):
    if col is None:
        return []
    try:
        docs = list(
            col.find({"user_id": user_id, "session_id": session_id}, {"_id": 0})
            .sort("created_at", -1)
            .limit(MEMORY_TURNS)
        )
        docs.reverse()
        return docs
    except Exception:  # noqa: BLE001 - la memoire ne doit pas casser le chat
        return []


def _build_messages(history_docs, req):
    messages = [{"role": "system", "content": req.system}]
    for doc in history_docs:
        messages.append({"role": "user", "content": doc.get("prompt", "")})
        messages.append({"role": "assistant", "content": doc.get("response", "")})
    messages.append({"role": "user", "content": req.prompt})
    return messages


def _store(col, req, answer, provider_name, model_name, used_tools):
    if col is None:
        return False
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
        return True
    except Exception:  # noqa: BLE001
        return False


@app.post("/chat")
def chat(req: ChatRequest):
    col = get_collection()
    history_docs = _load_history(col, req.user_id, req.session_id)
    messages = _build_messages(history_docs, req)

    answer, provider_name, model_name, used_tools = run_agent(
        messages, req.user_id, use_tools=req.use_tools
    )
    stored = _store(col, req, answer, provider_name, model_name, used_tools)

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


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Meme chose que /chat, mais en Server-Sent Events (reponse mot a mot)."""
    col = get_collection()
    history_docs = _load_history(col, req.user_id, req.session_id)
    messages = _build_messages(history_docs, req)

    def generate():
        provider_name = model_name = None
        used_tools = []
        answer = ""
        try:
            yield _sse(
                "meta",
                {
                    "memory_enabled": col is not None,
                    "remembered_turns": len(history_docs),
                    "tools_enabled": bool(TOOLS_ENABLED and req.use_tools),
                },
            )
            for event, data in agent_stream(messages, req.user_id, req.use_tools):
                if event == "done":
                    answer = data["answer"]
                    provider_name = data["provider"]
                    model_name = data["model"]
                    used_tools = data["tools_used"]
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": str(exc)[:500]})
            return

        stored = _store(col, req, answer, provider_name, model_name, used_tools)
        yield _sse(
            "end",
            {
                "stored": stored,
                "fallback_used": provider_name != (PROVIDER_ORDER[0] if PROVIDER_ORDER else None),
            },
        )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
