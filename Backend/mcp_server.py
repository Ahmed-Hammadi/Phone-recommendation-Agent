# Backend/mcp_server.py
import os
import time
import math
import json
import hashlib
import requests
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from rapidfuzz import process, fuzz
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware

# --- New Imports ---
# When running as script, use absolute imports
import sys
from pathlib import Path
if __name__ == "__main__":
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

from Backend.llm_agent import generate_recommendation
from Backend.orchestrator import ToolOrchestrator

# Import new MCP tools
from Backend.tools import (
    phone_specs_tool,
    web_scraper_tool,
    specs_analyzer_tool,
    sentiment_analyzer_tool,
    price_extractor_tool,
    alternative_recommender_tool,
    spec_validator_tool,
    TOOLS_REGISTRY
)
# --- End New Imports ---

# PostgreSQL support
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

# PRAW (official Reddit API) - required for reddit tool in this repo
try:
    import praw
    PRAW_AVAILABLE = True
except Exception:
    PRAW_AVAILABLE = False

app = FastAPI(title="MCP Server (RAG + PRAW-only Reddit tool)")

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
allowed_origins = [frontend_origin]
if frontend_origin != "*":
    allowed_origins.append("http://127.0.0.1:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if frontend_origin != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "database", "mobile.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours default TTL for reddit queries

# ---------- Database Configuration ----------
DB_HOST = os.getenv("POSTGRES_HOST")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB")
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")

USE_POSTGRES = all([DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, SQLALCHEMY_AVAILABLE])

if USE_POSTGRES:
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    try:
        db_engine = create_engine(DATABASE_URL, poolclass=NullPool)
        # Test connection
        with db_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print(f"✅ PostgreSQL connected: {DB_NAME}@{DB_HOST}")
    except Exception as e:
        print(f"⚠️  PostgreSQL connection failed: {e}")
        print(f"📂 Falling back to CSV")
        USE_POSTGRES = False
        db_engine = None
else:
    db_engine = None
    if not SQLALCHEMY_AVAILABLE:
        print(f"📂 SQLAlchemy not available, using CSV")
    else:
        print(f"📂 PostgreSQL not configured, using CSV")

# ---------- Load CSV dataset (fallback) ----------
df = None
if not USE_POSTGRES:
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    
    # Normalize brand/model and create search_name
    df["brand"] = df.get("brand", "").astype(str).fillna("").str.strip()
    df["model"] = df.get("model", "").astype(str).fillna("").str.strip()
    df["search_name"] = (df["brand"].fillna("") + " " + df["model"].fillna("")).str.strip().str.lower()
    df = df.reset_index(drop=True)
    print("📊 Loaded", len(df), "phones from CSV")

# Initialize Tool Orchestrator
orchestrator = ToolOrchestrator(df=df, db_engine=db_engine, use_postgres=USE_POSTGRES)
print("🤖 Tool Orchestrator initialized")

# ---------- Utility functions ----------

# ---------- Helpers: JSON-safe ----------
def _serialize_value(v: Any) -> Any:
    import math, numpy as _np
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    if isinstance(v, (_np.integer,)):
        return int(v)
    if isinstance(v, (_np.floating,)):
        f = float(v)
        return f if math.isfinite(f) else None
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, int):
        return v
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return str(v)
    except Exception:
        return None

def _sanitize_spec_row(row: dict) -> dict:
    clean = {}
    for k, val in (row or {}).items():
        clean[k] = _serialize_value(val)
    return clean

def safe_json_error(msg: str) -> Dict[str, Any]:
    return {"error": {"message": msg}}


def build_tools_metadata() -> Dict[str, Any]:
    """Return metadata for all registered MCP tools."""
    from Backend.tools.phone_specs import PHONE_SPECS_SCHEMA
    from Backend.tools.web_scraper import WEB_SCRAPER_SCHEMA
    from Backend.tools.specs_analyzer import SPECS_ANALYZER_SCHEMA
    from Backend.tools.sentiment_analyzer import SENTIMENT_ANALYZER_SCHEMA
    from Backend.tools.price_extractor import PRICE_EXTRACTOR_SCHEMA
    from Backend.tools.alternative_recommender import ALTERNATIVE_RECOMMENDER_SCHEMA
    from Backend.tools.spec_validator import SPEC_VALIDATOR_SCHEMA

    tools_metadata = [
        PHONE_SPECS_SCHEMA,
        WEB_SCRAPER_SCHEMA,
        SPECS_ANALYZER_SCHEMA,
        SENTIMENT_ANALYZER_SCHEMA,
        PRICE_EXTRACTOR_SCHEMA,
        ALTERNATIVE_RECOMMENDER_SCHEMA,
        SPEC_VALIDATOR_SCHEMA,
    ]

    return {
        "tools": tools_metadata,
        "total_count": len(tools_metadata),
        "server": "MCP Phone Recommendation Server",
        "version": "2.0.0",
    }


async def execute_tool(tool_name: str, kwargs: Optional[Dict[str, Any]]) -> Any:
    """Execute a registered tool within the thread pool."""
    tool = MCP_TOOLS.get(tool_name)
    if tool is None:
        raise KeyError(tool_name)

    if kwargs is None:
        kwargs = {}

    if not isinstance(kwargs, dict):
        raise TypeError("Tool kwargs must be a dictionary")

    return await run_in_threadpool(tool, **kwargs)


async def run_recommendation_pipeline(phone_name: str, top_k: int, min_score: int) -> Dict[str, Any]:
    """Run the full recommendation pipeline in a background thread."""
    recommendation = await run_in_threadpool(
        generate_recommendation,
        phone_name=phone_name,
        top_k=top_k,
        min_score=min_score,
        structured=True,
    )

    return {"recommendation": recommendation}


async def run_agent_chat_pipeline(query: str, include_reasoning: bool = False, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Execute the agent chat workflow and format the response."""
    result = await run_in_threadpool(
        orchestrator.process_query,
        query=query,
        session_id=session_id,
    )

    response: Dict[str, Any] = {
        "query": result["query"],
        "response": result["response"],
        "total_time": result["total_time"],
    }

    if "session_id" in result:
        response["session_id"] = result["session_id"]

    if include_reasoning:
        response["reasoning"] = {
            "intent": result["analysis"]["intent"],
            "phone_detected": result["analysis"].get("phone_name"),
            "requirements_detected": result["analysis"].get("requirements"),
            "tools_selected": result["analysis"].get("selected_tools", []),
            "tools_executed": len(result["execution_results"].get("tool_executions", [])),
            "tool_details": result["execution_results"].get("tool_executions", []),
            "matched_phone_name": result["execution_results"].get("matched_phone_name"),
            "context_used": bool(result["analysis"].get("phone_from_context")),
        }

    return response


class JsonRpcError(Exception):
    """Custom exception to bubble JSON-RPC compliant errors."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


def jsonrpc_success(rpc_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": rpc_id}


def jsonrpc_error(rpc_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    error_payload = {"code": code, "message": message}
    if data is not None:
        error_payload["data"] = data
    return {"jsonrpc": "2.0", "error": error_payload, "id": rpc_id}


async def dispatch_rpc_method(method: str, params: Any) -> Any:
    """Dispatch supported JSON-RPC methods."""
    if method == "mcp.list_tools":
        if params not in (None, {}):
            raise JsonRpcError(-32602, "Invalid params", "This method does not accept parameters")
        return build_tools_metadata()

    if method == "mcp.call_tool":
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise JsonRpcError(-32602, "Invalid params", "Expected params to be an object")

        tool_name = params.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise JsonRpcError(-32602, "Invalid params", "tool_name (string) is required")

        kwargs = params.get("kwargs", {})
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise JsonRpcError(-32602, "Invalid params", "kwargs must be an object")

        try:
            result = await execute_tool(tool_name.strip(), kwargs)
        except KeyError:
            raise JsonRpcError(-32601, f"Tool '{tool_name}' not found")
        except TypeError as exc:
            raise JsonRpcError(-32602, "Invalid params", str(exc))
        except Exception as exc:
            raise JsonRpcError(-32603, "Internal error", str(exc))

        return jsonable_encoder(result)

    if method == "agent.recommend":
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise JsonRpcError(-32602, "Invalid params", "Expected params to be an object")

        phone_name = params.get("phone_name")
        if not isinstance(phone_name, str) or not phone_name.strip():
            raise JsonRpcError(-32602, "Invalid params", "phone_name (string) is required")

        top_k = params.get("top_k", 3)
        min_score = params.get("min_score", 60)

        try:
            top_k_int = int(top_k)
            min_score_int = int(min_score)
        except Exception:
            raise JsonRpcError(-32602, "Invalid params", "top_k and min_score must be integers")

        try:
            result = await run_recommendation_pipeline(phone_name.strip(), top_k_int, min_score_int)
        except Exception as exc:
            raise JsonRpcError(-32603, "Internal error", str(exc))

        return jsonable_encoder(result)

    if method == "agent.chat":
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise JsonRpcError(-32602, "Invalid params", "Expected params to be an object")

        query = params.get("query")
        if not isinstance(query, str) or not query.strip():
            raise JsonRpcError(-32602, "Invalid params", "query (string) is required")

        include_reasoning = bool(params.get("include_reasoning", False))
        session_id = params.get("session_id")
        if session_id is not None and not isinstance(session_id, str):
            raise JsonRpcError(-32602, "Invalid params", "session_id must be a string when provided")

        try:
            result = await run_agent_chat_pipeline(
                query.strip(),
                include_reasoning=include_reasoning,
                session_id=session_id.strip() if isinstance(session_id, str) else None,
            )
        except Exception as exc:
            raise JsonRpcError(-32603, "Internal error", str(exc))

        return jsonable_encoder(result)

    if method in {"system.ping", "rpc.ping"}:
        return {"status": "ok", "time": time.time()}

    raise JsonRpcError(-32601, f"Method '{method}' not found")


async def handle_rpc_call(call: Any) -> Optional[Dict[str, Any]]:
    """Process a single JSON-RPC call or notification."""
    if not isinstance(call, dict):
        return jsonrpc_error(None, -32600, "Invalid Request")

    rpc_id = call.get("id") if "id" in call else None
    version = call.get("jsonrpc")
    method = call.get("method")
    params = call.get("params", {})

    if version != "2.0":
        return jsonrpc_error(rpc_id if rpc_id is not None else None, -32600, "Invalid Request", "jsonrpc must be '2.0'")

    if not isinstance(method, str):
        return jsonrpc_error(rpc_id if rpc_id is not None else None, -32600, "Invalid Request", "method must be a string")

    try:
        result = await dispatch_rpc_method(method, params)
    except JsonRpcError as exc:
        response_id = rpc_id if rpc_id is not None else None
        return jsonrpc_error(response_id, exc.code, exc.message, exc.data)
    except Exception as exc:  # Fallback internal error
        response_id = rpc_id if rpc_id is not None else None
        return jsonrpc_error(response_id, -32603, "Internal error", str(exc))

    if rpc_id is None:
        # Notification => no response
        return None

    return jsonrpc_success(rpc_id, result)


def _jsonrpc_error_response(exc: JsonRpcError) -> Dict[str, Any]:
    """Convert a JsonRpcError into the REST error payload shape used by the legacy endpoints."""
    message = f"[JSON-RPC {exc.code}] {exc.message}"
    if exc.data is not None:
        message = f"{message}: {exc.data}"
    return {"error": {"message": message, "code": exc.code, "data": exc.data}}

# ---------- Simple disk cache (JSON) ----------
def _cache_key_for_query(prefix: str, params: dict) -> str:
    key_src = prefix + json.dumps(params, sort_keys=True, ensure_ascii=False)
    key_hash = hashlib.sha256(key_src.encode("utf-8")).hexdigest()
    return key_hash

def _cache_get(prefix: str, params: dict, ttl_seconds: int = CACHE_TTL_SECONDS) -> Optional[dict]:
    key = _cache_key_for_query(prefix, params)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(path):
        return None
    try:
        st = os.stat(path)
        age = time.time() - st.st_mtime
        if age > ttl_seconds:
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _cache_set(prefix: str, params: dict, payload: dict):
    key = _cache_key_for_query(prefix, params)
    path = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
    except Exception:
        pass

# ---------- PRAW helper (required) ----------
def _get_praw_instance():
    if not PRAW_AVAILABLE:
        raise RuntimeError("PRAW package not installed. Run `pip install praw`.")
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "phone-recommendation-agent/0.1")
    if not client_id or not client_secret:
        raise RuntimeError("Reddit credentials missing. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.")
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, check_for_async=False)

# ---------- reddit_scraper using PRAW only ----------
def reddit_scraper(
    phone_name: str,
    max_results: int = 20,
    subreddits: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_ttl: int = CACHE_TTL_SECONDS,
) -> Dict[str, Any]:
    """
    Reddit fetcher using PRAW (official Reddit API). This function:
      - requires PRAW & Reddit credentials,
      - searches configured subreddits for submissions matching the query,
      - returns text from top submissions and their top comments (up to max_results).
    """
    try:
        if not phone_name or not isinstance(phone_name, str):
            return safe_json_error("phone_name must be a non-empty string")

        if not PRAW_AVAILABLE:
            return safe_json_error("PRAW package not installed. Run `pip install praw`.")

        subreddits = subreddits or ["gadgets", "android", "iphone", "smartphones", "all"]

        cache_params = {"phone_name": phone_name, "max_results": max_results, "subreddits": subreddits}
        if use_cache:
            cached = _cache_get("reddit_praw", cache_params, ttl_seconds=cache_ttl)
            if cached:
                return cached

        try:
            reddit = _get_praw_instance()
        except Exception as e:
            return safe_json_error(str(e))

        collected_texts: List[str] = []
        seen = set()
        
        # Extract model number for filtering (e.g., "23" from "Samsung Galaxy S23 Ultra")
        phone_lower = phone_name.lower()
        # Try to extract key identifying terms (model numbers, specific names)
        import re
        
        # Extract the primary model number (most specific)
        model_numbers = re.findall(r'\b\d+\b', phone_lower)
        primary_model = model_numbers[0] if model_numbers else None
        
        # Extract other identifiers
        model_tokens = set(re.findall(r'\bpro\b|\bmax\b|\bmini\b|\bplus\b|\bultra\b|\bse\b', phone_lower))

        # Search submissions in each subreddit, take titles/selftext + top comments
        for sr in subreddits:
            try:
                subreddit = reddit.subreddit(sr)
                # Using subreddit.search (sort by relevance first, then new). Increase limit if needed.
                for submission in subreddit.search(phone_name, limit=50, sort="relevance"):
                    if submission is None:
                        continue
                    
                    # Filter: Skip if title/selftext mention other model numbers that conflict
                    title = getattr(submission, "title", "") or ""
                    selftext = getattr(submission, "selftext", "") or ""
                    combined_text = (title + " " + selftext).lower()

                    if primary_model:
                        found_numbers = re.findall(r'\b\d+\b', combined_text)
                        # If post has model numbers and the primary doesn't match, skip
                        if found_numbers and primary_model not in found_numbers:
                            # Allow if it's just mentioning in comparison context
                            if not any(word in combined_text for word in ['upgrade from', 'coming from', 'switched from', 'vs', 'compared to']):
                                continue
                    
                    if any(word in combined_text for word in ['rumor', 'leak', 'upcoming', 'launch', 'release date', 'announcement', 'will have', 'expected to', 'will be released', 'coming soon']):
                        continue

                    # Collect title + selftext
                    for text in (title, selftext):
                        text_clean = text.strip()
                        if len(text_clean) >= 40 and text_clean not in seen:
                            collected_texts.append(text_clean)
                            seen.add(text_clean)
                            if len(collected_texts) >= max_results:
                                break
                    if len(collected_texts) >= max_results:
                        break

                    # collect top-level comments
                    try:
                        submission.comments.replace_more(limit=0)
                        for c in submission.comments.list():
                            if len(collected_texts) >= max_results:
                                break
                            body = getattr(c, "body", None)
                            if not body or not isinstance(body, str):
                                continue
                            body_clean = body.strip()
                            if len(body_clean) < 40:
                                continue
                            
                            # Filter out rumor/speculation comments and wrong model numbers
                            body_lower = body_clean.lower()
                            if any(word in body_lower for word in ['rumor', 'leak', 'upcoming', 'will be', 'expected to', 'might have', 'could have', 'coming soon', 'will be released']):
                                continue

                            if primary_model:
                                comment_numbers = re.findall(r'\b\d+\b', body_lower)
                                if comment_numbers and primary_model not in comment_numbers:
                                    # Allow comparison contexts
                                    if not any(word in body_lower for word in ['upgrade from', 'coming from', 'switched from', 'vs', 'compared to']):
                                        continue
                            
                            if body_clean not in seen:
                                collected_texts.append(body_clean)
                                seen.add(body_clean)
                    except Exception:
                        pass

                if len(collected_texts) >= max_results:
                    break
            except Exception:
                # ignore subreddit-specific errors and continue
                continue

        # final trim
        final_comments = collected_texts[:max_results]

        result_payload = {"reddit_comments": final_comments}
        result_payload["diag"] = {
            "method": "praw_only",
            "subreddits_searched": subreddits,
            "returned_count": len(final_comments),
        }

        if use_cache:
            _cache_set("reddit_praw", cache_params, result_payload)

        return result_payload

    except Exception as e:
        return safe_json_error(f"reddit_scraper exception: {e}")

# ---------- phone_specs_tool ----------
def phone_specs_tool(phone_name: str, top_k: int = 5, min_score: int = 60) -> Dict[str, Any]:
    try:
        if not phone_name or not isinstance(phone_name, str):
            return {"error": {"message": "phone_name must be a non-empty string"}}
        
        query = str(phone_name).strip().lower()
        
        # PostgreSQL implementation
        if USE_POSTGRES and db_engine:
            try:
                with db_engine.connect() as conn:
                    # Use PostgreSQL trigram similarity for fuzzy matching
                    sql = text("""
                        SELECT 
                            *,
                            similarity(search_name, :query) as score
                        FROM phones
                        WHERE search_name % :query
                        ORDER BY score DESC
                        LIMIT :limit
                    """)
                    
                    result = conn.execute(sql, {"query": query, "limit": top_k * 2})
                    rows = result.fetchall()
                    columns = result.keys()
                    
                    if not rows:
                        # Fallback to LIKE search if trigram returns nothing
                        sql_fallback = text("""
                            SELECT *
                            FROM phones
                            WHERE search_name ILIKE :pattern
                            LIMIT :limit
                        """)
                        result = conn.execute(sql_fallback, {"pattern": f"%{query}%", "limit": top_k})
                        rows = result.fetchall()
                        columns = result.keys()
                    
                    if not rows:
                        return {"error": {"message": f"No matches found for '{phone_name}'"}}
                    
                    results = []
                    diag_matches = []
                    
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        # Convert similarity score to percentage (0-1 -> 0-100)
                        score = int(row_dict.get('score', 0) * 100) if 'score' in row_dict else 100
                        
                        diag_matches.append({
                            "match_name": row_dict.get("search_name", ""),
                            "score": score,
                            "idx": row_dict.get("id", -1)
                        })
                        
                        if score < min_score:
                            continue
                        
                        results.append({
                            "brand": _serialize_value(row_dict.get("brand")),
                            "model": _serialize_value(row_dict.get("model")),
                            "search_name": row_dict.get("search_name", ""),
                            "score": score,
                            "specs": _sanitize_spec_row(row_dict)
                        })
                    
                    if not results:
                        return {"error": {"message": f"No matches above min_score={min_score} for '{phone_name}'", "diag": {"matches": diag_matches}}}
                    
                    return {"results": results, "diag": {"matches": diag_matches, "source": "postgresql"}}
                    
            except Exception as db_exc:
                print(f"⚠️  PostgreSQL query failed: {db_exc}, falling back to CSV")
                # Fall through to CSV implementation
        
        # CSV implementation (fallback or default)
        if df is None:
            return {"error": {"message": "No data source available (PostgreSQL failed and CSV not loaded)"}}
        
        if "search_name" not in df.columns:
            return {"error": {"message": "Dataset missing 'search_name' column"}}

        choices = df["search_name"].astype(str).tolist()
        matches = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=top_k)

        if not matches:
            return {"error": {"message": f"No matches found for '{phone_name}'"}}

        results = []
        diag_matches = []
        for match_name, score, idx in matches:
            diag_matches.append({"match_name": match_name, "score": int(score), "idx": int(idx)})
            if int(score) < int(min_score):
                continue
            try:
                row = df.iloc[idx].to_dict()
            except Exception:
                alt = df[df["search_name"] == match_name]
                row = alt.iloc[0].to_dict() if not alt.empty else {}
            results.append({
                "brand": _serialize_value(row.get("brand")),
                "model": _serialize_value(row.get("model")),
                "search_name": match_name,
                "score": int(score),
                "specs": _sanitize_spec_row(row)
            })

        if not results:
            return {"error": {"message": f"No matches above min_score={min_score} for '{phone_name}'", "diag": {"matches": diag_matches}}}

        return {"results": results, "diag": {"matches": diag_matches, "source": "csv"}}
    except Exception as exc:
        return {"error": {"message": f"phone_specs_tool exception: {exc}", "diag": {"type": type(exc).__name__}}}

# ---------- MCP tool registry & endpoint ----------
MCP_TOOLS = {
    # Wrappers for original tools that pass database context
    "phone_specs": lambda **kwargs: phone_specs_tool(**kwargs, df=df, db_engine=db_engine, use_postgres=USE_POSTGRES),
    "web_scraper": web_scraper_tool,
    
    # New specialized tools
    "specs_analyzer": specs_analyzer_tool,
    "sentiment_analyzer": sentiment_analyzer_tool,
    "price_extractor": price_extractor_tool,
    "alternative_recommender": alternative_recommender_tool,
    "spec_validator": spec_validator_tool,
}


# --- NEW: Pydantic models for request validation ---
class McpToolRequest(BaseModel):
    tool_name: str
    kwargs: Dict[str, Any] = {}

class RecommendationRequest(BaseModel):
    phone_name: str
    top_k: int = 3
    min_score: int = 60

class AgentQueryRequest(BaseModel):
    query: str
    include_reasoning: bool = False
# --- END Pydantic models ---


@app.post("/mcp_tool")
async def mcp_tool(payload: McpToolRequest): # Use Pydantic model
    """
    Run a specified MCP tool in a non-blocking thread pool.
    """
    try:
        params = {"tool_name": payload.tool_name, "kwargs": payload.kwargs or {}}
        return await dispatch_rpc_method("mcp.call_tool", params)
    except JsonRpcError as exc:
        return _jsonrpc_error_response(exc)
    except Exception as exc:
        return safe_json_error(f"Internal tool error: {exc}")


# --- NEW: Endpoint for Streamlit App ---
@app.post("/recommend")
async def recommend(payload: RecommendationRequest):
    """
    New endpoint to run the full recommendation pipeline.
    This is what Streamlit will call.
    """
    try:
        params = {
            "phone_name": payload.phone_name,
            "top_k": payload.top_k,
            "min_score": payload.min_score,
        }
        result = await dispatch_rpc_method("agent.recommend", params)
        return result
    except JsonRpcError as exc:
        return _jsonrpc_error_response(exc)
    except Exception as exc:
        return safe_json_error(f"Recommendation pipeline failed: {exc}")


@app.get("/mcp/tools")
async def list_tools():
    """
    MCP tool discovery endpoint.
    Returns metadata and schemas for all available tools.
    """
    # Import schemas from tool modules
    from .tools.phone_specs import PHONE_SPECS_SCHEMA
    from .tools.web_scraper import WEB_SCRAPER_SCHEMA
    from .tools.specs_analyzer import SPECS_ANALYZER_SCHEMA
    from .tools.sentiment_analyzer import SENTIMENT_ANALYZER_SCHEMA
    from .tools.price_extractor import PRICE_EXTRACTOR_SCHEMA
    from .tools.alternative_recommender import ALTERNATIVE_RECOMMENDER_SCHEMA
    from .tools.spec_validator import SPEC_VALIDATOR_SCHEMA
    
    tools_metadata = [
        PHONE_SPECS_SCHEMA,
        WEB_SCRAPER_SCHEMA,
        SPECS_ANALYZER_SCHEMA,
        SENTIMENT_ANALYZER_SCHEMA,
    PRICE_EXTRACTOR_SCHEMA,
        ALTERNATIVE_RECOMMENDER_SCHEMA,
        SPEC_VALIDATOR_SCHEMA,
    ]
    
    return {
        "tools": tools_metadata,
        "total_count": len(tools_metadata),
        "server": "MCP Phone Recommendation Server",
        "version": "2.0.0"
    }


@app.post("/agent/chat")
async def agent_chat(payload: AgentQueryRequest):
    """
    Intelligent agent endpoint that analyzes queries and dynamically selects tools.
    This is the main agentic interface.
    """
    try:
        params = {"query": payload.query, "include_reasoning": payload.include_reasoning}
        return await dispatch_rpc_method("agent.chat", params)
    except JsonRpcError as exc:
        return _jsonrpc_error_response(exc)
    except Exception as exc:
        return safe_json_error(f"Agent query failed: {exc}")


# ---------- JSON-RPC 2.0 endpoint ----------
@app.post("/rpc")
async def json_rpc_endpoint(request: Request):
    """
    Generic JSON-RPC 2.0 HTTP endpoint. Accepts a single request or a batch array.
    Reuses existing `handle_rpc_call` dispatch logic and returns JSON-RPC responses.
    """
    try:
        payload = await request.json()
    except Exception:
        # Parse error
        return JSONResponse(content=jsonrpc_error(None, -32700, "Parse error"), status_code=400)

    # Batch call
    if isinstance(payload, list):
        responses = []
        for call in payload:
            resp = await handle_rpc_call(call)
            if resp is not None:
                responses.append(resp)

        # Per JSON-RPC, empty response for-only notifications -> return 204 No Content
        if not responses:
            return Response(status_code=204)

        return JSONResponse(content=responses)

    # Single call
    if isinstance(payload, dict):
        resp = await handle_rpc_call(payload)
        if resp is None:
            return Response(status_code=204)  # notification
        return JSONResponse(content=resp)

    # Invalid request
    return JSONResponse(content=jsonrpc_error(None, -32600, "Invalid Request"), status_code=400)

# ========== Server Startup ==========
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting MCP Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

