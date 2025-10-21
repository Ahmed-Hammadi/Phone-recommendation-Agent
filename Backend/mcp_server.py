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
from fastapi import FastAPI, Request

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

# ---------- Config ----------
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "database", "mobile.csv")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours default TTL for reddit queries

# ---------- Load dataset ----------
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset file not found at {DATASET_PATH}")

df = pd.read_csv(DATASET_PATH)
df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

# Normalize brand/model and create search_name
df["brand"] = df.get("brand", "").astype(str).fillna("").str.strip()
df["model"] = df.get("model", "").astype(str).fillna("").str.strip()
df["search_name"] = (df["brand"].fillna("") + " " + df["model"].fillna("")).str.strip().str.lower()
df = df.reset_index(drop=True)

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

        # Search submissions in each subreddit, take titles/selftext + top comments
        for sr in subreddits:
            try:
                subreddit = reddit.subreddit(sr)
                # Using subreddit.search (sort by new). Increase limit if needed.
                for submission in subreddit.search(phone_name, limit=50, sort="new"):
                    if submission is None:
                        continue
                    # title + selftext
                    title = getattr(submission, "title", "") or ""
                    selftext = getattr(submission, "selftext", "") or ""
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
                            if body_clean not in seen:
                                collected_texts.append(body_clean)
                                seen.add(body_clean)
                    except Exception:
                        # comments sometimes fail for a single submission — ignore and continue
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
        if "search_name" not in df.columns:
            return {"error": {"message": "Dataset missing 'search_name' column"}}

        query = str(phone_name).strip().lower()
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

        return {"results": results, "diag": {"matches": diag_matches}}
    except Exception as exc:
        return {"error": {"message": f"phone_specs_tool exception: {exc}", "diag": {"type": type(exc).__name__}}}

# ---------- MCP tool registry & endpoint ----------
MCP_TOOLS = {
    "phone_specs": phone_specs_tool,
    "web_scraper": reddit_scraper,  # PRAW-only reddit tool
}

@app.post("/mcp_tool")
async def mcp_tool(req: Request):
    try:
        body = await req.json()
    except Exception:
        return safe_json_error("Invalid JSON body")
    if not isinstance(body, dict):
        return safe_json_error("Request JSON must be an object")
    tool_name = body.get("tool_name")
    if not tool_name or not isinstance(tool_name, str):
        return safe_json_error("tool_name (string) is required")
    if "kwargs" in body and isinstance(body.get("kwargs"), dict):
        kwargs = body["kwargs"].copy()
    else:
        kwargs = body.copy()
        kwargs.pop("tool_name", None)
    tool = MCP_TOOLS.get(tool_name)
    if tool is None:
        return safe_json_error(f"Tool '{tool_name}' not found")
    try:
        result = tool(**kwargs)
        return result
    except TypeError as e:
        return safe_json_error(f"Tool argument error: {e}")
    except Exception as e:
        return safe_json_error(f"Internal tool error: {e}")
