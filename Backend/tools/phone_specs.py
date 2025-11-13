"""
Phone Specs Tool
Searches phone database by model name and returns specifications.
Supports both PostgreSQL (with trigram similarity) and CSV (with fuzzy matching).
"""

import time
import os
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, "database", "mobile.csv")
_DEFAULT_DF = None
_EXCLUDED_FIELDS = {"price", "price_usd", "price_eur", "price_tnd"}


def _load_default_dataframe() -> Optional[pd.DataFrame]:
    """Lazy-load the fallback CSV dataset when no DataFrame is provided."""
    global _DEFAULT_DF
    if _DEFAULT_DF is not None:
        return _DEFAULT_DF

    if not os.path.exists(DEFAULT_DATASET_PATH):
        return None

    df = pd.read_csv(DEFAULT_DATASET_PATH)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    df["brand"] = df.get("brand", "").astype(str).fillna("").str.strip()
    df["model"] = df.get("model", "").astype(str).fillna("").str.strip()
    df["search_name"] = (df["brand"].fillna("") + " " + df["model"].fillna("")).str.strip().str.lower()
    df = df.reset_index(drop=True)
    _DEFAULT_DF = df
    return _DEFAULT_DF

# Import database configuration from parent module
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.pool import NullPool
    SQLALCHEMY_AVAILABLE = True
except Exception:
    SQLALCHEMY_AVAILABLE = False

from rapidfuzz import process, fuzz


class PhoneSpecsInput(ToolInput):
    """Input schema for phone specs search."""
    phone_name: str = Field(description="Phone model name to search for")
    top_k: int = Field(default=3, description="Number of top matches to return")
    min_score: int = Field(default=60, description="Minimum similarity score (0-100)")


def _serialize_value(v: Any) -> Any:
    """Serialize value to JSON-safe format."""
    if v is None:
        return None
    if isinstance(v, (str, bool)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
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
    """Sanitize specification row for JSON output."""
    clean = {}
    for k, val in (row or {}).items():
        if k.lower() in _EXCLUDED_FIELDS:
            continue
        clean[k] = _serialize_value(val)
    return clean


def phone_specs_tool(
    phone_name: str,
    top_k: int = 5,
    min_score: int = 60,
    df: Optional[pd.DataFrame] = None,
    db_engine = None,
    use_postgres: bool = False
) -> ToolOutput:
    """
    Search phone database for specifications (extracted from mcp_server.py).
    
    Args:
        phone_name: Phone model to search for
        top_k: Number of results to return
        min_score: Minimum similarity score (0-100)
        df: Optional DataFrame (CSV data)
        db_engine: Optional SQLAlchemy engine (PostgreSQL)
        use_postgres: Whether to try PostgreSQL first
        
    Returns:
        ToolOutput with phone specifications
    """
    start_time = time.time()
    
    try:
        if not phone_name or not isinstance(phone_name, str):
            return ToolOutput(
                success=False,
                error="phone_name must be a non-empty string"
            )
        
        query = str(phone_name).strip().lower()
        
        # PostgreSQL implementation
        if use_postgres and db_engine and SQLALCHEMY_AVAILABLE:
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
                        return ToolOutput(
                            success=False,
                            error=f"No matches found for '{phone_name}'",
                            metadata={"execution_time": round(time.time() - start_time, 3)}
                        )
                    
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
                        return ToolOutput(
                            success=False,
                            error=f"No matches above min_score={min_score} for '{phone_name}'",
                            data={"diag": {"matches": diag_matches}},
                            metadata={"execution_time": round(time.time() - start_time, 3)}
                        )
                    
                    return ToolOutput(
                        success=True,
                        data={
                            "results": results,
                            "diag": {"matches": diag_matches, "source": "postgresql"}
                        },
                        metadata={
                            "execution_time": round(time.time() - start_time, 3),
                            "source": "postgresql"
                        }
                    )
                    
            except Exception as db_exc:
                print(f"⚠️  PostgreSQL query failed: {db_exc}, falling back to CSV")
                # Fall through to CSV implementation
        
        # CSV implementation (fallback or default)
        if df is None:
            df = _load_default_dataframe()

        if df is None:
            return ToolOutput(
                success=False,
                error="No data source available (PostgreSQL failed and CSV not loaded)"
            )
        
        if "search_name" not in df.columns:
            return ToolOutput(
                success=False,
                error="Dataset missing 'search_name' column"
            )

        choices = df["search_name"].astype(str).tolist()
        matches = process.extract(query, choices, scorer=fuzz.token_set_ratio, limit=top_k)

        if not matches:
            return ToolOutput(
                success=False,
                error=f"No matches found for '{phone_name}'",
                metadata={"execution_time": round(time.time() - start_time, 3)}
            )

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
            return ToolOutput(
                success=False,
                error=f"No matches above min_score={min_score} for '{phone_name}'",
                data={"diag": {"matches": diag_matches}},
                metadata={"execution_time": round(time.time() - start_time, 3)}
            )

        execution_time = time.time() - start_time
        
        return ToolOutput(
            success=True,
            data={
                "results": results,
                "diag": {"matches": diag_matches, "source": "csv"}
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "source": "csv"
            }
        )
    
    except Exception as e:
        return ToolOutput(
            success=False,
            error=f"Phone specs search failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
PHONE_SPECS_SCHEMA = {
    "name": "phone_specs",
    "description": "Search phone database by model name and return detailed specifications using fuzzy matching",
    "input_schema": {
        "type": "object",
        "properties": {
            "phone_name": {
                "type": "string",
                "description": "Phone model name to search for"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of top matches to return",
                "default": 3
            },
            "min_score": {
                "type": "integer",
                "description": "Minimum similarity score (0-100)",
                "default": 60
            }
        },
        "required": ["phone_name"]
    },
    "version": "1.0.0"
}
