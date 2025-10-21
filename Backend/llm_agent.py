# Backend/llm_agent.py

import os
import json
from typing import List, Dict, Any
from .mcp_client import call_mcp_tool
from dotenv import load_dotenv
# Mistral SDK
try:
    from mistralai import Mistral
    MISTRAL_SDK_AVAILABLE = True
except Exception:
    MISTRAL_SDK_AVAILABLE = False
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")

def _parse_mistral_response(res: Any) -> str:
    """
    Robust parser for possible mistral SDK responses.
    """
    try:
        # dict-like response with 'choices'
        if isinstance(res, dict):
            # common pattern: res['choices'][0]['message']['content']
            choices = res.get("choices") or res.get("outputs") or res.get("result")
            if choices and isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                if isinstance(first, dict):
                    # typical structure
                    message = first.get("message") or first.get("text") or first.get("content") or first.get("output")
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        if isinstance(content, str):
                            return content
                        if isinstance(content, list):
                            # join text fragments
                            parts = []
                            for c in content:
                                if isinstance(c, dict):
                                    parts.append(c.get("text") or c.get("content") or "")
                                elif isinstance(c, str):
                                    parts.append(c)
                            return " ".join([p for p in parts if p])
                    if isinstance(message, str):
                        return message
                    # fallback: maybe 'text' field
                    if "text" in first:
                        return first["text"]
            # fallback to any 'output' or 'data'
            if "output" in res:
                out = res["output"]
                if isinstance(out, (str, int, float)):
                    return str(out)
                if isinstance(out, list):
                    return " ".join(map(str, out))
            # last resort
            return json.dumps(res, ensure_ascii=False)
        # object-like response (SDK objects)
        if hasattr(res, "choices"):
            choices = getattr(res, "choices")
            if choices and len(choices) > 0:
                first = choices[0]
                # try typical attributes
                if hasattr(first, "message") and getattr(first.message, "content", None):
                    return getattr(first.message, "content")
                if hasattr(first, "text"):
                    return getattr(first, "text")
            return str(res)
        return str(res)
    except Exception as exc:
        return f"Error parsing LLM response: {exc}"

def _heuristic_summary_from_specs_and_reviews(specs: Dict[str, Any], reviews: List[str]) -> str:
    """
    Simple fallback summarizer (used when no Mistral key) — quick pros/cons heuristics.
    """
    pros = []
    cons = []
    # Heuristic rules
    battery = specs.get("nominal_battery_capacity") or specs.get("nominal_battery_capacity")
    try:
        if battery and float(battery) >= 4000:
            pros.append("Large battery capacity (>= 4000 mAh).")
        elif battery and float(battery) < 3000:
            cons.append("Small battery capacity (< 3000 mAh).")
    except Exception:
        pass

    ram = specs.get("ram_capacity_(converted)") or specs.get("memory_capacity") or specs.get("ram_capacity")
    if ram:
        try:
            # try to parse numeric from strings like "4 GiB RAM"
            s = str(ram)
            if "g" in s.lower():
                num = float(''.join([c for c in s if (c.isdigit() or c=='.')]) or 0)
                if num >= 8:
                    pros.append("High RAM (>= 8GB). Good for multitasking.")
                elif num <= 2:
                    cons.append("Low RAM (<= 2GB). May feel slow under load.")
        except Exception:
            pass

    cam1 = specs.get("cam1_mp") or specs.get("cam1")
    try:
        if cam1 and float(cam1) >= 12:
            pros.append("Primary camera >= 12MP.")
    except Exception:
        pass

    # Simple review signals
    if reviews:
        # take some sample polarity via keywords (very naive)
        negative_keywords = ["trash", "don't buy", "broken", "battery drains", "overheat", "issue", "bug", "problem", "no signal"]
        positive_keywords = ["love", "great", "excellent", "amazing", "smooth", "fast", "battery life"]
        neg_count = 0
        pos_count = 0
        for r in reviews[:10]:
            rl = r.lower()
            for k in negative_keywords:
                if k in rl:
                    neg_count += 1
            for k in positive_keywords:
                if k in rl:
                    pos_count += 1
        if pos_count > neg_count:
            pros.append("Multiple positive user reports (sampled from Reddit).")
        elif neg_count > pos_count:
            cons.append("Multiple negative user reports (sampled from Reddit).")

    # Create human readable fallback
    out = []
    out.append("Fallback heuristic recommendation (no LLM key configured):")
    out.append("\nPros:")
    if pros:
        out += [f"- {p}" for p in pros]
    else:
        out.append("- No clear pros extracted heuristically.")
    out.append("\nCons:")
    if cons:
        out += [f"- {c}" for c in cons]
    else:
        out.append("- No clear cons extracted heuristically.")
    return "\n".join(out)

def generate_recommendation(phone_name: str, top_k: int = 3, min_score: int = 60) -> str:
    """
    Get specs via MCP, fetch Reddit reviews, then produce a structured recommendation via Mistral LLM.
    Uses the Mistral SDK correctly (chat.complete).
    """
    phone_name_clean = phone_name.strip()
    # 1) fetch top-k fuzzy matches (RAG)
    specs_resp = call_mcp_tool("phone_specs", phone_name=phone_name_clean, top_k=top_k, min_score=min_score)
    if not isinstance(specs_resp, dict) or "error" in specs_resp:
        if isinstance(specs_resp, dict) and "error" in specs_resp:
            return f"Error fetching specs: {specs_resp['error']['message']}"
        return "Error fetching specs: unknown error"

    matches = specs_resp.get("results", [])
    if not matches:
        diag = specs_resp.get("diag") or {}
        return f"No confident matches for '{phone_name_clean}'. Diagnostic: {diag}"

    # choose top match
    top = matches[0]
    model_search_name = top.get("search_name") or f"{top.get('brand','')} {top.get('model','')}".strip()
    specs = top.get("specs", {})

    # 2) fetch reddit reviews using search_name
    reviews_resp = call_mcp_tool("web_scraper", phone_name=model_search_name, max_results=20, use_cache=True)
    reddit_comments = []
    if isinstance(reviews_resp, dict) and "reddit_comments" in reviews_resp:
        reddit_comments = reviews_resp.get("reddit_comments", [])
    elif isinstance(reviews_resp, dict) and "error" in reviews_resp:
        # keep the error message but continue with empty comments
        reddit_comments = []

    # Build prompt
    other_matches_text = ""
    if len(matches) > 1:
        other_matches_text = "\nClose candidates:\n"
        for i, m in enumerate(matches, start=1):
            other_matches_text += f"{i}. {m.get('search_name')} (score: {m.get('score')})\n"

    prompt = f"""
You are a phone expert. Analyze the following phone candidate and user feedback, then produce a concise structured recommendation.

Primary candidate:
Name: {model_search_name}
Top specs (truncated): { {k: specs.get(k) for k in list(specs.keys())[:10]} }

{other_matches_text}

User reviews (sample - up to 10):
{json.dumps(reddit_comments[:10], ensure_ascii=False, indent=2)}

Provide output as plain text with these sections:
1) Pros (bullet points)
2) Cons (bullet points)
3) Best use cases
4) Notes on discrepancies between specs and real-world experience
"""

    # If Mistral key not set or SDK not available -> return fallback heuristic summary
    if not MISTRAL_API_KEY or not MISTRAL_SDK_AVAILABLE:
        return _heuristic_summary_from_specs_and_reviews(specs, reddit_comments)

    # Call Mistral properly
    try:
        with Mistral(api_key=MISTRAL_API_KEY) as mistral:
            res = mistral.chat.complete(
                model=os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
                messages=[{"content": prompt, "role": "user"}],
                stream=False
            )
        return _parse_mistral_response(res)
    except Exception as e:
        # return fallback heuristic if LLM fails, and include error for debugging
        fallback = _heuristic_summary_from_specs_and_reviews(specs, reddit_comments)
        return f"LLM call failed: {e}\n\nFallback summary:\n{fallback}"

if __name__ == "__main__":
    # quick local test (when running within Backend folder)
    print(generate_recommendation("iPhone 12 mini"))