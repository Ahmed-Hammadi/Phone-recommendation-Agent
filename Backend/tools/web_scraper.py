"""
Web Scraper Tool (Reddit)
Scrapes Reddit for phone reviews and discussions using PRAW API.
Includes filtering for relevant content and rumor detection.
Extracted from working mcp_server.py implementation.
"""

import time
import os
import re
from typing import Dict, Any, List, Optional, Set
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput

# PRAW (official Reddit API)
try:
    import praw
    PRAW_AVAILABLE = True
except Exception:
    PRAW_AVAILABLE = False


class WebScraperInput(ToolInput):
    """Input schema for web scraper (Reddit)."""
    phone_name: str = Field(description="Phone model name to search for")
    max_results: int = Field(default=20, description="Maximum number of comments/posts to return")
    subreddits: Optional[List[str]] = Field(
        default=None,
        description="List of subreddits to search (default: gadgets, android, iphone, smartphones, all)"
    )


def _get_praw_instance():
    """Get configured PRAW Reddit instance."""
    if not PRAW_AVAILABLE:
        raise RuntimeError("PRAW package not installed. Run `pip install praw`.")
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "phone-recommendation-agent/0.1")
    if not client_id or not client_secret:
        raise RuntimeError("Reddit credentials missing. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars.")
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False
    )


def web_scraper_tool(
    phone_name: str,
    max_results: int = 20,
    subreddits: Optional[List[str]] = None
) -> ToolOutput:
    """
    Reddit scraper using PRAW (official Reddit API) - extracted from mcp_server.py.
    
    Args:
        phone_name: Phone model to search for
        max_results: Maximum number of results to return
        subreddits: List of subreddits to search
        
    Returns:
        ToolOutput with Reddit comments and posts
    """
    start_time = time.time()
    
    try:
        if not phone_name or not isinstance(phone_name, str):
            return ToolOutput(
                success=False,
                error="phone_name must be a non-empty string"
            )

        if not PRAW_AVAILABLE:
            return ToolOutput(
                success=False,
                error="PRAW package not installed. Run `pip install praw`."
            )

        subreddits = subreddits or ["gadgets", "android", "iphone", "smartphones", "all"]
        print(f"[WebScraper] Searching Reddit for '{phone_name}' across: {', '.join(subreddits)}")

        try:
            reddit = _get_praw_instance()
        except Exception as e:
            print(f"[WebScraper] Authentication failed: {e}")
            return ToolOutput(
                success=False,
                error=f"Reddit authentication failed: {str(e)}"
            )

        collected_texts: List[str] = []
        seen: Set[str] = set()
        
        # Extract model number for filtering (e.g., "23" from "Samsung Galaxy S23 Ultra")
        phone_lower = phone_name.lower()
        
        # Extract the primary model number (most specific)
        model_numbers = re.findall(r'\b\d+\b', phone_lower)
        primary_model = model_numbers[0] if model_numbers else None
        
        # Extract other identifiers
        model_tokens = set(re.findall(r'\bpro\b|\bmax\b|\bmini\b|\bplus\b|\bultra\b|\bse\b', phone_lower))

        # Search submissions in each subreddit
        for sr in subreddits:
            try:
                subreddit = reddit.subreddit(sr)
                # Using subreddit.search (sort by relevance first, then new)
                for submission in subreddit.search(phone_name, limit=50, sort="relevance"):
                    if submission is None:
                        continue
                    
                    # Filter: Skip if title/selftext mention other model numbers that conflict
                    title = getattr(submission, "title", "") or ""
                    selftext = getattr(submission, "selftext", "") or ""
                    combined_text = (title + " " + selftext).lower()

                    print(f"[WebScraper] Considering submission: {title[:80]!r}")

                    if primary_model:
                        found_numbers = re.findall(r'\b\d+\b', combined_text)
                        # If post has model numbers and the primary doesn't match, skip
                        if found_numbers and primary_model not in found_numbers:
                            # Allow if it's just mentioning in comparison context
                            if not any(word in combined_text for word in ['upgrade from', 'coming from', 'switched from', 'vs', 'compared to']):
                                continue
                    
                    # Filter out rumors and leaks
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
                        print(f"[WebScraper] Reached max results via submission content in r/{sr}")
                    if len(collected_texts) >= max_results:
                        break

                    # Collect top-level comments
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
                            
                            # Filter comment for model number relevance
                            body_lower = body_clean.lower()
                            
                            # Filter out rumor/speculation comments
                            if any(word in body_lower for word in ['rumor', 'leak', 'upcoming', 'will be', 'expected to', 'might have', 'could have', 'coming soon', 'will be released']):
                                continue
                            
                            if primary_model:
                                found_numbers = re.findall(r'\b\d+\b', body_lower)
                                if found_numbers and primary_model not in found_numbers:
                                    if not any(word in body_lower for word in ['upgrade from', 'coming from', 'switched from', 'vs', 'compared to']):
                                        continue
                            
                            if body_clean not in seen:
                                collected_texts.append(body_clean)
                                seen.add(body_clean)
                        if len(collected_texts) >= max_results:
                            print(f"[WebScraper] Reached max results via comments in r/{sr}")
                    except Exception as comment_exc:
                        # If comment fetching fails, just skip
                        print(f"[WebScraper] Failed to fetch comments: {comment_exc}")
                        pass
                
                if len(collected_texts) >= max_results:
                    print(f"[WebScraper] Collected {len(collected_texts)} texts; stopping subreddit loop")
                    break
                    
            except Exception as sr_exc:
                # If one subreddit fails, continue with others
                print(f"⚠️  Failed to search subreddit {sr}: {sr_exc}")
                continue

        # Trim to max_results
        final_texts = collected_texts[:max_results]
        
        if not final_texts:
            print(f"[WebScraper] No Reddit content found for '{phone_name}'")
            return ToolOutput(
                success=True,
                data={
                    "texts": [],
                    "count": 0,
                    "phone_name": phone_name,
                    "message": f"No Reddit content found for '{phone_name}'"
                },
                metadata={
                    "execution_time": round(time.time() - start_time, 3),
                    "subreddits_searched": subreddits
                }
            )

        execution_time = time.time() - start_time
        print(f"[WebScraper] Returning {len(final_texts)} texts in {execution_time:.2f}s")
        
        return ToolOutput(
            success=True,
            data={
                "texts": final_texts,
                "count": len(final_texts),
                "phone_name": phone_name
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "subreddits_searched": subreddits,
                "filtering": {
                    "primary_model": primary_model,
                    "model_tokens": list(model_tokens) if model_tokens else []
                }
            }
        )
    
    except Exception as e:
        print(f"[WebScraper] Scraping failed: {e}")
        return ToolOutput(
            success=False,
            error=f"Reddit scraping failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
WEB_SCRAPER_SCHEMA = {
    "name": "web_scraper",
    "description": "Scrapes Reddit for phone reviews and discussions using PRAW API with intelligent filtering",
    "input_schema": {
        "type": "object",
        "properties": {
            "phone_name": {
                "type": "string",
                "description": "Phone model name to search for"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of comments/posts to return",
                "default": 20
            },
            "subreddits": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of subreddits to search (default: gadgets, android, iphone, smartphones, all)"
            }
        },
        "required": ["phone_name"]
    },
    "version": "1.0.0"
}
