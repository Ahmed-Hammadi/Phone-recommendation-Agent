"""
Intelligent Tool Orchestrator
Analyzes user queries and dynamically selects and executes relevant MCP tools.
Uses Mistral LLM for query understanding and result synthesis.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv

# Load environment
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Import Mistral SDK
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False
    print("⚠️  Mistral SDK not available in orchestrator")

# Import tools
from .tools import (
    phone_specs_tool,
    web_scraper_tool,
    specs_analyzer_tool,
    sentiment_analyzer_tool,
    price_extractor_tool,
    alternative_recommender_tool,
    spec_validator_tool
)


class ToolOrchestrator:
    """
    Intelligent orchestrator that:
    1. Analyzes user query to understand intent
    2. Selects relevant tools dynamically
    3. Executes tools in optimal order
    4. Synthesizes results into coherent response
    """

    def __init__(self, df=None, db_engine=None, use_postgres=False):
        self.df = df
        self.db_engine = db_engine
        self.use_postgres = use_postgres

        # Initialize Mistral client
        self.mistral_client = None
        if MISTRAL_AVAILABLE:
            api_key = os.getenv("MISTRAL_API_KEY")
            if api_key:
                try:
                    self.mistral_client = Mistral(api_key=api_key)
                except Exception as e:
                    print(f"⚠️  Mistral client initialization failed: {e}")

        # Tool registry with descriptions
        self.tools = {
            "phone_specs": {
                "function": lambda **kwargs: phone_specs_tool(**kwargs, df=self.df, db_engine=self.db_engine, use_postgres=self.use_postgres),
                "description": "Search phone database by model name. Use when user mentions specific phone model.",
                "keywords": ["phone", "model", "specs", "specifications", "find phone", "search"]
            },
            "web_scraper": {
                "function": web_scraper_tool,
                "description": "Scrape Reddit reviews and discussions. Use when user asks about opinions, reviews, or what people say.",
                "keywords": ["review", "reddit", "opinion", "what do people say", "user feedback", "community"]
            },
            "specs_analyzer": {
                "function": specs_analyzer_tool,
                "description": "Analyze technical specifications and categorize by performance/battery/camera/display. Use for detailed spec analysis.",
                "keywords": ["analyze", "performance", "battery", "camera", "display", "technical", "quality"]
            },
            "sentiment_analyzer": {
                "function": sentiment_analyzer_tool,
                "description": "Extract pros/cons and sentiment from reviews. Use when analyzing user opinions or feedback.",
                "keywords": ["pros", "cons", "sentiment", "positive", "negative", "feedback analysis"]
            },
            "price_extractor": {
                "function": price_extractor_tool,
                "description": "Fetch up-to-date pricing information from Tunisian retailers.",
                "keywords": ["price", "cost", "buy", "pricing", "how much", "price range"]
            },
            "alternative_recommender": {
                "function": alternative_recommender_tool,
                "description": "Suggest alternative phones based on similar specs and budget.",
                "keywords": ["alternative", "similar", "recommend", "instead", "other options"]
            },
            "spec_validator": {
                "function": spec_validator_tool,
                "description": "Validate if a phone meets specific requirements (battery, camera, budget).",
                "keywords": ["requirement", "battery", "camera", "budget", "validator", "meets"]
            }
        }

        # Simple in-memory session store for conversational context
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def _get_session(self, session_id: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        """Return the session key and mutable session context."""
        key = session_id.strip() if isinstance(session_id, str) else None
        if not key:
            key = "default"
        session = self.sessions.setdefault(key, {
            "last_phone_name": None,
        })
        return key, session

    def _apply_session_context(self, analysis: Dict[str, Any], session: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich the query analysis with persisted session context when needed."""
        enriched = dict(analysis)
        last_phone = session.get("last_phone_name")

        current_phone = enriched.get("phone_name")
        normalized_phone = (current_phone or "").strip().lower()
        generic_tokens = {"it", "its", "spec", "specs", "phone", "model", "device", "this", "that"}
        is_generic = False
        if normalized_phone:
            token_parts = [token for token in normalized_phone.replace("-", " ").split() if token]
            if normalized_phone in generic_tokens:
                is_generic = True
            elif token_parts and all(token in generic_tokens for token in token_parts):
                is_generic = True

        if last_phone and (not normalized_phone or is_generic):
            enriched["phone_name"] = last_phone
            if enriched.get("intent") in (None, "general"):
                enriched["intent"] = "phone_inquiry"
            enriched["phone_from_context"] = True

        raw_selected = enriched.get("selected_tools", [])
        selected_tools = list(raw_selected) if isinstance(raw_selected, list) else []
        if enriched.get("phone_name") and "phone_specs" not in selected_tools:
            selected_tools.insert(0, "phone_specs")
        enriched["selected_tools"] = list(dict.fromkeys(selected_tools))

        if enriched.get("phone_name") and not enriched.get("price_query") and "price_extractor" in enriched["selected_tools"]:
            enriched["price_query"] = enriched["phone_name"]

        return enriched

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query using LLM when available, otherwise heuristics."""
        if self.mistral_client:
            llm_result = self._llm_analyze_query(query)
            if llm_result:
                print("[Orchestrator] Using LLM-based analysis")
                return llm_result
            else:
                print("[Orchestrator] LLM analysis unavailable, falling back to heuristics")
        else:
            print("[Orchestrator] Mistral client not configured, using heuristics")
        return self._heuristic_analyze_query(query)

    def _heuristic_analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze user query to extract phone name, intent, and requirements.
        """
        query_lower = query.lower()
        phone_name = None
        requirements: List[str] = []

        # Extract phone name if mentioned in quotes
        if '"' in query:
            parts = query.split('"')
            if len(parts) > 1:
                phone_name = parts[1].strip()

        # Simple heuristics to extract requirements
        requirement_keywords = {
            "battery": ["battery", "battery life", "mah", "charging"],
            "camera": ["camera", "megapixel", "photo", "video"],
            "performance": ["performance", "processor", "chip", "speed", "fast"],
            "display": ["display", "screen", "amoled", "ips", "lcd"],
            "price": ["price", "budget", "cheap", "expensive", "cost"],
            "storage": ["storage", "rom", "memory", "gb"],
            "ram": ["ram", "memory", "multitask"]
        }

        for req_key, keywords in requirement_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                requirements.append(req_key)

        words = query.split()

        # Identify phone name without quotes
        if not phone_name and words:
            trigger_words = [
                "price", "cost", "specs", "specifications", "review", "details",
                "recommend", "compare", "versus", "vs", "alternative",
            ]
            phone_tokens: List[str] = []
            capture = False
            filler_tokens = {"of", "the", "a", "an", "for"}
            for word in words:
                clean_word = "".join(ch for ch in word if ch.isalnum() or ch in ['-', '+'])
                if not clean_word:
                    continue
                lower_clean = clean_word.lower()
                if lower_clean in trigger_words:
                    if phone_tokens:
                        break
                    capture = True
                    continue
                if capture:
                    if lower_clean in filler_tokens:
                        continue
                    phone_tokens.append(clean_word)
            if phone_tokens:
                phone_name = " ".join(phone_tokens)

        if not phone_name and words:
            stop_words = {
                "what", "whats", "who", "are", "is", "do", "does", "did", "people", "on",
                "reddit", "saying", "about", "the", "a", "an", "to", "for", "me", "tell",
                "show", "please", "give", "share", "any", "opinions", "think", "does",
                "much", "how", "price", "cost", "buy", "cheap", "expensive", "compare", "vs",
                "right", "now", "currently", "today", "available", "availability", "under",
                "over", "between", "around", "roughly", "approximately", "approx", "at",
                "with", "and", "or",
            }
            fallback_tokens: List[str] = []
            capturing = False
            for word in words:
                clean_word = "".join(ch for ch in word if ch.isalnum() or ch in ['-', '+'])
                if not clean_word:
                    continue
                lower_clean = clean_word.lower()
                if not capturing:
                    if lower_clean in stop_words:
                        continue
                    has_upper = any(ch.isupper() for ch in clean_word)
                    has_digit = any(ch.isdigit() for ch in clean_word)
                    long_enough = len(clean_word) > 3
                    if has_upper or has_digit or long_enough:
                        capturing = True
                        fallback_tokens.append(clean_word)
                else:
                    if lower_clean in stop_words:
                        break
                    fallback_tokens.append(clean_word)

                if len(fallback_tokens) >= 6:
                    break

            if fallback_tokens and not any(ch.isalpha() for token in fallback_tokens for ch in token):
                fallback_tokens = []

            if fallback_tokens:
                phone_name = " ".join(fallback_tokens)

        # Determine intent
        intent = "general"
        if any(word in query_lower for word in ["review", "opinion", "people say", "reddit", "feedback"]):
            intent = "review_analysis"
        elif any(word in query_lower for word in ["price", "cost", "buy", "cheap", "expensive"]):
            intent = "price_inquiry"
        elif any(word in query_lower for word in ["compare", "versus", "vs", "alternative", "similar"]):
            intent = "comparison"
        elif any(word in query_lower for word in ["recommend", "suggest", "best", "looking for"]):
            intent = "recommendation"
        elif phone_name:
            intent = "phone_inquiry"

        price_query = None

        # Select tools based on intent and keywords
        selected_tools: List[str] = []

        if phone_name:
            selected_tools.append("phone_specs")

        if intent == "review_analysis":
            selected_tools.extend(["web_scraper", "sentiment_analyzer"])
        elif intent == "price_inquiry" and phone_name:
            selected_tools.append("price_extractor")
            price_query = phone_name
        elif intent == "comparison" and phone_name:
            selected_tools.extend(["specs_analyzer", "alternative_recommender"])
        elif intent == "recommendation":
            if requirements:
                selected_tools.append("spec_validator")
            selected_tools.extend(["specs_analyzer", "alternative_recommender"])
        elif intent == "phone_inquiry":
            selected_tools.append("specs_analyzer")
            if any(word in query_lower for word in ["review", "opinion"]):
                selected_tools.extend(["web_scraper", "sentiment_analyzer"])

        for tool_name, tool_info in self.tools.items():
            if tool_name not in selected_tools:
                if any(keyword in query_lower for keyword in tool_info["keywords"]):
                    selected_tools.append(tool_name)

        if not selected_tools and phone_name:
            selected_tools = ["phone_specs", "specs_analyzer"]

        return {
            "phone_name": phone_name,
            "requirements": requirements,
            "intent": intent,
            "selected_tools": list(dict.fromkeys(selected_tools)),
            "price_query": price_query
        }

    def _llm_analyze_query(self, query: str) -> Optional[Dict[str, Any]]:
        """Use the LLM to decide which tools to invoke."""
        if not self.mistral_client:
            return None

        allowed_tools = list(self.tools.keys())
        tool_catalog = [
            {
                "name": name,
                "description": info.get("description", "")
            }
            for name, info in self.tools.items()
        ]

        system_prompt = """You are the orchestration planner for a phone recommendation agent.\nDecide which tools are most relevant for the user's query and extract the phone name if mentioned.\nAlways respond with a single JSON object and nothing else."""

        user_prompt = json.dumps(
            {
                "query": query,
                "tools": tool_catalog,
                "fields": {
                    "phone_name": "String or null. Exact phone model if mentioned, else null.",
                    "requirements": "Array of strings capturing user requirements (battery, camera, price, etc).",
                    "intent": "One of: price_inquiry, review_analysis, comparison, recommendation, phone_inquiry, general.",
                    "selected_tools": "Array of tool names (from provided list) in the order they should run.",
                    "price_query": "String or null. Canonical phone name to use for live price scraping when price_extractor is selected."
                },
                "notes": [
                    "Only include tools that add value for the query.",
                    "Include 'phone_specs' whenever a specific phone name is detected.",
                    "For pricing questions, include 'price_extractor'.",
                    "For opinions, include 'web_scraper' (and optionally 'sentiment_analyzer').",
                    "Return valid JSON."
                ]
            },
            indent=2
        )

        try:
            response = self.mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=512
            )
            content = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"⚠️  LLM analysis failed: {exc}")
            return None

        def _parse_json_block(text: str) -> Optional[Dict[str, Any]]:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    snippet = text[start:end + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        return None
            return None

        parsed = _parse_json_block(content)
        if not parsed:
            print("⚠️  LLM returned non-JSON analysis. Falling back to heuristics.")
            return None

        phone_name = parsed.get("phone_name")
        if isinstance(phone_name, str):
            phone_name = phone_name.strip() or None
        else:
            phone_name = None

        requirements = parsed.get("requirements")
        if not isinstance(requirements, list):
            requirements = []
        else:
            requirements = [str(req).strip().lower() for req in requirements if isinstance(req, str) and req.strip()]

        intent = parsed.get("intent")
        if not isinstance(intent, str) or not intent.strip():
            intent = "general"
        else:
            intent = intent.strip()

        price_query = parsed.get("price_query")
        if isinstance(price_query, str):
            price_query = price_query.strip() or None
        else:
            price_query = None

        selected_tools_raw = parsed.get("selected_tools")
        selected_tools: List[str] = []
        if isinstance(selected_tools_raw, list):
            for name in selected_tools_raw:
                if not isinstance(name, str):
                    continue
                normalized = name.strip()
                if not normalized or normalized not in allowed_tools:
                    continue
                if normalized not in selected_tools:
                    selected_tools.append(normalized)

        heuristic = None
        if not phone_name or not selected_tools or ("price_extractor" in selected_tools and not price_query):
            heuristic = self._heuristic_analyze_query(query)
            print("[Orchestrator] LLM analysis incomplete; supplementing with heuristics")

        if not phone_name and heuristic:
            phone_name = heuristic.get("phone_name")
        if heuristic:
            heuristic_requirements = heuristic.get("requirements", [])
            requirements = list(dict.fromkeys(requirements + heuristic_requirements))
            if not price_query:
                price_query = heuristic.get("price_query")
        if not selected_tools and heuristic:
            selected_tools = heuristic.get("selected_tools", [])
        if heuristic and (intent == "general"):
            intent = heuristic.get("intent", intent)

        if not selected_tools:
            # As a final safeguard, fall back entirely to heuristics
            return heuristic

        return {
            "phone_name": phone_name,
            "requirements": requirements,
            "intent": intent,
            "selected_tools": selected_tools,
            "price_query": price_query
        }

    def execute_tool_chain(self, query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute selected tools in sequence and collect results.
        """
        results = {
            "query_analysis": query_analysis,
            "tool_executions": [],
            "execution_time": 0,
            "errors": []
        }

        start_time = time.time()
        phone_name = query_analysis.get("phone_name")
        selected_tools = query_analysis.get("selected_tools", [])
        requirements = query_analysis.get("requirements", [])
        price_query_hint = query_analysis.get("price_query")

        phone_specs = None
        phone_name_matched = None
        reddit_comments = None

        for tool_name in selected_tools:
            try:
                tool_start = time.time()
                tool_func = self.tools[tool_name]["function"]
                tool_result = None

                if tool_name == "phone_specs" and phone_name:
                    tool_result = tool_func(phone_name=phone_name, top_k=1)
                    if tool_result.success:
                        phone_specs = tool_result.data["results"][0]["specs"]
                        phone_name_matched = tool_result.data["results"][0]["search_name"]

                elif tool_name == "web_scraper" and (phone_name_matched or phone_name):
                    tool_result = tool_func(phone_name=phone_name_matched or phone_name, max_results=20)
                    if tool_result.success:
                        reddit_comments = tool_result.data.get("texts", [])

                elif tool_name == "specs_analyzer" and phone_specs:
                    tool_result = tool_func(specs=phone_specs)

                elif tool_name == "sentiment_analyzer" and reddit_comments:
                    tool_result = tool_func(comments=reddit_comments, phone_model=phone_name_matched or phone_name)

                elif tool_name == "price_extractor":
                    candidate_queue: List[str] = []
                    seen_candidates = set()

                    def enqueue(name: Optional[str]):
                        if not name:
                            return
                        normalized = name.strip()
                        if not normalized:
                            return
                        normalized_compact = normalized.replace(" ", "")
                        if normalized_compact.isdigit():
                            return
                        if len(normalized_compact) < 3:
                            return
                        key = normalized.lower()
                        if key in seen_candidates:
                            return
                        candidate_queue.append(normalized)
                        seen_candidates.add(key)
                        print(f"[Orchestrator][price_extractor] Enqueued candidate: '{normalized}'")

                    def clean_query(name: str) -> str:
                        tokens = name.split()
                        if not tokens:
                            return name
                        prefix_stop = {
                            "what", "whats", "does", "do", "is", "are", "the", "a", "an", "how",
                            "much", "please", "tell", "me", "about", "for", "find", "show", "of"
                        }
                        suffix_stop = {"price", "prices", "cost", "costs", "pricing"}
                        cleaned_tokens: List[str] = []
                        started = False
                        for token in tokens:
                            normalized_token = "".join(ch for ch in token.lower() if ch.isalnum())
                            lower = normalized_token or token.lower()
                            if not started:
                                if lower in prefix_stop:
                                    continue
                                started = True
                            if started and lower in suffix_stop:
                                break
                            cleaned_tokens.append(token)
                        cleaned_name = " ".join(cleaned_tokens) if cleaned_tokens else name.strip()
                        if cleaned_name != name:
                            print(f"[Orchestrator][price_extractor] Cleaned candidate '{name}' -> '{cleaned_name}'")
                        return cleaned_name

                    enqueue(price_query_hint)
                    enqueue(phone_name_matched)
                    enqueue(phone_name)

                    for candidate in list(candidate_queue):
                        refined = clean_query(candidate)
                        enqueue(refined)

                    for candidate in candidate_queue:
                        candidate_clean = candidate.strip()
                        if not candidate_clean:
                            continue
                        print(f"[Orchestrator][price_extractor] Executing price extractor with candidate '{candidate_clean}'")
                        tool_result = tool_func(phone_name=candidate_clean)
                        if tool_result and tool_result.success:
                            data = tool_result.data or {}
                            prices = data.get("prices") if isinstance(data, dict) else None
                            if prices:
                                print(f"[Orchestrator][price_extractor] Candidate '{candidate_clean}' returned {len(prices)} listings")
                                break
                        if tool_result and not tool_result.success:
                            print(f"[Orchestrator][price_extractor] Candidate '{candidate_clean}' failed: {tool_result.error}")

                elif tool_name == "alternative_recommender" and phone_specs and self.df is not None:
                    db_list = self.df.to_dict('records') if self.df is not None else []
                    tool_result = tool_func(phone_specs=phone_specs, database=db_list, max_results=3)

                elif tool_name == "spec_validator" and phone_specs and requirements:
                    tool_result = tool_func(specs=phone_specs, requirements=requirements)

                tool_time = time.time() - tool_start

                if tool_result:
                    results["tool_executions"].append({
                        "tool_name": tool_name,
                        "success": tool_result.success,
                        "data": tool_result.data,
                        "error": tool_result.error,
                        "execution_time": tool_time
                    })
                else:
                    results["tool_executions"].append({
                        "tool_name": tool_name,
                        "success": False,
                        "error": "Tool execution skipped (missing dependencies)",
                        "execution_time": tool_time
                    })

            except Exception as exc:
                results["errors"].append({
                    "tool_name": tool_name,
                    "error": str(exc)
                })

        results["execution_time"] = time.time() - start_time
        if phone_name_matched:
            results["matched_phone_name"] = phone_name_matched
        elif phone_name:
            results["matched_phone_name"] = phone_name
        return results

    def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """End-to-end helper used by the API layer, with lightweight session context."""
        overall_start = time.time()

        session_key, session_ctx = self._get_session(session_id)
        initial_analysis = self.analyze_query(query)
        analysis = self._apply_session_context(initial_analysis, session_ctx)
        analysis["session_id"] = session_key

        execution_results = self.execute_tool_chain(analysis)

        try:
            response_text = self.synthesize_response(query, execution_results)
        except Exception as exc:  # pragma: no cover - safety net
            print(f"⚠️  Response synthesis failed: {exc}")
            response_text = self._fallback_synthesis(query, execution_results)

        total_time = time.time() - overall_start
        execution_results.setdefault("query_analysis", analysis)

        matched_name = execution_results.get("matched_phone_name")
        final_phone = matched_name or analysis.get("phone_name")
        if final_phone:
            session_ctx["last_phone_name"] = final_phone

        return {
            "query": query,
            "analysis": analysis,
            "execution_results": execution_results,
            "response": response_text,
            "total_time": total_time,
            "session_id": session_key,
        }

    def synthesize_response(self, query: str, execution_results: Dict[str, Any]) -> str:
        """
        Use Mistral LLM to synthesize tool results into a coherent natural language response.
        """
        if not self.mistral_client:
            return self._fallback_synthesis(query, execution_results)

        try:
            tool_outputs = []
            for exec_info in execution_results["tool_executions"]:
                if exec_info["success"]:
                    tool_outputs.append({
                        "tool": exec_info["tool_name"],
                        "data": exec_info["data"]
                    })

            system_prompt = """You are an intelligent phone recommendation assistant.
You have access to various tools that provide phone information.
Synthesize the tool outputs into a helpful, natural response to the user's query.
Be conversational, accurate, and cite specific data from the tools."""

            user_prompt = f"""User Query: {query}

Tool Results:
{json.dumps(tool_outputs, indent=2)}

Please provide a comprehensive, natural language response to the user's query based on the tool results above."""

            response = self.mistral_client.chat.complete(
                model="mistral-small-latest",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as exc:
            print(f"⚠️  LLM synthesis failed: {exc}")
            return self._fallback_synthesis(query, execution_results)

    def _fallback_synthesis(self, query: str, execution_results: Dict[str, Any]) -> str:
        """Fallback synthesis without LLM."""
        parts = [
            f"Query: {query}\n",
            f"Intent: {execution_results['query_analysis']['intent']}\n",
            f"Tools executed: {len(execution_results['tool_executions'])}\n\n"
        ]

        for exec_info in execution_results["tool_executions"]:
            if exec_info["success"]:
                parts.append(f"### {exec_info['tool_name'].replace('_', ' ').title()}\n")
                parts.append(f"{json.dumps(exec_info['data'], indent=2)}\n\n")
            else:
                parts.append(f"⚠️  {exec_info['tool_name']} failed: {exec_info['error']}\n\n")

        if execution_results["errors"]:
            parts.append("Errors encountered:\n")
            for error in execution_results["errors"]:
                parts.append(f"- {error['tool_name']}: {error['error']}\n")

        return "".join(parts)
