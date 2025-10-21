# Backend/test_mcp.py

import pprint
import llm_agent  # local import (expected when running from Backend/)
import mcp_client

def test_mcp_pipeline(phone_name: str):
    print(f"\nTesting MCP tools for phone: {phone_name}\n")

    # 1) Get fuzzy specs
    print("Fetching phone specs (top matches)...")
    specs_result = mcp_client.call_mcp_tool("phone_specs", phone_name=phone_name, top_k=5, min_score=60)
    pprint.pprint(specs_result)

    # 2) Fetch reddit reviews
    print("\nFetching Reddit reviews...")
    reviews_result = mcp_client.call_mcp_tool("web_scraper", phone_name=phone_name, max_results=10, use_cache=True)
    pprint.pprint(reviews_result)

    # 3) LLM generation (this will call mistral if configured)
    print("\nGenerating LLM recommendation (may use cached reviews)...")
    rec = llm_agent.generate_recommendation(phone_name)
    print("\n=== Recommendation ===\n")
    print(rec)

if __name__ == "__main__":
    test_mcp_pipeline("iphone 12 mini")
