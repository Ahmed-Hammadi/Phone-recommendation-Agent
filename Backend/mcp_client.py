# Backend/mcp_client.py

import requests
import os

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp_tool")
DEFAULT_TIMEOUT = int(os.getenv("MCP_TIMEOUT_SECONDS", "300"))

def call_mcp_tool(tool_name: str, **kwargs):
    """
    Send { "tool_name": "...", "kwargs": { ... } } to MCP server.
    """
    payload = {"tool_name": tool_name, "kwargs": kwargs}
    try:
        resp = requests.post(MCP_SERVER_URL, json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"error": {"message": "Invalid JSON response from MCP server"}}
    except requests.HTTPError as http_err:
        return {"error": {"message": f"HTTP error: {http_err}"}}
    except requests.RequestException as req_err:
        return {"error": {"message": f"Request error: {req_err}"}}
    except Exception as e:
        return {"error": {"message": str(e)}}

if __name__ == "__main__":
    phone = "iPhone 16"
    print("Specs:", call_mcp_tool("phone_specs", phone_name=phone, top_k=3))
    print("Reviews:", call_mcp_tool("web_scraper", phone_name=phone, max_results=3))
