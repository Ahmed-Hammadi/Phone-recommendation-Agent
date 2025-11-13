# Backend/mcp_client.py

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import requests

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000")
JSONRPC_ENDPOINT = os.getenv("MCP_JSONRPC_ENDPOINT", MCP_SERVER_URL.rstrip('/') + "/rpc")
DEFAULT_TIMEOUT = int(os.getenv("MCP_TIMEOUT_SECONDS", "300"))

JsonRpcCall = Dict[str, Any]


def _prepare_rpc_payload(method: str, params: Optional[Dict[str, Any]], rpc_id: Any) -> JsonRpcCall:
    payload: JsonRpcCall = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    if rpc_id is not None:
        payload["id"] = rpc_id
    return payload


def call_jsonrpc(
    method: str,
    params: Optional[Dict[str, Any]] = None,
    rpc_id: Any = 1,
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Send a JSON-RPC 2.0 request to the server and return the parsed response.

    If rpc_id is None, a notification is sent and no response is expected.
    """
    payload = _prepare_rpc_payload(method, params, rpc_id)

    try:
        resp = requests.post(JSONRPC_ENDPOINT, json=payload, timeout=timeout)
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


def send_notification(method: str, params: Optional[Dict[str, Any]] = None, timeout: int = DEFAULT_TIMEOUT) -> Optional[Dict[str, Any]]:
    """Send a JSON-RPC notification (id=None). Should return None for 204 responses.

    Returns an error dict if the server responds with content despite notification semantics.
    """
    payload = _prepare_rpc_payload(method, params, rpc_id=None)
    try:
        resp = requests.post(JSONRPC_ENDPOINT, json=payload, timeout=timeout)
        if resp.status_code == 204:
            return None
        if resp.content:
            try:
                return resp.json()
            except ValueError:
                return {"error": {"message": "Invalid JSON response from MCP server"}}
        return None
    except requests.RequestException as exc:
        return {"error": {"message": f"Request error: {exc}"}}


def call_batch(calls: Sequence[Tuple[str, Optional[Dict[str, Any]], Any]], timeout: int = DEFAULT_TIMEOUT) -> Union[List[Any], Dict[str, Any]]:
    """Send a batch of JSON-RPC calls.

    Args:
        calls: iterable of tuples ``(method, params, rpc_id)``. Use ``None`` for rpc_id to send a notification.

    Returns:
        A list with the server responses or an error dict if the HTTP request failed.
    """
    payload: List[JsonRpcCall] = []
    for method, params, rpc_id in calls:
        payload.append(_prepare_rpc_payload(method, params, rpc_id))

    try:
        resp = requests.post(JSONRPC_ENDPOINT, json=payload, timeout=timeout)
        resp.raise_for_status()
        if not resp.content:
            return []
        try:
            return resp.json()
        except ValueError:
            return {"error": {"message": "Invalid JSON response from MCP server"}}
    except requests.HTTPError as http_err:
        return {"error": {"message": f"HTTP error: {http_err}"}}
    except requests.RequestException as req_err:
        return {"error": {"message": f"Request error: {req_err}"}}
    except Exception as exc:
        return {"error": {"message": str(exc)}}


def call_mcp_tool(tool_name: str, **kwargs):
    """Call MCP tool via JSON-RPC mcp.call_tool method and return the inner result or error dict.
    Keeps a similar return shape as before (dict with 'error' on failure or tool output on success).
    """
    rpc_resp = call_jsonrpc("mcp.call_tool", params={"tool_name": tool_name, "kwargs": kwargs})
    if not isinstance(rpc_resp, dict):
        return {"error": {"message": "Invalid RPC response"}}

    if "error" in rpc_resp:
        return {"error": rpc_resp["error"]}

    # JSON-RPC success wrapper
    if "result" in rpc_resp:
        return rpc_resp["result"]

    return {"error": {"message": "Unexpected RPC response format"}}


def list_mcp_tools() -> Dict[str, Any]:
    """Fetch available MCP tools via JSON-RPC."""
    return call_jsonrpc("mcp.list_tools")


def agent_recommend(phone_name: str, top_k: int = 3, min_score: int = 60) -> Dict[str, Any]:
    """Invoke the agent.recommend JSON-RPC method."""
    params = {"phone_name": phone_name, "top_k": top_k, "min_score": min_score}
    return call_jsonrpc("agent.recommend", params=params)


def agent_chat(query: str, include_reasoning: bool = False) -> Dict[str, Any]:
    """Invoke the agent.chat JSON-RPC method."""
    params = {"query": query, "include_reasoning": include_reasoning}
    return call_jsonrpc("agent.chat", params=params)

if __name__ == "__main__":
    phone = "iPhone 16"
    print("Specs:", call_mcp_tool("phone_specs", phone_name=phone, top_k=3))
    print("Reviews:", call_mcp_tool("web_scraper", phone_name=phone, max_results=3))