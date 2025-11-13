"""
Base schemas for MCP tools.
Defines common Pydantic models and tool response formats.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolInput(BaseModel):
    """Base class for tool inputs with common validation."""
    pass


class ToolOutput(BaseModel):
    """Standard tool output format for MCP compliance."""
    success: bool = Field(description="Whether the tool execution succeeded")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Tool result data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "data": {"result": "example"},
                "error": None,
                "metadata": {"execution_time": 0.5}
            }
        }


class ToolSchema(BaseModel):
    """MCP tool schema definition."""
    name: str = Field(description="Unique tool identifier")
    description: str = Field(description="Human-readable tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for tool input")
    output_schema: Dict[str, Any] = Field(description="JSON schema for tool output")
    version: str = Field(default="1.0.0", description="Tool version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "example_tool",
                "description": "An example tool",
                "input_schema": {"type": "object", "properties": {}},
                "output_schema": {"type": "object", "properties": {}},
                "version": "1.0.0"
            }
        }
