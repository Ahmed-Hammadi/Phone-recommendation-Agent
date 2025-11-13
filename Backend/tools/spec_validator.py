"""
Spec Validator Tool
Validates if phone specifications meet user requirements.
Interprets natural language requirements and checks compatibility.
"""

import time
import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput


class SpecValidatorInput(ToolInput):
    """Input schema for spec validator."""
    specs: Dict[str, Any] = Field(
        description="Phone specifications to validate"
    )
    requirements: List[str] = Field(
        description="User requirements in natural language (e.g., 'good battery', '5G support')"
    )


# Requirement patterns and validation logic
REQUIREMENT_PATTERNS = {
    # Battery requirements
    'good_battery': {
        'keywords': ['good battery', 'long battery', 'battery life'],
        'validator': lambda specs: extract_battery_mah(specs.get('battery', '')) >= 4000
    },
    'excellent_battery': {
        'keywords': ['excellent battery', 'great battery', 'amazing battery'],
        'validator': lambda specs: extract_battery_mah(specs.get('battery', '')) >= 5000
    },
    'fast_charging': {
        'keywords': ['fast charging', 'quick charge', 'rapid charging'],
        'validator': lambda specs: 'fast' in str(specs.get('charging', '')).lower() or 
                                   extract_charging_wattage(specs.get('charging', '')) >= 30
    },
    
    # Camera requirements
    'good_camera': {
        'keywords': ['good camera', 'nice camera', 'decent camera'],
        'validator': lambda specs: extract_camera_mp(specs.get('main_camera', '')) >= 48
    },
    'excellent_camera': {
        'keywords': ['excellent camera', 'great camera', 'amazing camera', 'pro camera'],
        'validator': lambda specs: extract_camera_mp(specs.get('main_camera', '')) >= 64
    },
    
    # Performance requirements
    'good_performance': {
        'keywords': ['good performance', 'fast', 'smooth performance'],
        'validator': lambda specs: any(keyword in str(specs.get('chipset', '')).lower() 
                                      for keyword in ['snapdragon 7', 'snapdragon 8', 'a15', 'a16', 'a17', 'dimensity 9'])
    },
    'gaming': {
        'keywords': ['gaming', 'play games', 'mobile gaming'],
        'validator': lambda specs: any(keyword in str(specs.get('chipset', '')).lower() 
                                      for keyword in ['snapdragon 8', 'a16', 'a17', 'dimensity 9'])
    },
    
    # Display requirements
    'large_screen': {
        'keywords': ['large screen', 'big screen', 'big display'],
        'validator': lambda specs: extract_display_size(specs.get('display_size', '')) >= 6.5
    },
    'oled': {
        'keywords': ['oled', 'amoled', 'oled display'],
        'validator': lambda specs: any(tech in str(specs.get('display_type', '')).lower() 
                                      for tech in ['oled', 'amoled'])
    },
    'high_refresh': {
        'keywords': ['high refresh', '120hz', '90hz', 'smooth display'],
        'validator': lambda specs: '120hz' in str(specs.get('display_type', '')).lower() or 
                                   '90hz' in str(specs.get('display_type', '')).lower()
    },
    
    # Connectivity requirements
    '5g': {
        'keywords': ['5g', '5g support', '5g network'],
        'validator': lambda specs: '5g' in str(specs.get('network', '')).lower()
    },
    'nfc': {
        'keywords': ['nfc', 'contactless payment', 'tap to pay'],
        'validator': lambda specs: specs.get('nfc', '').lower() not in ['no', 'unknown', '-', '']
    },
    
    # Storage requirements
    'large_storage': {
        'keywords': ['large storage', 'lots of storage', 'high storage'],
        'validator': lambda specs: extract_storage_gb(specs.get('internal_memory', '')) >= 256
    },
    
    # Build requirements
    'premium_build': {
        'keywords': ['premium', 'premium build', 'metal', 'glass'],
        'validator': lambda specs: any(material in str(specs.get('body', '')).lower() 
                                      for material in ['metal', 'aluminum', 'glass', 'ceramic'])
    },
    'water_resistant': {
        'keywords': ['water resistant', 'waterproof', 'ip rating'],
        'validator': lambda specs: 'ip' in str(specs.get('body', '')).lower()
    }
}


def extract_battery_mah(battery_str: str) -> int:
    """Extract battery capacity in mAh from string."""
    try:
        if 'mah' in battery_str.lower():
            return int(''.join(filter(str.isdigit, battery_str.split('mah')[0])))
    except:
        pass
    return 0


def extract_charging_wattage(charging_str: str) -> int:
    """Extract charging wattage from string."""
    try:
        if 'w' in charging_str.lower():
            return int(''.join(filter(str.isdigit, charging_str.split('w')[0].split()[-1])))
    except:
        pass
    return 0


def extract_camera_mp(camera_str: str) -> int:
    """Extract camera megapixels from string."""
    try:
        if 'mp' in camera_str.lower():
            return int(''.join(filter(str.isdigit, camera_str.split('mp')[0].split()[-1])))
    except:
        pass
    return 0


def extract_display_size(display_str: str) -> float:
    """Extract display size in inches from string."""
    try:
        # Look for patterns like "6.5 inches" or "6.5\""
        match = re.search(r'(\d+\.\d+)', display_str)
        if match:
            return float(match.group(1))
    except:
        pass
    return 0.0


def extract_storage_gb(storage_str: str) -> int:
    """Extract storage in GB from string."""
    try:
        if 'gb' in storage_str.lower():
            return int(''.join(filter(str.isdigit, storage_str.split('gb')[0].split()[-1])))
        elif 'tb' in storage_str.lower():
            return int(''.join(filter(str.isdigit, storage_str.split('tb')[0].split()[-1]))) * 1024
    except:
        pass
    return 0


def match_requirement(requirement: str, specs: Dict[str, Any]) -> Dict[str, Any]:
    """Match a single requirement against phone specs."""
    requirement_lower = requirement.lower().strip()
    
    # Find matching pattern
    for pattern_name, pattern_data in REQUIREMENT_PATTERNS.items():
        for keyword in pattern_data['keywords']:
            if keyword in requirement_lower:
                try:
                    is_met = pattern_data['validator'](specs)
                    return {
                        "requirement": requirement,
                        "category": pattern_name.replace('_', ' ').title(),
                        "met": is_met,
                        "status": "✓ Met" if is_met else "✗ Not Met"
                    }
                except Exception as e:
                    return {
                        "requirement": requirement,
                        "category": "Unknown",
                        "met": False,
                        "status": "⚠ Could not validate",
                        "error": str(e)
                    }
    
    # If no pattern matched, return unknown
    return {
        "requirement": requirement,
        "category": "Unknown",
        "met": None,
        "status": "? Unclear requirement"
    }


def calculate_compatibility_score(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall compatibility score."""
    met_count = sum(1 for r in validation_results if r['met'] is True)
    not_met_count = sum(1 for r in validation_results if r['met'] is False)
    total_count = len([r for r in validation_results if r['met'] is not None])
    
    if total_count == 0:
        return {
            "score": 0,
            "percentage": 0,
            "label": "Unable to assess"
        }
    
    percentage = (met_count / total_count) * 100
    
    if percentage >= 80:
        label = "Excellent match"
    elif percentage >= 60:
        label = "Good match"
    elif percentage >= 40:
        label = "Partial match"
    else:
        label = "Poor match"
    
    return {
        "score": round(percentage / 10, 1),  # 0-10 scale
        "percentage": round(percentage, 1),
        "met": met_count,
        "not_met": not_met_count,
        "total": total_count,
        "label": label
    }


def spec_validator_tool(specs: Dict[str, Any], requirements: List[str]) -> ToolOutput:
    """
    Main spec validator function.
    
    Args:
        specs: Phone specifications to validate
        requirements: List of user requirements
        
    Returns:
        ToolOutput with validation results
    """
    start_time = time.time()
    
    try:
        if not requirements:
            return ToolOutput(
                success=False,
                error="No requirements provided for validation"
            )
        
        # Validate each requirement
        validation_results = []
        for req in requirements:
            result = match_requirement(req, specs)
            validation_results.append(result)
        
        # Calculate compatibility score
        compatibility = calculate_compatibility_score(validation_results)
        
        # Categorize results
        met_requirements = [r for r in validation_results if r['met'] is True]
        not_met_requirements = [r for r in validation_results if r['met'] is False]
        unclear_requirements = [r for r in validation_results if r['met'] is None]
        
        # Generate summary
        summary = {
            "phone_name": specs.get('name', 'Unknown'),
            "compatibility": compatibility,
            "requirements_met": len(met_requirements),
            "requirements_not_met": len(not_met_requirements),
            "unclear_requirements": len(unclear_requirements)
        }
        
        execution_time = time.time() - start_time
        
        return ToolOutput(
            success=True,
            data={
                "summary": summary,
                "met_requirements": met_requirements,
                "not_met_requirements": not_met_requirements,
                "unclear_requirements": unclear_requirements,
                "all_results": validation_results
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "total_requirements": len(requirements)
            }
        )
    
    except Exception as e:
        return ToolOutput(
            success=False,
            error=f"Spec validation failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
SPEC_VALIDATOR_SCHEMA = {
    "name": "spec_validator",
    "description": "Validates if phone specifications meet user requirements using natural language understanding",
    "input_schema": {
        "type": "object",
        "properties": {
            "specs": {
                "type": "object",
                "description": "Phone specifications to validate"
            },
            "requirements": {
                "type": "array",
                "items": {"type": "string"},
                "description": "User requirements (e.g., 'good battery', '5G support', 'gaming')"
            }
        },
        "required": ["specs", "requirements"]
    },
    "version": "1.0.0"
}
