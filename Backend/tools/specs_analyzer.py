"""
Specs Analyzer Tool
Categorizes and analyzes phone specifications into meaningful categories.
Provides technical insights on performance, battery, camera, display, etc.
"""

import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput


class SpecsAnalyzerInput(ToolInput):
    """Input schema for specs analyzer."""
    specs: Dict[str, Any] = Field(
        description="Phone specifications to analyze (from database)"
    )
    focus_areas: Optional[list[str]] = Field(
        default=None,
        description="Specific areas to focus on: 'performance', 'battery', 'camera', 'display', 'connectivity'"
    )


class SpecsAnalyzerOutput(ToolOutput):
    """Output includes categorized specs with analysis."""
    pass


def analyze_performance(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze performance-related specs."""
    chipset = specs.get('chipset', 'Unknown')
    cpu = specs.get('cpu', 'Unknown')
    gpu = specs.get('gpu', 'Unknown')
    
    # Simple categorization
    performance_tier = "mid-range"
    if any(keyword in str(chipset).lower() for keyword in ['snapdragon 8', 'a17', 'a16', 'dimensity 9']):
        performance_tier = "flagship"
    elif any(keyword in str(chipset).lower() for keyword in ['snapdragon 4', 'helio', 'a13', 'exynos 9']):
        performance_tier = "budget"
    
    return {
        "chipset": chipset,
        "cpu": cpu,
        "gpu": gpu,
        "tier": performance_tier,
        "gaming_capable": performance_tier in ["flagship", "mid-range"]
    }


def analyze_battery(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze battery specs."""
    battery = specs.get('battery', 'Unknown')
    charging = specs.get('charging', 'Unknown')
    
    # Extract battery capacity
    battery_mah = 0
    if isinstance(battery, str) and 'mah' in battery.lower():
        try:
            battery_mah = int(''.join(filter(str.isdigit, battery.split('mah')[0])))
        except:
            pass
    
    battery_rating = "average"
    if battery_mah >= 5000:
        battery_rating = "excellent"
    elif battery_mah >= 4000:
        battery_rating = "good"
    elif battery_mah >= 3000:
        battery_rating = "average"
    elif battery_mah > 0:
        battery_rating = "below_average"
    
    # Check fast charging
    fast_charging = False
    charging_wattage = 0
    if isinstance(charging, str):
        fast_charging = any(keyword in charging.lower() for keyword in ['fast', 'quick', 'turbo', 'dash'])
        # Extract wattage
        try:
            if 'w' in charging.lower():
                charging_wattage = int(''.join(filter(str.isdigit, charging.split('w')[0].split()[-1])))
        except:
            pass
    
    return {
        "capacity_mah": battery_mah if battery_mah > 0 else None,
        "rating": battery_rating,
        "fast_charging": fast_charging,
        "charging_wattage": charging_wattage if charging_wattage > 0 else None,
        "wireless_charging": "wireless" in str(charging).lower() if charging else False
    }


def analyze_camera(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze camera specs."""
    main_camera = specs.get('main_camera', 'Unknown')
    selfie_camera = specs.get('selfie_camera', 'Unknown')
    video = specs.get('video', 'Unknown')
    
    # Extract main camera MP
    main_mp = 0
    if isinstance(main_camera, str) and 'mp' in main_camera.lower():
        try:
            main_mp = int(''.join(filter(str.isdigit, main_camera.split('mp')[0].split()[-1])))
        except:
            pass
    
    # Camera features
    features = []
    if isinstance(main_camera, str):
        if 'optical' in main_camera.lower() or 'ois' in main_camera.lower():
            features.append("optical_stabilization")
        if 'telephoto' in main_camera.lower() or 'periscope' in main_camera.lower():
            features.append("telephoto")
        if 'ultra' in main_camera.lower() or 'wide' in main_camera.lower():
            features.append("ultrawide")
    
    # Video capabilities
    video_4k = "4k" in str(video).lower() if video else False
    video_8k = "8k" in str(video).lower() if video else False
    
    camera_quality = "average"
    if main_mp >= 64 or len(features) >= 2:
        camera_quality = "excellent"
    elif main_mp >= 48 or len(features) >= 1:
        camera_quality = "good"
    
    return {
        "main_camera_mp": main_mp if main_mp > 0 else None,
        "camera_quality": camera_quality,
        "features": features,
        "video_4k": video_4k,
        "video_8k": video_8k,
        "selfie_camera": selfie_camera
    }


def analyze_display(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze display specs."""
    display_type = specs.get('display_type', 'Unknown')
    display_size = specs.get('display_size', 'Unknown')
    display_resolution = specs.get('display_resolution', 'Unknown')
    
    # Display technology
    is_oled = any(tech in str(display_type).lower() for tech in ['oled', 'amoled', 'super amoled'])
    is_lcd = 'lcd' in str(display_type).lower() or 'ips' in str(display_type).lower()
    
    # Refresh rate
    refresh_rate = 60
    if isinstance(display_type, str):
        if '144hz' in display_type.lower():
            refresh_rate = 144
        elif '120hz' in display_type.lower():
            refresh_rate = 120
        elif '90hz' in display_type.lower():
            refresh_rate = 90
    
    display_quality = "average"
    if is_oled and refresh_rate >= 120:
        display_quality = "flagship"
    elif is_oled or refresh_rate >= 90:
        display_quality = "good"
    
    return {
        "type": "OLED" if is_oled else ("LCD" if is_lcd else "Unknown"),
        "size": display_size,
        "resolution": display_resolution,
        "refresh_rate": refresh_rate,
        "quality": display_quality,
        "hdr_support": "hdr" in str(display_type).lower() if display_type else False
    }


def analyze_connectivity(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze connectivity specs."""
    network = specs.get('network', 'Unknown')
    wlan = specs.get('wlan', 'Unknown')
    bluetooth = specs.get('bluetooth', 'Unknown')
    nfc = specs.get('nfc', 'Unknown')
    
    has_5g = '5g' in str(network).lower() if network else False
    has_nfc = nfc and str(nfc).lower() not in ['no', 'unknown', '-']
    
    # WiFi generation
    wifi_gen = 5
    if isinstance(wlan, str):
        if 'wi-fi 7' in wlan.lower() or '802.11be' in wlan:
            wifi_gen = 7
        elif 'wi-fi 6' in wlan.lower() or '802.11ax' in wlan:
            wifi_gen = 6
    
    return {
        "5g_support": has_5g,
        "wifi_generation": wifi_gen,
        "nfc": has_nfc,
        "bluetooth": bluetooth,
        "future_proof": has_5g and wifi_gen >= 6
    }


def specs_analyzer_tool(specs: Dict[str, Any], focus_areas: Optional[list[str]] = None) -> ToolOutput:
    """
    Main specs analyzer function.
    
    Args:
        specs: Phone specifications dictionary
        focus_areas: Optional list of areas to focus on
        
    Returns:
        ToolOutput with categorized analysis
    """
    start_time = time.time()
    
    try:
        # Default to all areas if not specified
        if not focus_areas:
            focus_areas = ['performance', 'battery', 'camera', 'display', 'connectivity']
        
        analysis = {}
        
        if 'performance' in focus_areas:
            analysis['performance'] = analyze_performance(specs)
        
        if 'battery' in focus_areas:
            analysis['battery'] = analyze_battery(specs)
        
        if 'camera' in focus_areas:
            analysis['camera'] = analyze_camera(specs)
        
        if 'display' in focus_areas:
            analysis['display'] = analyze_display(specs)
        
        if 'connectivity' in focus_areas:
            analysis['connectivity'] = analyze_connectivity(specs)
        
        # Overall score (simple heuristic)
        scores = []
        if 'performance' in analysis:
            tier_scores = {'flagship': 9, 'mid-range': 7, 'budget': 5}
            scores.append(tier_scores.get(analysis['performance']['tier'], 6))
        if 'battery' in analysis:
            rating_scores = {'excellent': 9, 'good': 7, 'average': 5, 'below_average': 3}
            scores.append(rating_scores.get(analysis['battery']['rating'], 5))
        if 'camera' in analysis:
            quality_scores = {'excellent': 9, 'good': 7, 'average': 5}
            scores.append(quality_scores.get(analysis['camera']['camera_quality'], 5))
        if 'display' in analysis:
            quality_scores = {'flagship': 9, 'good': 7, 'average': 5}
            scores.append(quality_scores.get(analysis['display']['quality'], 5))
        
        overall_score = sum(scores) / len(scores) if scores else 6.0
        
        execution_time = time.time() - start_time
        
        return ToolOutput(
            success=True,
            data={
                "analysis": analysis,
                "overall_score": round(overall_score, 1),
                "phone_name": specs.get('name', 'Unknown')
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "areas_analyzed": focus_areas
            }
        )
    
    except Exception as e:
        return ToolOutput(
            success=False,
            error=f"Specs analysis failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
SPECS_ANALYZER_SCHEMA = {
    "name": "specs_analyzer",
    "description": "Analyzes phone specifications and categorizes them into performance, battery, camera, display, and connectivity with quality ratings",
    "input_schema": {
        "type": "object",
        "properties": {
            "specs": {
                "type": "object",
                "description": "Phone specifications dictionary from database"
            },
            "focus_areas": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional areas to focus on: performance, battery, camera, display, connectivity"
            }
        },
        "required": ["specs"]
    },
    "version": "1.0.0"
}
