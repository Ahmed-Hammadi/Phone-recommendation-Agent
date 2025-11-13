"""
Alternative Recommender Tool
Suggests similar phones based on specs, price, and features.
Uses database similarity search and intelligent matching.
"""

import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput


class AlternativeRecommenderInput(ToolInput):
    """Input schema for alternative recommender."""
    phone_specs: Dict[str, Any] = Field(
        description="Specifications of the reference phone"
    )
    database: List[Dict[str, Any]] = Field(
        description="Phone database to search from"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of alternatives to return"
    )
    criteria: Optional[List[str]] = Field(
        default=None,
        description="Specific criteria to match: 'price', 'brand', 'performance', 'camera', 'battery'"
    )


def calculate_similarity_score(phone1: Dict[str, Any], phone2: Dict[str, Any], criteria: List[str]) -> float:
    """Calculate similarity score between two phones based on criteria."""
    score = 0.0
    factors = 0
    
    # Price similarity (if available)
    if 'price' in criteria:
        try:
            # Extract numeric price from string like "$500" or "500 USD"
            price1_str = str(phone1.get('price', '0'))
            price2_str = str(phone2.get('price', '0'))
            
            price1 = float(''.join(filter(str.isdigit, price1_str.split('.')[0])))
            price2 = float(''.join(filter(str.isdigit, price2_str.split('.')[0])))
            
            if price1 > 0 and price2 > 0:
                # Score higher if prices are within 20% of each other
                price_diff = abs(price1 - price2) / max(price1, price2)
                score += max(0, 1 - price_diff * 2)  # 0 to 1 scale
                factors += 1
        except:
            pass
    
    # Brand similarity
    if 'brand' in criteria:
        brand1 = str(phone1.get('brand', '')).lower()
        brand2 = str(phone2.get('brand', '')).lower()
        if brand1 and brand2 and brand1 == brand2:
            score += 1.0
        factors += 1
    
    # Performance similarity (chipset)
    if 'performance' in criteria:
        chipset1 = str(phone1.get('chipset', '')).lower()
        chipset2 = str(phone2.get('chipset', '')).lower()
        
        # Extract chipset family (e.g., "snapdragon 8", "a16")
        perf_keywords = ['snapdragon', 'a16', 'a17', 'a15', 'dimensity', 'exynos', 'tensor']
        common_keywords = sum(1 for kw in perf_keywords if kw in chipset1 and kw in chipset2)
        if common_keywords > 0:
            score += 0.8
        factors += 1
    
    # Camera similarity
    if 'camera' in criteria:
        try:
            camera1 = str(phone1.get('main_camera', ''))
            camera2 = str(phone2.get('main_camera', ''))
            
            # Extract MP (e.g., "48 MP" -> 48)
            mp1 = int(''.join(filter(str.isdigit, camera1.split('mp')[0].split()[-1]))) if 'mp' in camera1.lower() else 0
            mp2 = int(''.join(filter(str.isdigit, camera2.split('mp')[0].split()[-1]))) if 'mp' in camera2.lower() else 0
            
            if mp1 > 0 and mp2 > 0:
                mp_diff = abs(mp1 - mp2) / max(mp1, mp2)
                score += max(0, 1 - mp_diff)
            factors += 1
        except:
            factors += 1
    
    # Battery similarity
    if 'battery' in criteria:
        try:
            battery1 = str(phone1.get('battery', ''))
            battery2 = str(phone2.get('battery', ''))
            
            # Extract mAh
            mah1 = int(''.join(filter(str.isdigit, battery1.split('mah')[0]))) if 'mah' in battery1.lower() else 0
            mah2 = int(''.join(filter(str.isdigit, battery2.split('mah')[0]))) if 'mah' in battery2.lower() else 0
            
            if mah1 > 0 and mah2 > 0:
                mah_diff = abs(mah1 - mah2) / max(mah1, mah2)
                score += max(0, 1 - mah_diff)
            factors += 1
        except:
            factors += 1
    
    # Return normalized score
    return score / factors if factors > 0 else 0.0


def find_alternatives(reference_phone: Dict[str, Any], database: List[Dict[str, Any]], 
                     max_results: int, criteria: List[str]) -> List[Dict[str, Any]]:
    """Find alternative phones similar to the reference phone."""
    reference_name = reference_phone.get('name', '').lower()
    
    # Calculate similarity for each phone in database
    alternatives = []
    for phone in database:
        phone_name = phone.get('name', '').lower()
        
        # Skip the reference phone itself
        if phone_name == reference_name:
            continue
        
        # Calculate similarity score
        similarity = calculate_similarity_score(reference_phone, phone, criteria)
        
        if similarity > 0.1:  # Only include if reasonably similar
            alternatives.append({
                "phone": phone,
                "similarity_score": round(similarity, 2),
                "matching_criteria": criteria
            })
    
    # Sort by similarity score (descending)
    alternatives.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return alternatives[:max_results]


def generate_comparison_summary(reference_phone: Dict[str, Any], alternatives: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary comparing the reference phone with alternatives."""
    if not alternatives:
        return {"message": "No similar alternatives found"}
    
    best_alternative = alternatives[0]['phone']
    
    # Compare key specs
    comparisons = []
    
    # Price comparison
    ref_price = reference_phone.get('price', 'N/A')
    alt_price = best_alternative.get('price', 'N/A')
    if ref_price != 'N/A' and alt_price != 'N/A':
        comparisons.append(f"Price: {ref_price} vs {alt_price}")
    
    # Camera comparison
    ref_camera = reference_phone.get('main_camera', 'N/A')
    alt_camera = best_alternative.get('main_camera', 'N/A')
    comparisons.append(f"Camera: {ref_camera} vs {alt_camera}")
    
    # Battery comparison
    ref_battery = reference_phone.get('battery', 'N/A')
    alt_battery = best_alternative.get('battery', 'N/A')
    comparisons.append(f"Battery: {ref_battery} vs {alt_battery}")
    
    return {
        "best_alternative": best_alternative.get('name', 'Unknown'),
        "similarity": alternatives[0]['similarity_score'],
        "key_comparisons": comparisons,
        "total_alternatives": len(alternatives)
    }


def alternative_recommender_tool(phone_specs: Dict[str, Any], database: List[Dict[str, Any]], 
                                max_results: int = 5, criteria: Optional[List[str]] = None) -> ToolOutput:
    """
    Main alternative recommender function.
    
    Args:
        phone_specs: Specifications of reference phone
        database: Phone database to search
        max_results: Maximum alternatives to return
        criteria: Matching criteria
        
    Returns:
        ToolOutput with alternative recommendations
    """
    start_time = time.time()
    
    try:
        # Default criteria if not specified
        if not criteria:
            criteria = ['price', 'performance', 'camera', 'battery']
        
        # Find alternatives
        alternatives = find_alternatives(phone_specs, database, max_results, criteria)
        
        if not alternatives:
            return ToolOutput(
                success=True,
                data={
                    "reference_phone": phone_specs.get('name', 'Unknown'),
                    "alternatives": [],
                    "message": "No similar alternatives found in database"
                },
                metadata={"execution_time": round(time.time() - start_time, 3)}
            )
        
        # Generate comparison summary
        summary = generate_comparison_summary(phone_specs, alternatives)
        
        # Format alternatives for output
        formatted_alternatives = []
        for alt in alternatives:
            formatted_alternatives.append({
                "name": alt['phone'].get('name', 'Unknown'),
                "brand": alt['phone'].get('brand', 'Unknown'),
                "similarity_score": alt['similarity_score'],
                "key_specs": {
                    "price": alt['phone'].get('price', 'N/A'),
                    "chipset": alt['phone'].get('chipset', 'N/A'),
                    "main_camera": alt['phone'].get('main_camera', 'N/A'),
                    "battery": alt['phone'].get('battery', 'N/A'),
                    "display": alt['phone'].get('display_type', 'N/A')
                },
                "match_score": f"{int(alt['similarity_score'] * 100)}% match"
            })
        
        execution_time = time.time() - start_time
        
        return ToolOutput(
            success=True,
            data={
                "reference_phone": phone_specs.get('name', 'Unknown'),
                "alternatives": formatted_alternatives,
                "summary": summary,
                "criteria_used": criteria
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "database_size": len(database),
                "results_returned": len(alternatives)
            }
        )
    
    except Exception as e:
        return ToolOutput(
            success=False,
            error=f"Alternative recommendation failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
ALTERNATIVE_RECOMMENDER_SCHEMA = {
    "name": "alternative_recommender",
    "description": "Suggests similar alternative phones based on specs, price, and features using intelligent similarity matching",
    "input_schema": {
        "type": "object",
        "properties": {
            "phone_specs": {
                "type": "object",
                "description": "Specifications of the reference phone"
            },
            "database": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Phone database to search from"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of alternatives to return",
                "default": 5
            },
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Matching criteria: price, brand, performance, camera, battery"
            }
        },
        "required": ["phone_specs", "database"]
    },
    "version": "1.0.0"
}
