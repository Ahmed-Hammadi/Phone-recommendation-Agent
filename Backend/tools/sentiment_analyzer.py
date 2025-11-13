"""
Sentiment Analyzer Tool
Analyzes Reddit comments for sentiment, themes, pros/cons.
Uses keyword-based analysis with contextual understanding.
"""

import time
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput


class SentimentAnalyzerInput(ToolInput):
    """Input schema for sentiment analyzer."""
    comments: List[str] = Field(
        description="List of Reddit comments to analyze"
    )
    phone_model: Optional[str] = Field(
        default=None,
        description="Phone model name for context-aware analysis"
    )


# Sentiment keywords
POSITIVE_KEYWORDS = {
    'excellent', 'amazing', 'great', 'awesome', 'fantastic', 'love', 'perfect',
    'best', 'impressive', 'outstanding', 'superb', 'wonderful', 'brilliant',
    'solid', 'good', 'nice', 'decent', 'reliable', 'smooth', 'fast', 'beautiful',
    'recommend', 'happy', 'satisfied', 'pleased', 'works well', 'no issues'
}

NEGATIVE_KEYWORDS = {
    'terrible', 'awful', 'bad', 'worst', 'poor', 'hate', 'disappointing',
    'useless', 'horrible', 'pathetic', 'trash', 'garbage', 'issue', 'problem',
    'broke', 'broken', 'slow', 'lag', 'crash', 'freeze', 'overheat', 'buggy',
    'regret', 'waste', 'avoid', 'overpriced', 'cheap', 'fail', 'died'
}

# Feature-specific keywords
FEATURE_KEYWORDS = {
    'battery': {'battery', 'charge', 'charging', 'battery life', 'mah', 'power'},
    'camera': {'camera', 'photo', 'picture', 'video', 'lens', 'megapixel', 'mp', 'selfie'},
    'performance': {'performance', 'speed', 'fast', 'slow', 'lag', 'fps', 'gaming', 'processor', 'ram'},
    'display': {'screen', 'display', 'brightness', 'oled', 'amoled', 'refresh rate', 'resolution'},
    'build': {'build', 'quality', 'design', 'premium', 'plastic', 'metal', 'glass', 'weight'},
    'software': {'software', 'ui', 'android', 'ios', 'update', 'bloatware', 'os'},
    'price': {'price', 'value', 'worth', 'expensive', 'cheap', 'budget', 'cost', 'overpriced'}
}


def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of a single comment."""
    text_lower = text.lower()
    
    # Count positive and negative keywords
    positive_count = sum(1 for word in POSITIVE_KEYWORDS if word in text_lower)
    negative_count = sum(1 for word in NEGATIVE_KEYWORDS if word in text_lower)
    
    # Calculate sentiment score (-1 to 1)
    total = positive_count + negative_count
    if total == 0:
        sentiment_score = 0
        sentiment_label = "neutral"
    else:
        sentiment_score = (positive_count - negative_count) / total
        if sentiment_score > 0.2:
            sentiment_label = "positive"
        elif sentiment_score < -0.2:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
    
    return {
        "score": round(sentiment_score, 2),
        "label": sentiment_label,
        "positive_keywords": positive_count,
        "negative_keywords": negative_count
    }


def extract_features_mentioned(text: str) -> List[str]:
    """Extract which phone features are mentioned in the text."""
    text_lower = text.lower()
    features = []
    
    for feature, keywords in FEATURE_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            features.append(feature)
    
    return features


def extract_pros_cons(comments: List[str]) -> Dict[str, List[str]]:
    """Extract common pros and cons from comments."""
    pros = []
    cons = []
    
    for comment in comments:
        sentiment = analyze_sentiment(comment)
        features = extract_features_mentioned(comment)
        
        # Extract sentences
        sentences = re.split(r'[.!?]+', comment)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            sentence_sentiment = analyze_sentiment(sentence)
            sentence_features = extract_features_mentioned(sentence)
            
            # Categorize as pro or con
            if sentence_sentiment['label'] == 'positive' and sentence_features:
                pros.append(f"{', '.join(sentence_features)}: {sentence.strip()[:80]}")
            elif sentence_sentiment['label'] == 'negative' and sentence_features:
                cons.append(f"{', '.join(sentence_features)}: {sentence.strip()[:80]}")
    
    # Return top 5 most common
    pros_counter = Counter(pros)
    cons_counter = Counter(cons)
    
    return {
        "pros": [item for item, count in pros_counter.most_common(5)],
        "cons": [item for item, count in cons_counter.most_common(5)]
    }


def extract_themes(comments: List[str]) -> Dict[str, int]:
    """Extract common themes/topics from comments."""
    feature_counts = Counter()
    
    for comment in comments:
        features = extract_features_mentioned(comment)
        for feature in features:
            feature_counts[feature] += 1
    
    return dict(feature_counts.most_common())


def calculate_feature_sentiments(comments: List[str]) -> Dict[str, Dict[str, Any]]:
    """Calculate sentiment for each feature category."""
    feature_sentiments = {feature: [] for feature in FEATURE_KEYWORDS.keys()}
    
    for comment in comments:
        features = extract_features_mentioned(comment)
        sentiment = analyze_sentiment(comment)
        
        for feature in features:
            feature_sentiments[feature].append(sentiment['score'])
    
    # Calculate average sentiment per feature
    result = {}
    for feature, scores in feature_sentiments.items():
        if scores:
            avg_score = sum(scores) / len(scores)
            result[feature] = {
                "avg_sentiment": round(avg_score, 2),
                "mention_count": len(scores),
                "label": "positive" if avg_score > 0.2 else ("negative" if avg_score < -0.2 else "neutral")
            }
    
    return result


def sentiment_analyzer_tool(comments: List[str], phone_model: Optional[str] = None) -> ToolOutput:
    """
    Main sentiment analyzer function.
    
    Args:
        comments: List of Reddit comments to analyze
        phone_model: Optional phone model name for context
        
    Returns:
        ToolOutput with sentiment analysis results
    """
    start_time = time.time()
    
    try:
        if not comments:
            return ToolOutput(
                success=False,
                error="No comments provided for analysis"
            )
        
        # Analyze each comment
        comment_analyses = []
        for comment in comments:
            sentiment = analyze_sentiment(comment)
            features = extract_features_mentioned(comment)
            comment_analyses.append({
                "text": comment[:100] + "..." if len(comment) > 100 else comment,
                "sentiment": sentiment,
                "features_mentioned": features
            })
        
        # Calculate overall sentiment
        overall_scores = [c['sentiment']['score'] for c in comment_analyses]
        overall_sentiment = sum(overall_scores) / len(overall_scores)
        
        # Sentiment distribution
        sentiment_labels = [c['sentiment']['label'] for c in comment_analyses]
        sentiment_distribution = dict(Counter(sentiment_labels))
        
        # Extract pros/cons
        pros_cons = extract_pros_cons(comments)
        
        # Extract themes
        themes = extract_themes(comments)
        
        # Feature-specific sentiments
        feature_sentiments = calculate_feature_sentiments(comments)
        
        # Summary
        summary = {
            "overall_sentiment": round(overall_sentiment, 2),
            "overall_label": "positive" if overall_sentiment > 0.2 else ("negative" if overall_sentiment < -0.2 else "neutral"),
            "total_comments": len(comments),
            "sentiment_distribution": sentiment_distribution
        }
        
        execution_time = time.time() - start_time
        
        return ToolOutput(
            success=True,
            data={
                "summary": summary,
                "pros": pros_cons["pros"],
                "cons": pros_cons["cons"],
                "themes": themes,
                "feature_sentiments": feature_sentiments,
                "phone_model": phone_model
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "comments_analyzed": len(comments)
            }
        )
    
    except Exception as e:
        return ToolOutput(
            success=False,
            error=f"Sentiment analysis failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
SENTIMENT_ANALYZER_SCHEMA = {
    "name": "sentiment_analyzer",
    "description": "Analyzes Reddit comments for sentiment, pros/cons, themes, and feature-specific opinions",
    "input_schema": {
        "type": "object",
        "properties": {
            "comments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of Reddit comments to analyze"
            },
            "phone_model": {
                "type": "string",
                "description": "Optional phone model name for context-aware analysis"
            }
        },
        "required": ["comments"]
    },
    "version": "1.0.0"
}
