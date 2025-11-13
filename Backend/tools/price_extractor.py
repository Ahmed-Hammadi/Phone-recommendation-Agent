"""
Price Comparator Tool
Scrapes real prices from Tunisian tech retailers and provides price insights.
Currently supports Tunisianet, Zoom, and Spacenet.
"""

import time
import re
import random
import json
import requests
import cloudscraper
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from .base_schemas import ToolInput, ToolOutput


class PriceComparatorInput(ToolInput):
    """Input schema for price comparator."""
    phone_name: str = Field(description="Phone model name to search for")
    include_used: bool = Field(
        default=False,
        description="Include used/refurbished prices"
    )


# Tunisian retailer configurations
TUNISIAN_RETAILERS = {
    "tunisianet": {
        "name": "Tunisianet",
        "base_url": "https://www.tunisianet.com.tn",
        "search_url": "https://www.tunisianet.com.tn/recherche?controller=search&s={query}",
        "currency": "TND",
        "enabled": True
    },
    "zoom": {
        "name": "Zoom",
        "base_url": "https://zoom.com.tn",
        "search_url": "https://zoom.com.tn/recherche?controller=search&s={query}",
        "currency": "TND",
        "enabled": True
    },
    "spacenet": {
        "name": "Spacenet",
        "base_url": "https://spacenet.tn",
        "search_url": "https://spacenet.tn/module/ambjolisearch/jolisearch?orderby=position&orderway=desc&search_query={query}&submit_search=",
        "currency": "TND",
        "enabled": True
    }
}

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7'
}


# Keywords used to normalize and filter search results
OPTIONAL_QUERY_TOKENS = {
    "apple", "samsung", "xiaomi", "huawei", "oppo", "realme", "google", "sony",
    "nokia", "mobile", "smartphone", "phone", "telephones", "telephone", "android",
    "ios"
}

DISALLOWED_KEYWORDS = {
    "coque", "case", "cover", "etui", "protector", "protection", "verre", "film",
    "chargeur", "chargeurs", "cable", "cables", "adaptateur", "adaptateurs",
    "reparation", "service", "assurance", "support", "dock", "keyboard", "clavier",
    "earbuds", "ecouteurs", "casque", "coques", "stickers", "skin", "protections",
    "pochette", "housse", "bag", "trépied", "trepied", "trépieds", "trepieds",
    "gaming", "covering", "kit", "bundle", "pack", "socle", "support", "charge"
}

VARIANT_KEYWORDS = {"max", "ultra", "lite", "mini", "plus", "fe", "flip", "fold"}


def normalize_product_tokens(text: str) -> List[str]:
    """Split product names into comparable lowercase tokens."""
    if not text:
        return []

    text = text.lower()
    tokens: List[str] = []

    # Replace separators with spaces then split into alphanumeric chunks
    for chunk in re.findall(r"[a-z0-9]+", text):
        for part in re.findall(r"[a-z]+|\d+", chunk):
            if not part:
                continue
            tokens.append(part)

    return tokens


def _is_storage_token(token: str) -> bool:
    """Heuristic to identify storage-capacity tokens (e.g., 128, 256)."""
    return token.isdigit() and len(token) >= 3


def is_relevant_product(title: str, query: str) -> bool:
    """Validate that a scraped product title matches the requested phone model."""
    title_tokens = normalize_product_tokens(title)
    if not title_tokens:
        return False

    if any(token in DISALLOWED_KEYWORDS for token in title_tokens):
        return False

    query_tokens = normalize_product_tokens(query)
    if not query_tokens:
        return True

    title_token_set = set(title_tokens)
    query_token_set = set(query_tokens)

    # Reject variant keywords (e.g., "max", "ultra") when not present in the query
    for keyword in VARIANT_KEYWORDS:
        if keyword in title_token_set and keyword not in query_token_set:
            return False

    required_tokens: List[str] = []
    for token in query_tokens:
        if token in OPTIONAL_QUERY_TOKENS:
            continue
        if _is_storage_token(token):
            continue
        required_tokens.append(token)

    if not required_tokens:
        required_tokens = [token for token in query_tokens if token not in OPTIONAL_QUERY_TOKENS]

    if not required_tokens:
        return bool(title_token_set & query_token_set)

    return all(token in title_token_set for token in required_tokens)


def normalize_search_query(text: str) -> str:
    """Normalize phone name for retailer search URLs."""
    if not text:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def fetch_zoom_detail_price(url: str, timeout: int = 15) -> Optional[float]:
    """Retrieve product price from a Zoom product detail page."""
    if not url:
        return None

    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        price_elem = soup.select_one('.product-prices .current-price .price')
        if price_elem:
            price_text = price_elem.get_text(strip=True)
            price = extract_price_from_text(price_text)
            if price:
                return price

        data_holder = soup.find(attrs={'data-product': True})
        if data_holder:
            raw_payload = data_holder.get('data-product')
            if raw_payload:
                try:
                    payload = json.loads(raw_payload)
                    if 'price_amount' in payload and payload['price_amount']:
                        try:
                            return float(str(payload['price_amount']).replace(',', '.'))
                        except ValueError:
                            pass
                    if 'price' in payload and payload['price']:
                        guess_price = extract_price_from_text(str(payload['price']))
                        if guess_price:
                            return guess_price
                except json.JSONDecodeError:
                    pass

    except Exception as error:
        print(f"Zoom detail fetch failed: {error}")

    return None


def extract_price_from_text(text: str) -> Optional[float]:
    """
    Extract price from text string (handles TND, DT, etc.)
    Handles formats like: 3,799.000 DT, 3 799,000 TND, 3799 TND
    """
    if not text:
        return None
    
    # Store original for debugging
    original_text = text
    
    # Remove spaces first
    text = text.strip()
    
    # Tunisian price formats:
    # Format 1: "3,799.000 DT" or "3 799.000 DT" (comma/space as thousands separator, dot for decimals)
    # Format 2: "3799.000 TND" (no separator)
    # Format 3: "3 799,000 TND" (space as thousands separator, comma for decimals)
    
    # Try to find price patterns with currency
    patterns = [
        # Pattern: digits with separators + currency (e.g., "3,799.000 DT" or "3 799.000 TND")
        r'([\d\s,\.]+)\s*(?:TND|DT|dt|tnd)',
        # Pattern: just numbers with separators
        r'([\d\s,\.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                price_str = match.group(1)

                # Normalize non-breaking and thin spaces used in Tunisian price formats
                for special_space in ('\u202f', '\xa0', '\u2009', '\u2007', '\u2005'):
                    price_str = price_str.replace(special_space, ' ')

                # Remove all standard whitespace
                price_str = re.sub(r'\s+', '', price_str)
                
                # Determine if comma is decimal separator or thousands separator
                # If there's both comma and dot, the last one is decimal separator
                if ',' in price_str and '.' in price_str:
                    # Find which comes last
                    last_comma = price_str.rfind(',')
                    last_dot = price_str.rfind('.')
                    if last_dot > last_comma:
                        # Dot is decimal separator: "3,799.000" -> remove comma, keep dot
                        price_str = price_str.replace(',', '')
                    else:
                        # Comma is decimal separator: "3.799,000" -> remove dot, replace comma with dot
                        price_str = price_str.replace('.', '').replace(',', '.')
                elif ',' in price_str:
                    # Only comma: could be thousands or decimal
                    # In Tunisian format: "99,000" means 99.000 TND (comma as decimal separator)
                    # But "6499,000" means 6,499 TND (comma as thousands separator... wait no!)
                    # Actually: Both use comma as decimal! "6499,000" = 6499.000 TND
                    # The ".000" part is just displaying 3 decimal places
                    # So comma is ALWAYS decimal separator in Tunisian format
                    price_str = price_str.replace(',', '.')
                # If only dot, keep as is
                
                price = float(price_str)
                
                # Tunisian prices typically range 50-20000 TND
                if 50 < price < 25000:
                    return price
            except ValueError:
                continue
    
    return None

def scrape_tunisianet(phone_name: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """Scrape Tunisianet.com.tn for phone prices"""
    results = []
    try:
        search_query = normalize_search_query(phone_name).replace(' ', '+')
        search_url = TUNISIAN_RETAILERS["tunisianet"]["search_url"].format(query=search_query)
        response = requests.get(search_url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Tunisianet uses <article class="product-miniature">
        products = soup.find_all('article', class_='product-miniature')[:10]
        
        for product in products:
            try:
                # Title: <h2 class="h3 product-title">
                title_elem = product.find('h2', class_='product-title')
                if not title_elem:
                    title_elem = product.find('h2')
                if not title_elem:
                    title_elem = product.find('a')
                title = title_elem.get_text(strip=True) if title_elem else ''
                
                # Price: <span class="price">
                price_elem = product.find('span', class_='price')
                price_text = price_elem.get_text(strip=True) if price_elem else ''
                price = extract_price_from_text(price_text)
                
                # URL: <a href="...">
                link_elem = product.find('a', href=True)
                url = link_elem['href'] if link_elem else ''
                if url and not url.startswith('http'):
                    url = TUNISIAN_RETAILERS["tunisianet"]["base_url"] + url
                
                if price and title and is_relevant_product(title, phone_name):
                    results.append({
                        "retailer": "Tunisianet",
                        "title": title,
                        "price": price,
                        "currency": "TND",
                        "in_stock": True,
                        "condition": "new",
                        "url": url,
                        "shipping": "Livraison disponible"
                    })
            except Exception as e:
                continue
                
    except Exception as e:
        print(f"Tunisianet scraping failed: {e}")
    
    return results


def scrape_zoom(phone_name: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """Scrape Zoom.com.tn for phone prices."""
    results: List[Dict[str, Any]] = []
    detail_price_cache: Dict[str, Optional[float]] = {}

    try:
        search_query = normalize_search_query(phone_name).replace(' ', '+')
        search_url = TUNISIAN_RETAILERS["zoom"]["search_url"].format(query=search_query)
        response = requests.get(search_url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        product_cards = soup.find_all('div', class_='product-miniature')[:10]

        for card in product_cards:
            try:
                article = card.find('article') or card

                title_elem = (
                    article.find('h2')
                    or article.find('h3')
                    or article.find('a', class_='product-title')
                )

                if title_elem:
                    title = title_elem.get_text(strip=True)
                else:
                    img = article.find('img', alt=True)
                    title = img.get('alt', '') if img else ''

                price_elem = article.find('span', class_='price')
                price_text = price_elem.get_text(strip=True) if price_elem else ''
                price = extract_price_from_text(price_text)

                link_elem = article.find('a', href=True)
                url = link_elem['href'] if link_elem else ''
                if url and not url.startswith('http'):
                    url = TUNISIAN_RETAILERS["zoom"]["base_url"] + url

                if (not price) and url:
                    if url not in detail_price_cache:
                        detail_price_cache[url] = fetch_zoom_detail_price(url, timeout=timeout)
                    price = detail_price_cache.get(url)

                if price and title and is_relevant_product(title, phone_name):
                    results.append({
                        "retailer": "Zoom",
                        "title": title,
                        "price": price,
                        "currency": "TND",
                        "in_stock": True,
                        "condition": "new",
                        "url": url,
                        "shipping": "Livraison disponible"
                    })
            except Exception:
                continue

    except requests.Timeout:
        print("Zoom scraping failed: request timed out")
    except Exception as error:
        print(f"Zoom scraping failed: {error}")

    return results


def scrape_spacenet(phone_name: str, timeout: int = 15) -> List[Dict[str, Any]]:
    """Scrape Spacenet.tn for phone prices using cloudscraper."""
    results: List[Dict[str, Any]] = []

    try:
        search_query = normalize_search_query(phone_name).replace(' ', '+')
        search_url = TUNISIAN_RETAILERS["spacenet"]["search_url"].format(query=search_query)

        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )

        response = scraper.get(search_url, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        products = soup.find_all('article', class_='product-miniature')
        if not products:
            products = soup.find_all('div', class_='product-miniature')
        if not products:
            products = soup.find_all('div', class_='product')

        for product in products[:10]:
            try:
                title_elem = (
                    product.find('h2', class_='product-title')
                    or product.find('h2')
                    or product.find('h3')
                    or product.find('a')
                )
                title = title_elem.get_text(strip=True) if title_elem else ''

                price_elem = product.find('span', class_='price')
                price_text = price_elem.get_text(strip=True) if price_elem else ''
                price = extract_price_from_text(price_text)

                link_elem = product.find('a', href=True)
                url = ''
                if link_elem:
                    href = link_elem.get('href', '')
                    url = (
                        TUNISIAN_RETAILERS["spacenet"]["base_url"] + href
                        if href.startswith('/') else href
                    )

                if price and title and is_relevant_product(title, phone_name):
                    results.append({
                        "retailer": "Spacenet",
                        "title": title,
                        "price": price,
                        "currency": "TND",
                        "in_stock": True,
                        "condition": "new",
                        "url": url,
                        "shipping": "Livraison disponible"
                    })
            except Exception:
                continue

    except Exception as error:
        print(f"Spacenet scraping failed: {error}")

    return results

def scrape_tunisian_prices(phone_name: str) -> List[Dict[str, Any]]:
    """
    Scrape all enabled Tunisian retailers for phone prices.
    Returns combined results from all sources.
    """
    all_prices: List[Dict[str, Any]] = []

    scrapers: List[Tuple[str, callable]] = [
        ("tunisianet", scrape_tunisianet),
        ("zoom", scrape_zoom),
        ("spacenet", scrape_spacenet)
    ]

    tasks: List[Tuple[str, callable]] = [
        (retailer_key, scraper_func)
        for retailer_key, scraper_func in scrapers
        if TUNISIAN_RETAILERS[retailer_key]["enabled"]
    ]

    if not tasks:
        return all_prices

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_map = {
            executor.submit(scraper_func, phone_name): retailer_key
            for retailer_key, scraper_func in tasks
        }

        for future in as_completed(future_map):
            retailer_key = future_map[future]
            retailer_name = TUNISIAN_RETAILERS[retailer_key]["name"]
            try:
                print(f"Scraping {retailer_name}...")
                results = future.result()
                all_prices.extend(results)
                print(f"  {retailer_name}: found {len(results)} products")
            except Exception as error:
                print(f"  {retailer_name} scraping failed: {error}")

    return all_prices


def generate_mock_prices(phone_name: str, include_used: bool = False) -> List[Dict[str, Any]]:
    """Deprecated: mock listings disabled to avoid misleading results."""
    return []


def calculate_price_insights(prices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate insights from price data."""
    if not prices:
        return {}
    
    # Filter in-stock new prices for statistics
    new_prices = [p['price'] for p in prices if p['in_stock'] and p['condition'] == 'new']
    all_prices = [p['price'] for p in prices if p['in_stock']]
    
    if not new_prices:
        return {"error": "No in-stock prices found"}
    
    min_price = min(new_prices)
    max_price = max(new_prices)
    avg_price = sum(new_prices) / len(new_prices)
    
    # Find best deal
    best_deal = min([p for p in prices if p['in_stock'] and p['condition'] == 'new'], 
                    key=lambda x: x['price'])
    
    # Calculate savings potential
    savings_potential = max_price - min_price
    savings_percentage = (savings_potential / max_price) * 100 if max_price > 0 else 0
    
    # Price recommendation
    if min_price < avg_price * 0.85:
        recommendation = "Excellent deal available - price is below average"
    elif min_price < avg_price * 0.95:
        recommendation = "Good deal - price is competitive"
    else:
        recommendation = "Consider waiting for better deals or check refurbished options"
    
    return {
        "min_price": min_price,
        "max_price": max_price,
        "avg_price": round(avg_price, 2),
        "price_range": f"${min_price} - ${max_price}",
        "savings_potential": savings_potential,
        "savings_percentage": round(savings_percentage, 1),
        "best_deal": {
            "retailer": best_deal['retailer'],
            "price": best_deal['price'],
            "shipping": best_deal['shipping']
        },
        "recommendation": recommendation,
        "total_retailers": len(new_prices),
        "in_stock_count": len([p for p in prices if p['in_stock']])
    }


def price_extractor_tool(phone_name: str, include_used: bool = False, use_real_scraping: bool = True) -> ToolOutput:
    """
    Main price extraction function.
    
    Args:
        phone_name: Phone model to search for
        include_used: Whether to include used/refurbished prices
        use_real_scraping: If True, scrape Tunisian sites; if False, use mock data
        
    Returns:
        ToolOutput with price comparison data
    """
    start_time = time.time()
    
    try:
        # Try real scraping first (Tunisian retailers)
        prices = []
        data_source = "mock_data"
        
        if use_real_scraping:
            try:
                prices = scrape_tunisian_prices(phone_name)
                if prices:
                    data_source = "tunisian_web_scraping"
                    print(f"✅ Scraped {len(prices)} prices from Tunisian retailers")
            except Exception as scrape_error:
                print(f"⚠️  Web scraping failed: {scrape_error}, falling back to mock data")
        
        if not prices:
            return ToolOutput(
                success=False,
                error=f"No live Tunisian price listings found for {phone_name}",
                metadata={
                    "execution_time": round(time.time() - start_time, 3),
                    "retailers_checked": sum(1 for cfg in TUNISIAN_RETAILERS.values() if cfg.get("enabled", True)),
                    "data_source": "tunisian_web_scraping",
                    "real_scraping_enabled": use_real_scraping,
                }
            )
        
        # Calculate insights
        insights = calculate_price_insights(prices)
        
        # Sort prices by price (ascending)
        prices_sorted = sorted(prices, key=lambda x: x['price'])
        
        execution_time = time.time() - start_time
        
        # Determine currency from results
        currency = prices[0].get('currency', 'TND') if prices else 'TND'
        
        return ToolOutput(
            success=True,
            data={
                "phone_name": phone_name,
                "prices": prices_sorted,
                "insights": insights,
                "currency": currency,
                "includes_used": include_used,
                "retailers_found": len(set(p['retailer'] for p in prices))
            },
            metadata={
                "execution_time": round(execution_time, 3),
                "retailers_checked": len(prices),
                "data_source": "tunisian_web_scraping",
                "real_scraping_enabled": use_real_scraping
            }
        )
    
    except Exception as e:
        return ToolOutput(
            success=False,
            error=f"Price extraction failed: {str(e)}",
            metadata={"execution_time": round(time.time() - start_time, 3)}
        )


# MCP Tool Schema
PRICE_EXTRACTOR_SCHEMA = {
    "name": "price_extractor",
    "description": "Scrapes live prices from Tunisian tech retailers (Tunisianet, Zoom, Spacenet) and returns structured pricing insights in TND.",
    "input_schema": {
        "type": "object",
        "properties": {
            "phone_name": {
                "type": "string",
                "description": "Phone model name to search for"
            },
            "include_used": {
                "type": "boolean",
                "description": "Whether to include used/refurbished prices",
                "default": False
            },
            "use_real_scraping": {
                "type": "boolean",
                "description": "If true, scrape Tunisian retailers; if false, use mock data",
                "default": True
            }
        },
        "required": ["phone_name"]
    },
    "version": "2.1.0"
}
