"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

"""Spider skill for fast web scraping with SignalWire AI Agents."""
import re
import logging
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
import requests
from lxml import html
from lxml.etree import XPathEvalError

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult


class SpiderSkill(SkillBase):
    """Fast web scraping skill optimized for speed and token efficiency."""
    
    SKILL_NAME = "spider"
    SKILL_DESCRIPTION = "Fast web scraping and crawling capabilities"
    SKILL_VERSION = "1.0.0"
    REQUIRED_PACKAGES = ["lxml"]  # beautifulsoup4 and requests are in base dependencies
    REQUIRED_ENV_VARS = []  # No required env vars by default
    SUPPORTS_MULTIPLE_INSTANCES = True
    
    # Compiled regex for performance
    WHITESPACE_REGEX = re.compile(r'\s+')
    
    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for Spider skill"""
        schema = super().get_parameter_schema()
        schema.update({
            "delay": {
                "type": "number",
                "description": "Delay between requests in seconds",
                "default": 0.1,
                "required": False,
                "minimum": 0.0
            },
            "concurrent_requests": {
                "type": "integer",
                "description": "Number of concurrent requests allowed",
                "default": 5,
                "required": False,
                "minimum": 1,
                "maximum": 20
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds",
                "default": 5,
                "required": False,
                "minimum": 1,
                "maximum": 60
            },
            "max_pages": {
                "type": "integer",
                "description": "Maximum number of pages to scrape",
                "default": 1,
                "required": False,
                "minimum": 1,
                "maximum": 100
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum crawl depth (0 = single page only)",
                "default": 0,
                "required": False,
                "minimum": 0,
                "maximum": 5
            },
            "extract_type": {
                "type": "string",
                "description": "Content extraction method",
                "default": "fast_text",
                "required": False,
                "enum": ["fast_text", "clean_text", "full_text", "html", "custom"]
            },
            "max_text_length": {
                "type": "integer",
                "description": "Maximum text length to return",
                "default": 10000,
                "required": False,
                "minimum": 100,
                "maximum": 100000
            },
            "clean_text": {
                "type": "boolean",
                "description": "Whether to clean extracted text",
                "default": True,
                "required": False
            },
            "selectors": {
                "type": "object",
                "description": "Custom CSS/XPath selectors for extraction",
                "default": {},
                "required": False,
                "additionalProperties": {
                    "type": "string"
                }
            },
            "follow_patterns": {
                "type": "array",
                "description": "URL patterns to follow when crawling",
                "default": [],
                "required": False,
                "items": {
                    "type": "string"
                }
            },
            "user_agent": {
                "type": "string",
                "description": "User agent string for requests",
                "default": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "required": False
            },
            "headers": {
                "type": "object",
                "description": "Additional HTTP headers",
                "default": {},
                "required": False,
                "additionalProperties": {
                    "type": "string"
                }
            },
            "follow_robots_txt": {
                "type": "boolean",
                "description": "Whether to respect robots.txt",
                "default": True,
                "required": False
            },
            "cache_enabled": {
                "type": "boolean",
                "description": "Whether to cache scraped pages",
                "default": True,
                "required": False
            }
        })
        return schema
    
    def __init__(self, agent, params: Dict[str, Any]):
        """Initialize the spider skill with configuration parameters."""
        super().__init__(agent, params)
        
        # Performance settings
        self.delay = self.params.get('delay', 0.1)
        self.concurrent_requests = self.params.get('concurrent_requests', 5)
        self.timeout = self.params.get('timeout', 5)
        
        # Crawling limits
        self.max_pages = self.params.get('max_pages', 1)
        self.max_depth = self.params.get('max_depth', 0)
        
        # Content processing
        self.extract_type = self.params.get('extract_type', 'fast_text')
        self.max_text_length = self.params.get('max_text_length', 3000)
        self.clean_text = self.params.get('clean_text', True)
        
        # Features
        self.cache_enabled = self.params.get('cache_enabled', True)
        self.follow_robots_txt = self.params.get('follow_robots_txt', False)
        self.user_agent = self.params.get('user_agent', 'Spider/1.0 (SignalWire AI Agent)')
        
        # Optional headers
        self.headers = self.params.get('headers', {})
        self.headers['User-Agent'] = self.user_agent
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Cache for responses
        self.cache = {} if self.cache_enabled else None
        
        # XPath expressions for unwanted elements
        self.remove_xpaths = [
            '//script', '//style', '//nav', '//header', 
            '//footer', '//aside', '//noscript'
        ]
    
    def get_instance_key(self) -> str:
        """Return unique key for this skill instance."""
        tool_name = self.params.get('tool_name', self.SKILL_NAME)
        return f"{self.SKILL_NAME}_{tool_name}"
    
    def setup(self) -> bool:
        """Validate configuration and setup the skill."""
        # Validate delay is reasonable
        if self.delay < 0:
            self.logger.error("Delay cannot be negative")
            return False
            
        # Validate concurrent requests
        if not 1 <= self.concurrent_requests <= 20:
            self.logger.error("Concurrent requests must be between 1 and 20")
            return False
            
        # Validate max pages and depth
        if self.max_pages < 1:
            self.logger.error("Max pages must be at least 1")
            return False
            
        if self.max_depth < 0:
            self.logger.error("Max depth cannot be negative")
            return False
            
        self.logger.info(f"Spider skill configured: delay={self.delay}s, max_pages={self.max_pages}, max_depth={self.max_depth}")
        return True
    
    def register_tools(self) -> None:
        """Register the web scraping tools with the agent."""
        # Tool name prefix for multiple instances
        tool_prefix = self.params.get('tool_name', '')
        if tool_prefix:
            tool_prefix = f"{tool_prefix}_"
        
        # Register scrape_url tool
        self.define_tool(
            name=f"{tool_prefix}scrape_url",
            description="Extract text content from a single web page",
            parameters={
                "url": {
                    "type": "string",
                    "description": "The URL to scrape"
                }
            },
            required=["url"],
            handler=self._scrape_url_handler
        )
        
        # Register crawl_site tool
        self.define_tool(
            name=f"{tool_prefix}crawl_site",
            description="Crawl multiple pages starting from a URL",
            parameters={
                "start_url": {
                    "type": "string",
                    "description": "Starting URL for the crawl"
                }
            },
            required=["start_url"],
            handler=self._crawl_site_handler
        )
        
        # Register extract_structured_data tool
        self.define_tool(
            name=f"{tool_prefix}extract_structured_data",
            description="Extract specific data from a web page using selectors",
            parameters={
                "url": {
                    "type": "string",
                    "description": "The URL to scrape"
                }
            },
            required=["url"],
            handler=self._extract_structured_handler
        )
    
    def _fetch_url(self, url: str) -> Optional[requests.Response]:
        """Fetch a URL with caching and error handling."""
        # Check cache first
        if self.cache_enabled and url in self.cache:
            self.logger.debug(f"Cache hit for {url}")
            return self.cache[url]
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Cache successful responses
            if self.cache_enabled:
                self.cache[url] = response
                
            return response
            
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout fetching {url}")
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _fast_text_extract(self, response: requests.Response) -> str:
        """Ultra-fast text extraction using lxml."""
        try:
            # Parse HTML with lxml
            tree = html.fromstring(response.content)
            
            # Remove unwanted elements in one pass
            for xpath in self.remove_xpaths:
                for elem in tree.xpath(xpath):
                    elem.drop_tree()
            
            # Extract text
            text = tree.text_content()
            
            # Clean whitespace if requested
            if self.clean_text:
                text = self.WHITESPACE_REGEX.sub(' ', text).strip()
            
            # Smart truncation
            if len(text) > self.max_text_length:
                keep_start = self.max_text_length * 2 // 3
                keep_end = self.max_text_length // 3
                text = (
                    text[:keep_start] + 
                    "\n\n[...CONTENT TRUNCATED...]\n\n" + 
                    text[-keep_end:]
                )
            
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return ""
    
    def _markdown_extract(self, response: requests.Response) -> str:
        """Extract content in markdown format."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted tags
            for tag in ['script', 'style', 'nav', 'header', 'footer', 'aside']:
                for elem in soup.find_all(tag):
                    elem.decompose()
            
            # Convert to markdown-like format
            text_parts = []
            
            # Title
            title = soup.find('title')
            if title:
                text_parts.append(f"# {title.get_text().strip()}\n")
            
            # Main content
            for elem in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'code', 'pre']):
                if elem.name.startswith('h'):
                    level = int(elem.name[1])
                    text_parts.append(f"\n{'#' * level} {elem.get_text().strip()}\n")
                elif elem.name == 'p':
                    text_parts.append(f"\n{elem.get_text().strip()}\n")
                elif elem.name == 'li':
                    text_parts.append(f"- {elem.get_text().strip()}")
                elif elem.name in ['code', 'pre']:
                    text_parts.append(f"\n```\n{elem.get_text().strip()}\n```\n")
            
            text = '\n'.join(text_parts)
            
            # Truncate if needed
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length] + "\n\n[...TRUNCATED...]"
            
            return text
            
        except ImportError:
            self.logger.warning("BeautifulSoup not available, falling back to fast_text")
            return self._fast_text_extract(response)
        except Exception as e:
            self.logger.error(f"Error in markdown extraction: {e}")
            return self._fast_text_extract(response)
    
    def _structured_extract(self, response: requests.Response, selectors: Dict[str, str] = None) -> Dict[str, Any]:
        """Extract structured data using selectors."""
        try:
            tree = html.fromstring(response.content)
            result = {
                "url": response.url,
                "status_code": response.status_code,
                "title": "",
                "data": {}
            }
            
            # Get title
            title_elem = tree.xpath('//title/text()')
            if title_elem:
                result["title"] = title_elem[0].strip()
            
            # Extract using provided selectors
            if selectors:
                for field, selector in selectors.items():
                    try:
                        if selector.startswith('/'):  # XPath
                            values = tree.xpath(selector)
                        else:  # CSS selector
                            from lxml.cssselect import CSSSelector
                            sel = CSSSelector(selector)
                            values = sel(tree)
                        
                        # Extract text from elements
                        if values:
                            if len(values) == 1:
                                result["data"][field] = values[0].text_content().strip()
                            else:
                                result["data"][field] = [v.text_content().strip() for v in values]
                    except (XPathEvalError, Exception) as e:
                        self.logger.warning(f"Error with selector {selector}: {e}")
                        result["data"][field] = None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in structured extraction: {e}")
            return {"error": str(e)}
    
    def _scrape_url_handler(self, args: Dict[str, Any], raw_data: Dict[str, Any]) -> SwaigFunctionResult:
        """Handle single page scraping."""
        url = args.get('url', '').strip()
        if not url:
            return SwaigFunctionResult("Please provide a URL to scrape")
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return SwaigFunctionResult(f"Invalid URL: {url}")
        
        # Fetch the page
        response = self._fetch_url(url)
        if not response:
            return SwaigFunctionResult(f"Failed to fetch {url}")
        
        # Extract content based on configured type (not from args)
        extract_type = self.extract_type
        
        try:
            if extract_type == 'structured':
                # For structured extraction, use predefined selectors from config if available
                selectors = self.params.get('selectors', {})
                result = self._structured_extract(response, selectors)
                return SwaigFunctionResult(f"Extracted structured data from {url}: {result}")
            elif extract_type == 'markdown':
                content = self._markdown_extract(response)
            else:  # fast_text (default)
                content = self._fast_text_extract(response)
            
            if not content:
                return SwaigFunctionResult(f"No content extracted from {url}")
            
            # Format response
            char_count = len(content)
            header = f"Content from {url} ({char_count} characters):\n\n"
            
            return SwaigFunctionResult(header + content)
            
        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            return SwaigFunctionResult(f"Error processing {url}: {str(e)}")
    
    def _crawl_site_handler(self, args: Dict[str, Any], raw_data: Dict[str, Any]) -> SwaigFunctionResult:
        """Handle multi-page crawling."""
        start_url = args.get('start_url', '').strip()
        if not start_url:
            return SwaigFunctionResult("Please provide a starting URL for the crawl")
        
        # Use configured parameters (not from args)
        max_depth = self.max_depth
        max_pages = self.max_pages
        follow_patterns = self.params.get('follow_patterns', [])
        
        # Validate parameters
        if max_depth < 0:
            return SwaigFunctionResult("Max depth cannot be negative")
        if max_pages < 1:
            return SwaigFunctionResult("Max pages must be at least 1")
        
        # Simple breadth-first crawl
        visited = set()
        to_visit = [(start_url, 0)]  # (url, depth)
        results = []
        
        while to_visit and len(visited) < max_pages:
            if not to_visit:
                break
                
            url, depth = to_visit.pop(0)
            
            # Skip if already visited or depth exceeded
            if url in visited or depth > max_depth:
                continue
            
            # Fetch and process page
            response = self._fetch_url(url)
            if not response:
                continue
            
            visited.add(url)
            
            # Extract content
            content = self._fast_text_extract(response)
            if content:
                results.append({
                    'url': url,
                    'depth': depth,
                    'content_length': len(content),
                    'summary': content[:500] + '...' if len(content) > 500 else content
                })
            
            # Extract links if not at max depth
            if depth < max_depth:
                try:
                    tree = html.fromstring(response.content)
                    links = tree.xpath('//a[@href]/@href')
                    
                    for link in links:
                        absolute_url = urljoin(url, link)
                        
                        # Check if we should follow this link
                        if follow_patterns:
                            if not any(re.search(pattern, absolute_url) for pattern in follow_patterns):
                                continue
                        
                        # Only follow same domain by default
                        if urlparse(absolute_url).netloc == urlparse(start_url).netloc:
                            if absolute_url not in visited:
                                to_visit.append((absolute_url, depth + 1))
                                
                except Exception as e:
                    self.logger.warning(f"Error extracting links from {url}: {e}")
            
            # Respect delay between requests
            if self.delay > 0 and len(visited) < max_pages:
                import time
                time.sleep(self.delay)
        
        # Format results
        if not results:
            return SwaigFunctionResult(f"No pages could be crawled from {start_url}")
        
        summary = f"Crawled {len(results)} pages from {urlparse(start_url).netloc}:\n\n"
        
        for i, result in enumerate(results, 1):
            summary += f"{i}. {result['url']} (depth: {result['depth']}, {result['content_length']} chars)\n"
            summary += f"   Summary: {result['summary'][:100]}...\n\n"
        
        total_chars = sum(r['content_length'] for r in results)
        summary += f"\nTotal content: {total_chars:,} characters across {len(results)} pages"
        
        return SwaigFunctionResult(summary)
    
    def _extract_structured_handler(self, args: Dict[str, Any], raw_data: Dict[str, Any]) -> SwaigFunctionResult:
        """Handle structured data extraction."""
        url = args.get('url', '').strip()
        
        if not url:
            return SwaigFunctionResult("Please provide a URL")
        
        # Use configured selectors from params
        selectors = self.params.get('selectors', {})
        if not selectors:
            return SwaigFunctionResult("No selectors configured for structured data extraction")
        
        # Fetch the page
        response = self._fetch_url(url)
        if not response:
            return SwaigFunctionResult(f"Failed to fetch {url}")
        
        # Extract structured data
        result = self._structured_extract(response, selectors)
        
        if 'error' in result:
            return SwaigFunctionResult(f"Error extracting data: {result['error']}")
        
        # Format the response
        output = f"Extracted data from {url}:\n\n"
        output += f"Title: {result.get('title', 'N/A')}\n\n"
        
        if result.get('data'):
            output += "Data:\n"
            for field, value in result['data'].items():
                output += f"- {field}: {value}\n"
        else:
            output += "No data extracted with provided selectors"
        
        return SwaigFunctionResult(output)
    
    def get_hints(self) -> List[str]:
        """Return speech recognition hints for this skill."""
        return [
            "scrape", "crawl", "extract", "web page", "website",
            "get content from", "fetch data from", "spider"
        ]
    
    def cleanup(self) -> None:
        """Clean up resources when skill is unloaded."""
        if hasattr(self, 'session'):
            self.session.close()
        if hasattr(self, 'cache'):
            self.cache.clear()
        self.logger.info("Spider skill cleaned up")