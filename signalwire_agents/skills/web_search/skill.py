"""
Copyright (c) 2025 SignalWire

This file is part of the SignalWire AI Agents SDK.

Licensed under the MIT License.
See LICENSE file in the project root for full license information.
"""

import os
import requests
import time
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from typing import Optional, List, Dict, Any, Tuple

from signalwire_agents.core.skill_base import SkillBase
from signalwire_agents.core.function_result import SwaigFunctionResult

class GoogleSearchScraper:
    """Google Search and Web Scraping functionality with quality scoring"""

    def __init__(self, api_key: str, search_engine_id: str, max_content_length: int = 32768):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.max_content_length = max_content_length
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def search_google(self, query: str, num_results: int = 5) -> list:
        """Search Google using Custom Search JSON API"""
        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': query,
            'num': min(num_results, 10)
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            if 'items' not in data:
                return []

            results = []
            for item in data['items'][:num_results]:
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })

            return results

        except Exception as e:
            return []

    def is_reddit_url(self, url: str) -> bool:
        """Check if URL is from Reddit"""
        domain = urlparse(url).netloc.lower()
        return 'reddit.com' in domain or 'redd.it' in domain

    def extract_reddit_content(self, url: str, content_limit: int = None, timeout: int = 10) -> Tuple[str, Dict[str, Any]]:
        """
        Extract Reddit content using JSON API for better quality

        Returns:
            Tuple of (text_content, quality_metrics)
        """
        try:
            # Convert to JSON endpoint
            if not url.endswith('.json'):
                json_url = url.rstrip('/') + '.json'
            else:
                json_url = url

            # Fetch with proper headers (Reddit requires User-Agent)
            headers = {'User-Agent': 'SignalWire-WebSearch/2.0'}
            response = requests.get(json_url, headers=headers, timeout=timeout)
            response.raise_for_status()

            data = response.json()

            # Extract post information
            if not data or not isinstance(data, list) or len(data) < 1:
                return "", {"error": "Invalid Reddit JSON structure", "quality_score": 0}

            # First element is the post, second (if exists) contains comments
            post_data = data[0]['data']['children'][0]['data']

            # Build content from post
            content_parts = []

            # Add post title and metadata
            title = post_data.get('title', 'No title')
            author = post_data.get('author', 'unknown')
            score = post_data.get('score', 0)
            num_comments = post_data.get('num_comments', 0)
            subreddit = post_data.get('subreddit', '')

            content_parts.append(f"Reddit r/{subreddit} Discussion")
            content_parts.append(f"\nPost: {title}")
            content_parts.append(f"Author: {author} | Score: {score} | Comments: {num_comments}")

            # Add original post text if it's a text post
            selftext = post_data.get('selftext', '').strip()
            if selftext and selftext != '[removed]' and selftext != '[deleted]':
                content_parts.append(f"\nOriginal Post:\n{selftext[:1000]}")  # Limit post text

            # Extract top comments if available
            valid_comments = []
            if len(data) > 1 and 'data' in data[1] and 'children' in data[1]['data']:
                comments = data[1]['data']['children']

                # Filter and sort comments by score
                for comment in comments[:20]:  # Look at top 20 comments
                    if comment.get('kind') == 't1':  # t1 = comment
                        comment_data = comment.get('data', {})
                        body = comment_data.get('body', '').strip()
                        if (body and
                            body != '[removed]' and
                            body != '[deleted]' and
                            len(body) > 50):  # Skip very short comments
                            valid_comments.append({
                                'body': body,
                                'author': comment_data.get('author', 'unknown'),
                                'score': comment_data.get('score', 0)
                            })

                # Sort by score and take top comments
                valid_comments.sort(key=lambda x: x['score'], reverse=True)

                if valid_comments:
                    content_parts.append("\n--- Top Discussion ---")
                    for i, comment in enumerate(valid_comments[:5], 1):
                        # Truncate long comments
                        comment_text = comment['body'][:500]
                        if len(comment['body']) > 500:
                            comment_text += "..."

                        content_parts.append(f"\nComment {i} (Score: {comment['score']}, Author: {comment['author']}):")
                        content_parts.append(comment_text)

            # Join all content
            text = '\n'.join(content_parts)

            # Calculate quality metrics specifically for Reddit content
            metrics = {
                'text_length': len(text),
                'score': score,
                'num_comments': num_comments,
                'domain': urlparse(url).netloc.lower(),
                'is_reddit': True
            }

            # Quality score based on Reddit-specific factors
            length_score = min(1.0, len(text) / 2000)  # Want at least 2000 chars
            engagement_score = min(1.0, (score + num_comments) / 100)  # High engagement is good
            has_comments = 1.0 if len(valid_comments) > 0 else 0.3  # Heavily penalize if no good comments

            quality_score = (
                length_score * 0.4 +
                engagement_score * 0.3 +
                has_comments * 0.3
            )

            metrics['quality_score'] = round(quality_score, 3)

            # Limit content if needed
            limit = content_limit if content_limit is not None else self.max_content_length
            if len(text) > limit:
                text = text[:limit]

            return text, metrics

        except Exception as e:
            # Fall back to HTML extraction if JSON fails
            return self.extract_html_content(url, content_limit, timeout)

    def extract_text_from_url(self, url: str, content_limit: int = None, timeout: int = 10) -> Tuple[str, Dict[str, Any]]:
        """
        Main extraction method that routes to appropriate extractor

        Returns:
            Tuple of (text_content, quality_metrics)
        """
        if self.is_reddit_url(url):
            return self.extract_reddit_content(url, content_limit, timeout)
        else:
            return self.extract_html_content(url, content_limit, timeout)

    def extract_html_content(self, url: str, content_limit: int = None, timeout: int = 10) -> Tuple[str, Dict[str, Any]]:
        """
        Original HTML extraction method (renamed from extract_text_from_url)
        """
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract main content areas (common content selectors)
            main_content = None
            content_selectors = [
                'article', 'main', '[role="main"]', '.content', '#content',
                '.post', '.entry-content', '.article-body', '.story-body',
                '.markdown-body', '.wiki-body', '.documentation'
            ]

            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            # If no main content found, use the whole body
            if not main_content:
                main_content = soup.find('body') or soup

            # Clone for processing
            content_soup = BeautifulSoup(str(main_content), 'html.parser')

            # Remove unwanted elements from the content area
            unwanted_tags = ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]
            for tag in unwanted_tags:
                for element in content_soup(tag):
                    element.decompose()

            # Remove elements with unwanted classes/ids
            unwanted_patterns = [
                'sidebar', 'navigation', 'menu', 'advertisement', 'ads', 'banner',
                'popup', 'modal', 'cookie', 'gdpr', 'subscribe', 'newsletter',
                'comments', 'related', 'share', 'social'
            ]

            for pattern in unwanted_patterns:
                # Remove by class
                for element in content_soup.find_all(class_=re.compile(pattern, re.I)):
                    element.decompose()
                # Remove by id
                for element in content_soup.find_all(id=re.compile(pattern, re.I)):
                    element.decompose()

            # Extract text
            text = content_soup.get_text()

            # Clean up the text
            lines = [line.strip() for line in text.splitlines()]
            # Remove empty lines and join
            lines = [line for line in lines if line]
            text = ' '.join(lines)

            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()

            # Calculate quality metrics (need to pass query for relevance)
            quality_metrics = self._calculate_content_quality(text, url, "")

            # Limit text length
            limit = content_limit if content_limit is not None else self.max_content_length
            if len(text) > limit:
                text = text[:limit]

            return text, quality_metrics

        except Exception as e:
            return "", {"error": str(e), "quality_score": 0}

    def _calculate_content_quality(self, text: str, url: str, query: str = "") -> Dict[str, Any]:
        """
        Calculate quality metrics for extracted content

        Quality factors:
        - Text length (substantive content)
        - Word diversity (not repetitive)
        - Sentence structure (proper formatting)
        - Lack of boilerplate phrases
        - Domain reputation
        """
        if not text:
            return {"quality_score": 0, "text_length": 0}

        metrics = {}

        # Text length (MUCH stricter - prefer 2000-10000 chars of actual content)
        text_length = len(text)
        metrics['text_length'] = text_length
        if text_length < 500:
            length_score = 0  # Too short to be useful
        elif text_length < 2000:
            length_score = (text_length - 500) / 1500 * 0.5  # Scale to 0.5 max
        elif text_length <= 10000:
            length_score = 1.0  # Ideal range
        else:
            length_score = max(0.8, 1.0 - (text_length - 10000) / 20000)

        # Word diversity (unique words / total words)
        words = text.lower().split()
        if words:
            unique_words = len(set(words))
            total_words = len(words)
            diversity_score = min(1.0, unique_words / (total_words * 0.3))  # Expect 30% unique
            metrics['word_diversity'] = unique_words / total_words if total_words > 0 else 0
        else:
            diversity_score = 0
            metrics['word_diversity'] = 0

        # Check for boilerplate/navigation text (MUCH stricter)
        boilerplate_phrases = [
            'cookie', 'privacy policy', 'terms of service', 'subscribe',
            'sign up', 'log in', 'advertisement', 'sponsored', 'copyright',
            'all rights reserved', 'skip to', 'navigation', 'breadcrumb',
            'reddit inc', 'google llc', 'expand navigation', 'members •',
            'archived post', 'votes cannot be cast', 'r/', 'subreddit',
            'youtube', 'facebook', 'twitter', 'instagram', 'pinterest'
        ]

        lower_text = text.lower()
        boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in lower_text)
        boilerplate_penalty = max(0, 1.0 - (boilerplate_count * 0.15))  # -15% per boilerplate phrase
        metrics['boilerplate_count'] = boilerplate_count

        # Sentence detection (need MORE sentences for quality content)
        sentences = re.split(r'[.!?]+', text)
        sentence_count = len([s for s in sentences if len(s.strip()) > 30])  # Longer min sentence
        sentence_score = min(1.0, sentence_count / 10)  # Need at least 10 proper sentences
        metrics['sentence_count'] = sentence_count

        # Domain quality (MUCH harsher on social media)
        domain = urlparse(url).netloc.lower()
        quality_domains = [
            'wikipedia.org', 'starwars.fandom.com', 'imdb.com',
            'screenrant.com', 'denofgeek.com', 'ign.com',
            'hollywoodreporter.com', 'variety.com', 'ew.com',
            'stackexchange.com', 'stackoverflow.com',
            'github.com', 'medium.com', 'dev.to', 'arxiv.org',
            'nature.com', 'sciencedirect.com', 'ieee.org'
        ]

        low_quality_domains = [
            'reddit.com', 'youtube.com', 'facebook.com', 'twitter.com',
            'instagram.com', 'pinterest.com', 'tiktok.com', 'x.com'
        ]

        if any(d in domain for d in quality_domains):
            domain_score = 1.5  # Strong bonus for quality domains
        elif any(d in domain for d in low_quality_domains):
            domain_score = 0.1  # Severe penalty for social media
        else:
            domain_score = 1.0

        metrics['domain'] = domain

        # Query relevance scoring - check for query terms in content
        relevance_score = 0
        if query:
            # Split query into meaningful words (skip common words)
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'}
            query_words = [w.lower() for w in query.split() if w.lower() not in stop_words and len(w) > 2]

            if query_words:
                lower_content = text.lower()
                # Count how many query words appear in content
                words_found = sum(1 for word in query_words if word in lower_content)

                # Also check for exact phrase matches (bonus points)
                exact_phrase_bonus = 0
                if len(query_words) > 1:
                    # Check for consecutive word pairs
                    for i in range(len(query_words) - 1):
                        phrase = f"{query_words[i]} {query_words[i+1]}"
                        if phrase in lower_content:
                            exact_phrase_bonus += 0.2

                relevance_score = min(1.0, (words_found / len(query_words)) + exact_phrase_bonus)
                metrics['query_relevance'] = round(relevance_score, 3)
                metrics['query_words_found'] = f"{words_found}/{len(query_words)}"
        else:
            relevance_score = 0.5  # Neutral if no query provided
            metrics['query_relevance'] = 0.5

        # Calculate final quality score (adjusted weights with relevance)
        quality_score = (
            length_score * 0.25 +      # Length important
            diversity_score * 0.10 +    # Less weight
            boilerplate_penalty * 0.10 + # Less weight
            sentence_score * 0.15 +     # Sentences important
            domain_score * 0.15 +       # Domain important
            relevance_score * 0.25      # Query relevance important
        )

        metrics['quality_score'] = round(quality_score, 3)
        metrics['length_score'] = round(length_score, 3)
        metrics['diversity_score'] = round(diversity_score, 3)
        metrics['boilerplate_penalty'] = round(boilerplate_penalty, 3)
        metrics['sentence_score'] = round(sentence_score, 3)
        metrics['domain_score'] = round(domain_score, 3)

        return metrics

    def search_and_scrape_best(self, query: str, num_results: int = 3,
                               oversample_factor: float = 4.0, delay: float = 0.5,
                               min_quality_score: float = 0.2) -> str:
        """
        Search and scrape with quality filtering and source diversity

        Args:
            query: Search query
            num_results: Number of best results to return
            oversample_factor: How many extra results to fetch (e.g., 4.0 = fetch 4x)
            delay: Delay between requests
            min_quality_score: Minimum quality score to include a result

        Returns:
            Formatted string with the best N results from diverse sources
        """
        # Fetch more results than requested (increased to 4x for better pool)
        fetch_count = min(10, int(num_results * oversample_factor))
        search_results = self.search_google(query, fetch_count)

        if not search_results:
            return f"No search results found for query: {query}"

        # Process all results and collect quality metrics
        processed_results = []

        for result in search_results:
            # Extract content and quality metrics (pass query for relevance scoring)
            page_text, metrics = self.extract_text_from_url(result['url'])

            # Recalculate metrics with query relevance
            if page_text:
                metrics = self._calculate_content_quality(page_text, result['url'], query)

            if metrics.get('quality_score', 0) >= min_quality_score and page_text:
                processed_results.append({
                    'title': result['title'],
                    'url': result['url'],
                    'snippet': result['snippet'],
                    'content': page_text,
                    'metrics': metrics,
                    'quality_score': metrics.get('quality_score', 0),
                    'domain': metrics.get('domain', '')
                })

            # Small delay between requests
            if delay > 0:
                time.sleep(delay)

        if not processed_results:
            return f"No quality results found for query: {query}. All results were below quality threshold."

        # Sort by quality score
        processed_results.sort(key=lambda x: x['quality_score'], reverse=True)

        # Select diverse results (prefer different domains)
        best_results = []
        seen_domains = set()

        # First pass: Add highest quality result from each unique domain
        for result in processed_results:
            domain = result['domain']
            if domain not in seen_domains and len(best_results) < num_results:
                best_results.append(result)
                seen_domains.add(domain)

        # Second pass: If we need more results, add remaining high-quality ones
        if len(best_results) < num_results:
            for result in processed_results:
                if result not in best_results and len(best_results) < num_results:
                    best_results.append(result)

        if not best_results:
            return f"No quality results found for query: {query}. Try a different search term."

        # Calculate per-result content budget for the best results
        estimated_overhead_per_result = 400  # Including quality info
        total_overhead = len(best_results) * estimated_overhead_per_result
        available_for_content = self.max_content_length - total_overhead
        per_result_limit = max(2000, available_for_content // len(best_results))  # Increased minimum

        # Format the best results
        all_text = []
        all_text.append(f"Found {len(processed_results)} results meeting quality threshold from {len(search_results)} searched.")
        all_text.append(f"Showing top {len(best_results)} from diverse sources:\n")

        for i, result in enumerate(best_results, 1):
            text_content = f"=== RESULT {i} (Quality: {result['quality_score']:.2f}) ===\n"
            text_content += f"Title: {result['title']}\n"
            text_content += f"URL: {result['url']}\n"
            text_content += f"Source: {result['domain']}\n"
            text_content += f"Snippet: {result['snippet']}\n"

            # Add quality indicators
            metrics = result['metrics']
            text_content += f"Content Stats: {metrics.get('text_length', 0)} chars, "
            text_content += f"{metrics.get('sentence_count', 0)} sentences\n"
            text_content += f"Query Relevance: {metrics.get('query_relevance', 0):.2f} "
            text_content += f"(keywords: {metrics.get('query_words_found', 'N/A')})\n"
            text_content += f"Content:\n"

            # Truncate content if needed
            content = result['content']
            if len(content) > per_result_limit:
                content = content[:per_result_limit] + "..."
            text_content += content

            text_content += f"\n{'='*50}\n\n"
            all_text.append(text_content)

        return '\n'.join(all_text)

    def search_and_scrape(self, query: str, num_results: int = 3, delay: float = 0.5) -> str:
        """
        Backward compatible method that uses the improved search
        """
        return self.search_and_scrape_best(
            query=query,
            num_results=num_results,
            oversample_factor=4.0,
            delay=delay,
            min_quality_score=0.2
        )


class WebSearchSkill(SkillBase):
    """Web search capability using Google Custom Search API with quality filtering"""

    SKILL_NAME = "web_search"
    SKILL_DESCRIPTION = "Search the web for information using Google Custom Search API"
    SKILL_VERSION = "2.0.0"  # Bumped version for improved functionality
    REQUIRED_PACKAGES = ["bs4", "requests"]
    REQUIRED_ENV_VARS = []

    # Enable multiple instances support
    SUPPORTS_MULTIPLE_INSTANCES = True

    def get_instance_key(self) -> str:
        """Get the key used to track this skill instance"""
        search_engine_id = self.params.get('search_engine_id', 'default')
        tool_name = self.params.get('tool_name', 'web_search')
        return f"{self.SKILL_NAME}_{search_engine_id}_{tool_name}"

    def setup(self) -> bool:
        """Setup the web search skill"""
        # Validate required parameters
        required_params = ['api_key', 'search_engine_id']
        missing_params = [param for param in required_params if not self.params.get(param)]
        if missing_params:
            self.logger.error(f"Missing required parameters: {missing_params}")
            return False

        if not self.validate_packages():
            return False

        # Set parameters from config
        self.api_key = self.params['api_key']
        self.search_engine_id = self.params['search_engine_id']

        # Set default parameters
        self.default_num_results = self.params.get('num_results', 3)
        self.default_delay = self.params.get('delay', 0.5)
        self.max_content_length = self.params.get('max_content_length', 32768)

        # Quality control parameters (new)
        self.oversample_factor = self.params.get('oversample_factor', 2.5)
        self.min_quality_score = self.params.get('min_quality_score', 0.3)

        self.no_results_message = self.params.get('no_results_message',
            "I couldn't find quality results for '{query}'. "
            "The search returned only low-quality or inaccessible pages. "
            "Try rephrasing your search or asking about a different topic."
        )

        # Tool name (for multiple instances)
        self.tool_name = self.params.get('tool_name', 'web_search')

        # Initialize the improved search scraper
        self.search_scraper = GoogleSearchScraper(
            api_key=self.api_key,
            search_engine_id=self.search_engine_id,
            max_content_length=self.max_content_length
        )

        return True

    def register_tools(self) -> None:
        """Register web search tool with the agent"""
        self.define_tool(
            name=self.tool_name,
            description="Search the web for high-quality information, automatically filtering low-quality results",
            parameters={
                "query": {
                    "type": "string",
                    "description": "The search query - what you want to find information about"
                }
            },
            handler=self._web_search_handler
        )

    def _web_search_handler(self, args, raw_data):
        """Handler for web search tool with quality filtering"""
        query = args.get("query", "").strip()

        if not query:
            return SwaigFunctionResult(
                "Please provide a search query. What would you like me to search for?"
            )

        # Use the configured number of results
        num_results = self.default_num_results

        self.logger.info(f"Web search requested: '{query}' (requesting {num_results} quality results)")

        # Perform the improved search
        try:
            search_results = self.search_scraper.search_and_scrape_best(
                query=query,
                num_results=num_results,
                oversample_factor=self.oversample_factor,
                delay=self.default_delay,
                min_quality_score=self.min_quality_score
            )

            if not search_results or "No quality results found" in search_results or "No search results found" in search_results:
                formatted_message = self.no_results_message.format(query=query) if '{query}' in self.no_results_message else self.no_results_message
                return SwaigFunctionResult(formatted_message)

            response = f"Quality web search results for '{query}':\n\n{search_results}"
            return SwaigFunctionResult(response)

        except Exception as e:
            self.logger.error(f"Error performing web search: {e}")
            return SwaigFunctionResult(
                "Sorry, I encountered an error while searching. Please try again later."
            )

    def get_hints(self) -> List[str]:
        """Return speech recognition hints"""
        return []

    def get_global_data(self) -> Dict[str, Any]:
        """Return global data for agent context"""
        return {
            "web_search_enabled": True,
            "search_provider": "Google Custom Search",
            "quality_filtering": True
        }

    def get_prompt_sections(self) -> List[Dict[str, Any]]:
        """Return prompt sections to add to agent"""
        return [
            {
                "title": "Web Search Capability (Quality Enhanced)",
                "body": f"You can search the internet for high-quality information using the {self.tool_name} tool.",
                "bullets": [
                    f"Use the {self.tool_name} tool when users ask for information you need to look up",
                    "The search automatically filters out low-quality results like empty pages",
                    "Results are ranked by content quality, relevance, and domain reputation",
                    "Summarize the high-quality results in a clear, helpful way"
                ]
            }
        ]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Get the parameter schema for the web search skill"""
        schema = super().get_parameter_schema()

        # Add web search specific parameters
        schema.update({
            "api_key": {
                "type": "string",
                "description": "Google Custom Search API key",
                "required": True,
                "hidden": True,
                "env_var": "GOOGLE_SEARCH_API_KEY"
            },
            "search_engine_id": {
                "type": "string",
                "description": "Google Custom Search Engine ID",
                "required": True,
                "hidden": True,
                "env_var": "GOOGLE_SEARCH_ENGINE_ID"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of high-quality results to return",
                "default": 3,
                "required": False,
                "min": 1,
                "max": 10
            },
            "delay": {
                "type": "number",
                "description": "Delay between scraping pages in seconds",
                "default": 0.5,
                "required": False,
                "min": 0
            },
            "max_content_length": {
                "type": "integer",
                "description": "Maximum total response size in characters",
                "default": 32768,
                "required": False,
                "min": 1000
            },
            "oversample_factor": {
                "type": "number",
                "description": "How many extra results to fetch for quality filtering (e.g., 2.5 = fetch 2.5x requested)",
                "default": 2.5,
                "required": False,
                "min": 1.0,
                "max": 3.5
            },
            "min_quality_score": {
                "type": "number",
                "description": "Minimum quality score (0-1) for including a result",
                "default": 0.3,
                "required": False,
                "min": 0.0,
                "max": 1.0
            },
            "no_results_message": {
                "type": "string",
                "description": "Message to show when no quality results are found. Use {query} as placeholder.",
                "default": "I couldn't find quality results for '{query}'. The search returned only low-quality or inaccessible pages. Try rephrasing your search or asking about a different topic.",
                "required": False
            }
        })

        return schema