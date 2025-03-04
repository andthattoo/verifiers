import os
import json
import requests
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
load_dotenv()

class SearchResult:
    def __init__(self, title: str = "none", link: str = "none", snippet: str = "none", 
                 date: str = "none", position: int = -1):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.date = date
        self.position = position
    
    @classmethod
    def from_result_data(cls, result_data: Dict[str, Any]) -> "SearchResult":
        return cls(
            title=result_data.get("title", "none"),
            link=result_data.get("link", "none"),
            snippet=result_data.get("snippet", "none"),
            date=result_data.get("date", "none"),
            position=result_data.get("position", -1)
        )
    
    def to_formatted_string(self) -> str:
        return f"{self.title}\n{self.link}\n{self.snippet}\n{self.date}\n{self.position}"


class ScholarResult:
    def __init__(self, title: str = "none", link: str = "none", publication_info: str = "none", 
                 snippet: str = "none", year: int = -1, cited_by: int = -1):
        self.title = title
        self.link = link
        self.publication_info = publication_info
        self.snippet = snippet
        self.year = year
        self.cited_by = cited_by
    
    @classmethod
    def from_result_data(cls, result_data: Dict[str, Any]) -> "ScholarResult":
        return cls(
            title=result_data.get("title", "none"),
            link=result_data.get("link", "none"),
            publication_info=result_data.get("publicationInfo", "none"),
            snippet=result_data.get("snippet", "none"),
            year=result_data.get("year", -1),
            cited_by=result_data.get("citedBy", -1)
        )
    
    def to_formatted_string(self) -> str:
        return f"{self.title}\n{self.link}\n{self.publication_info}\n{self.snippet}\n{self.year}\n{self.cited_by}"


class NewsResult:
    def __init__(self, title: str = "none", link: str = "none", snippet: str = "none", 
                 date: str = "none", source: str = "none", image_url: str = "none", position: int = -1):
        self.title = title
        self.link = link
        self.snippet = snippet
        self.date = date
        self.source = source
        self.image_url = image_url
        self.position = position
    
    @classmethod
    def from_result_data(cls, result_data: Dict[str, Any]) -> "NewsResult":
        return cls(
            title=result_data.get("title", "none"),
            link=result_data.get("link", "none"),
            snippet=result_data.get("snippet", "none"),
            date=result_data.get("date", "none"),
            source=result_data.get("source", "none"),
            image_url=result_data.get("imageUrl", "none"),
            position=result_data.get("position", -1)
        )
    
    def to_formatted_string(self) -> str:
        return f"{self.title}\n{self.link}\n{self.snippet}\n{self.date}\n{self.source}\n{self.image_url}\n{self.position}"


def serper_search(query: str, search_type: str = "search", 
                 lang: str = "en", n_results: int = 3) -> str:
    """Conducts a web search using Serper API and returns the results.
    
    Args:
        query: The search query
        search_type: The search type (search, scholar, or news)
        lang: The language for the search
        n_results: The number of results to return, defaults to 3
        
    Returns:
        Formatted search results as a string
        
    Examples:
        serper_search("who invented the lightbulb", "search", "en", 3)
    """
    if search_type not in ["search", "scholar", "news"]:
        raise ValueError("Invalid search type")
    
    url = f"https://google.serper.dev/{search_type}"
    gl = lang if lang != "en" else "us"
    n_results = min(n_results, 10)
    
    payload = {
        "q": query,
        "gl": gl,
        "hl": lang,
        "page": 1,
        "num": n_results
    }
    
    if search_type == "scholar":
        payload.pop("num", None)
    
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is not set")
    
    response = requests.post(
        url,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        },
        json=payload
    )
    response_data = response.json()
    
    results = response_data.get("organic", [])
    
    if search_type == "search":
        formatted_results = [SearchResult.from_result_data(r).to_formatted_string() 
                            for r in results[:n_results]]
    elif search_type == "scholar":
        formatted_results = [ScholarResult.from_result_data(r).to_formatted_string() 
                            for r in results[:n_results]]
    elif search_type == "news":
        formatted_results = [NewsResult.from_result_data(r).to_formatted_string() 
                            for r in results[:n_results]]
    else:
        raise ValueError("Invalid search type")
    
    return "\n".join(formatted_results)


def raw_serper_search(query: str, search_type: str = "search", 
                     lang: str = "en", n_results: int = 5) -> str:
    """Returns raw JSON response from Serper API search.
    
    Args:
        query: The search query
        search_type: The search type (search, scholar, or news)
        lang: The language for the search
        n_results: The number of results to return
        
    Returns:
        Raw JSON response as a string
        
    Examples:
        raw_serper_search("who invented the lightbulb")
    """
    if search_type not in ["search", "scholar", "news"]:
        raise ValueError("Invalid search type")
    
    url = f"https://google.serper.dev/{search_type}"
    gl = lang if lang != "en" else "us"
    n_results = min(n_results, 10)
    
    payload = {
        "q": query,
        "gl": gl,
        "hl": lang,
        "page": 1,
        "num": n_results
    }
    
    if search_type == "scholar":
        payload.pop("num", None)
    
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise ValueError("SERPER_API_KEY environment variable is not set")
    
    response = requests.post(
        url,
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        },
        json=payload
    )
    response_data = response.json()
    
    return json.dumps(response_data)