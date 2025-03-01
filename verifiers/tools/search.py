from dotenv import load_dotenv
import os
import json
load_dotenv()

def search_web(query: str, num_results: int = 5) -> str:
    """Searches web and returns full texts of top results.
    
    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)
        
    Returns:
        Formatted string with bullet points of top results, each with title and brief summary
        
    Examples:
        {"query": "who invented the lightbulb", "num_results": 3}
    """
    from exa_py import Exa
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))
    result = exa.search_and_contents(
        query,
        text={"max_characters": 1000}
    )
    return json.dumps([r.text for r in result.results[:num_results]])