from dotenv import load_dotenv
import os
import json

load_dotenv()


def search_web(query: str, num_results: int = 3) -> str:
    """Searches web and returns full texts of top results.

    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)

    Returns:
        A list of documents exist on web, each with title, text and published date

    Examples:
        {"text": "who invented the lightbulb", "title": "lightbulb", "publishedDate":"2023-04-20T09:39:40.000Z"}
    """
    from exa_py import Exa
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))
    result = exa.search_and_contents(
        query,
        text={"max_characters": 1000}
    )
    return json.dumps([{"text":r.text, "title":r.title, "published_data":r.published_date} for r in result.results[:num_results]])