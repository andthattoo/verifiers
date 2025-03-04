import os
import requests
from dotenv import load_dotenv
load_dotenv()

def jina_scrape(website: str) -> str:
    """Scrapes text content from websites and returns in readable format.
    
    Args:
        website: The URL of the website to scrape
        
    Returns:
        Scraped content as text
        
    Examples:
        jina_scrape("https://example.com")
    """
    url = f"https://r.jina.ai/{website}"
    
    # Check if API key is available
    token = os.environ.get("JINA_API_KEY")
    
    if token:
        # Send request with token
        headers = {
            "Authorization": f"Bearer {token}"
            # Optional headers commented out as in the Rust version
            # "X-With-Generated-Alt": "false",
            # "X-With-Images-Summary": "false",
            # "X-With-Links-Summary": "false"
        }
        response = requests.get(url, headers=headers)
    else:
        # Send request without token
        response = requests.get(url)
    
    return response.text