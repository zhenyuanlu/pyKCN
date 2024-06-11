from serpapi import GoogleSearch
import json
import os

# SerpApi API key
api_key = os.environ["SERPAPI_API_KEY"]

# List of article titles to track
articles_to_track = [
    'Novel keyword co-occurrence network-based methods to foster systematic reviews of scientific literature',
    'Analysis of pain research literature through keyword co-occurrence networks',
    'Trends in adopting Industry 4.0 for asset life cycle management for sustainability: a keyword co-occurrence network review and analysis',
    'Navigating the Evolution of Digital Twins Research through Keyword Co-Occurence Network Analysis'
]

# Fetch citation counts for each article
citation_counts = {}
for article_title in articles_to_track:
    try:
        params = {
            "api_key": api_key,
            "engine": "google_scholar",
            "q": article_title
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        if "organic_results" in results:
            article = results["organic_results"][0]
            citation_count = article.get("inline_links", {}).get("cited_by", {}).get("total", 0)
            citation_counts[article_title] = citation_count
        else:
            print(f"No results found for: {article_title}")

    except Exception as e:
        print(f"Error occurred for article: {article_title}")
        print(f"Error message: {str(e)}")

# Save to a JSON file
with open('article_citations.json', 'w') as f:
    json.dump(citation_counts, f)

# Generate badge URLs
badge_urls = {}
for title, count in citation_counts.items():
    badge_urls[title] = f"https://img.shields.io/badge/Citations-{count}-blue"

# Save badge URLs to a JSON file
with open('badge_urls.json', 'w') as f:
    json.dump(badge_urls, f)
