from scholarly import scholarly
import json

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
    search_query = scholarly.search_pubs(article_title)
    article = next(search_query)
    citation_counts[article_title] = article['num_citations']

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
