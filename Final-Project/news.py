import feedparser
import pandas as pd
import re
import requests
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

rss_feeds = {
    "Times of India": "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
    "Mint": "https://www.livemint.com/rss/news",
    "Economic Times": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    "The Hindu": "https://www.thehindu.com/news/national/feeder/default.rss"
}

def fetch_latest_news(max_per_source=5):
    all_news = []

    for source, url in rss_feeds.items():
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_per_source]:
            all_news.append({
                "source": source,
                "title": entry.title,
                "summary": entry.get("summary", ""),
                "link": entry.link,
                "published": entry.get("published", "N/A")
            })
    
    return all_news


def parse_news_item(item):
    # Pattern 1: ISO 8601 date format
    iso_pattern = r"\[(.*?)\] (.*?) - (\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})"
    match = re.match(iso_pattern, item)
    if match:
        source, headline, date_str, time_str = match.groups()
        return {"Source": source, "Headline": headline, "Date": date_str, "Time": time_str}
    
    # Pattern 2: Custom datetime format (e.g. "Wed, 30 Apr 2025 15:48:44 +0530")
    custom_pattern = r"\[(.*?)\] (.*?) - \w{3}, (\d{2} \w{3} \d{4}) (\d{2}:\d{2}:\d{2})"
    match = re.match(custom_pattern, item)
    if match:
        source, headline, date_str, time_str = match.groups()
        # Optional: Convert to ISO date for consistency
        date_obj = datetime.strptime(date_str, "%d %b %Y")
        return {
            "Source": source,
            "Headline": headline,
            "Date": date_obj.strftime("%Y-%m-%d"),
            "Time": time_str
        }

    return None  # If format doesn't match

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(headline):
    sentiment = analyzer.polarity_scores(headline)
    if sentiment['compound'] >= 0.05:
        return 'Positive'
    elif sentiment['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'
    
# Example usage
if __name__ == "__main__":
    news = fetch_latest_news()
    combined_news = []

    for n in news:
        combined_news.append(f"[{n['source']}] {n['title']} - {n['published']}")


    # Parse all news items
    parsed_data = [parse_news_item(item) for item in combined_news if parse_news_item(item)]

    for news_item in parsed_data:
        news_item['Sentiment'] = analyze_sentiment(news_item['Headline'])

    # Convert to DataFrame for pretty display or further processing
    df = pd.DataFrame(parsed_data)

    print(df.head(20))

    print()

