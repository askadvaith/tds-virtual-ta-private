import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from tqdm import tqdm
import json
import base64
import os
from datetime import timedelta

from dotenv import load_dotenv
load_dotenv()

DISCOURSE_COOKIE = os.getenv("DISCOURSE_COOKIE") 

session = requests.Session()
session.cookies.set("_t", DISCOURSE_COOKIE, domain="discourse.onlinedegree.iitm.ac.in")
session.headers.update({"User-Agent": "Mozilla/5.0"})

BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"

def get_topic_ids(category_slug="courses/tds-kb", category_id=34):
    topics = []
    for page in tqdm(range(0, 20)):  # Adjust if you want more pages
        url = f"{BASE_URL}/c/{category_slug}/{category_id}.json?page={page}"
        r = session.get(url)
        if r.status_code != 200:
            break
        data = r.json()
        new_topics = data["topic_list"]["topics"]
        if not new_topics:
            break
        topics.extend(new_topics)
    return topics

def get_posts_in_topic(topic_id):
    r = session.get(f"{BASE_URL}/t/{topic_id}.json")
    if r.status_code != 200:
        return []
    data = r.json()
    slug = data.get("slug")  
    posts_data = []
    for post in data["post_stream"]["posts"]:
        soup = BeautifulSoup(post["cooked"], "html.parser")
        content = soup.get_text()
        images_base64 = []
        for img in soup.find_all("img"):
            img_url = img.get("src")
            if img_url:
                if img_url.startswith("/"):
                    img_url = BASE_URL + img_url
                try:
                    img_data = session.get(img_url).content
                    images_base64.append(base64.b64encode(img_data).decode("utf-8"))
                except Exception as e:
                    print(f"Could not download image {img_url}: {e}")

        posts_data.append(
            {
                "topic_id": topic_id,
                "slug": slug, 
                "post_number": post["post_number"],
                "username": post["username"],
                "created_at": post["created_at"],
                "content": content,
                "post_url": f"{BASE_URL}/t/{slug}/{topic_id}/{post['post_number']}",  # Use slug in URL
                "images_base64": images_base64,
            }
        )
    return posts_data

all_posts = []
topics = get_topic_ids()
# Define IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Define start and end datetimes in IST
start_ist = datetime(2025, 1, 1, tzinfo=IST)
end_ist = datetime(2025, 4, 15, 23, 59, 59, tzinfo=IST)

for topic in tqdm(topics):
    # Parse created_at as UTC
    created_at_utc = datetime.fromisoformat(topic["created_at"].replace("Z", "+00:00"))
    # Convert to IST
    created_at_ist = created_at_utc.astimezone(IST)
    # Filter by IST range
    if start_ist <= created_at_ist <= end_ist:
        posts = get_posts_in_topic(topic["id"])
        all_posts.extend(posts)

# Save the scraped posts into a JSON file
with open("data/tds_discourse_posts.json", "w", encoding="utf-8") as f:
    json.dump(all_posts, f, indent=2, ensure_ascii=False)

print(f"Scraped {len(all_posts)} posts.")