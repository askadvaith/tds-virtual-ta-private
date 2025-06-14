# Virtual Teaching Assistant - IITM
This is a virtual Teaching Assistant for the Tools in Data Science (TDS) course offered as part of the IIT Madras BSc Data Science degree.

It is designed to be a lightweight and accurate solution to generate answers to student questions using course content and previous forum discussions, freeing up valuable TA time (God knows how many pings they get the day before an assignment is due) and providing 24/7 support. 

To be quite honest, this hasn't reached a level of reliability and efficiency that I'm happy with yet, but it does _function_ and do the basics :)

## What's In This Repo
- **app.py:** This is the main API server that exposes endpoints you can query for responses. You can send base64 images as well along with your questions; currently responding using OpenAI's `gpt-4.1-mini` through [https://github.com/sanand0/aipipe](AIPipe).
- **setup.py:** Ensures that all required directory structures and environment variables are present and that you're ready to go.
- **extract_data.py:** Script to scrape the course Discourse forum through a specified date range to gather data which will be used as additional context for the LLM.
- **embedder.py:** Script that goes through all the course content as well as the scraped forum content and creates vector embeddings. Currently also using [https://github.com/sanand0/aipipe](AIPipe) with `text-embedding-ada-002`; I was using `all-MiniLM-L6-v2` at first to use an open source model, however `sentence-transformers` proved to be too heavy a library to run on a free serverless hosting platform like Vercel or Render.
- **data:** Directory that contains all the scraped forum data in `tds_discourse_posts.json`, and all the course content in `tools-in-data-science-public`. Course content was taken directly from https://github.com/sanand0/tools-in-data-science-public/tree/tds-2025-01; the AIPipe file was added later by me which just contains the documentation from https://github.com/sanand0/aipipe.
- **embeddings:** Contains the vectorised forms of the data mentioned above, along with helpful metadata
- **requirements.txt:** IMPORTANT: Only contains the dependencies for `app.py` alone (so that my Render deploy works properly heh). I'll be adding the overall project dependencies to another file later.

## Setting Up Locally
### Quickstart
**1. Clone the repo:**
```bash
git clone https://github.com/askadvaith/tds-virtual-ta-private.git
cd tds-virtual-ta-private
```

**2. Set up a virtual environment, install dependencies:**
```bash
echo "do this however you want"
uv venv
[activate venv]
uv pip install -r requirements.txt
```

**3. Set environment variables:**
```bash
cp .env.example .env
```
Add your AIPipe token into your `.env` file.

**4. Run your server:**
```bash
echo "again do this however you want"
uv run setup.py
uv run app.py
```

### Usage
You can now send questions to the API endpoint using curl requests like so:
```bash
curl -X POST "http://localhost:5000/api/"   -H "Content-Type: application/json"   -d "{\"question\": \"Your question goes here\", \"image\": \"$(base64 -w0 <image-path.webp>)\"}"
```

### Scraping Data
I'm assuming that you're scraping the Discourse forum, which needs authentication; if you're scraping a different source altogether, you may not need this.

**1. Set your Discourse cookie value:**\
Go to Discourse, login, and then open the Network tab in DevTools. Scroll down until new content is loaded on your feed. Discourse loads in new content by sending a request to its API, which will be captured in DevTools. The request will be made to a URL that looks like this: `https://discourse.onlinedegree.iitm.ac.in/t/176077/posts.json?...`

**2. Run the scraping script:**
```bash
uv run extract_data.py
```

### Embedding Data
Once you've got the data loaded, very simple:
```bash
uv run embedder.py
```
> **NOTE:** The course data and forum posts from January to April 2025 have already been scraped and embedded here; if you don't need more recent data, then you don't need to use these scripts at all.
