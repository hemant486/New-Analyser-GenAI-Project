import requests
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Configure page layout - MUST BE FIRST Streamlit command
st.set_page_config(layout="wide", page_title="News Analyzer")

# Set up the News API
API_KEY = "834e9a26da0b4f03a068138496eb8a33"
NEWS_API_URL = "https://newsapi.org/v2/top-headlines"

# Load models with caching
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource
def load_summarizer():
    return pipeline('text-generation', model='gpt2', max_length=150)

# Initialize models
embedding_model = load_embedding_model()
sentiment_analyzer = load_sentiment_analyzer()
summarizer = load_summarizer()

# Function to summarize using GPT-2
def summarize_text(content):
    try:
        if not content or len(content) < 50:
            return "Content too short to summarize."
        
        # Prepare prompt for GPT-2
        prompt = f"Summarize this news: {content[:500]}. Summary:"
        
        # Generate summary
        summary = summarizer(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        
        # Clean up the summary (remove the prompt)
        summary = summary.split("Summary:")[1].strip()
        
        return summary
    except Exception as e:
        st.error(f"Error in summarization: {str(e)}")
        return "Error generating summary."

# Sports categories dictionary
sports_categories = {
    "All Sports": "General Sports Coverage",
    "Cricket": "Cricket matches, tournaments, and updates",
    "Football": "Football/Soccer news and updates",
    "Basketball": "Basketball coverage including NBA",
    "Tennis": "Tennis tournaments and player updates",
    "Formula 1": "F1 racing news and updates",
    "Baseball": "Baseball coverage including MLB",
    "Rugby": "Rugby matches and tournaments",
    "Golf": "Golf tournaments and player updates",
    "Boxing": "Boxing matches and fighter updates",
    "MMA": "Mixed Martial Arts news and events"
}

# Function to fetch news with improved sports handling
def fetch_news(category="technology", sport_type=None):
    params = {
        "apiKey": API_KEY,
        "pageSize": 10
    }
    
    if category.lower() == "sports" and sport_type and sport_type != "All Sports":
        # Use 'everything' endpoint for sports queries
        url = "https://newsapi.org/v2/everything"
        params.update({
            "sortBy": "publishedAt",
            "language": "en"
        })
        
        # Sport-specific keywords
        sport_keywords = {
            "Cricket": "cricket OR ipl OR t20 OR test match",
            "Football": "football OR soccer OR premier league OR fifa",
            "Basketball": "basketball OR nba OR ncaa",
            "Tennis": "tennis OR atp OR wta OR grand slam",
            "Formula 1": "formula 1 OR f1 OR racing",
            "Baseball": "baseball OR mlb",
            "Rugby": "rugby OR six nations",
            "Golf": "golf OR pga",
            "Boxing": "boxing OR heavyweight",
            "MMA": "mma OR ufc OR mixed martial arts"
        }
        
        if sport_type in sport_keywords:
            params["q"] = sport_keywords[sport_type]
    else:
        url = NEWS_API_URL
        params["category"] = category.lower()

    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            articles = response.json().get("articles", [])
            if not articles and category.lower() == "sports":
                st.warning(f"No articles found for {sport_type if sport_type else category}. Fetching general sports news.")
                params = {"apiKey": API_KEY, "category": "sports", "pageSize": 10}
                response = requests.get(NEWS_API_URL, params=params)
                articles = response.json().get("articles", [])
            return [{"title": a["title"], "content": a.get("content", ""), "url": a["url"]} for a in articles]
        else:
            st.error(f"Error fetching news: {response.json().get('message', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return []

# Function to summarize and analyze articles
def summarize_and_analyze(articles):
    summarized_articles = []
    progress_bar = st.progress(0)
    
    for idx, article in enumerate(articles):
        content = article["content"] or "No content available."
        try:
            summary = summarize_text(content)
        except Exception:
            summary = "Error generating summary."

        try:
            sentiment = sentiment_analyzer(summary)[0]["label"]
        except Exception:
            sentiment = "Error analyzing sentiment."

        summarized_articles.append({
            "title": article["title"],
            "summary": summary,
            "sentiment": sentiment,
            "url": article["url"]
        })
        
        # Update progress bar
        progress_bar.progress((idx + 1) / len(articles))
    
    progress_bar.empty()
    return summarized_articles

# FAISS index creation and search functions
def create_faiss_index(articles):
    texts = [article["summary"] for article in articles]
    embeddings = embedding_model.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def search_faiss(query, index, articles):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(query_embedding, k=5)
    return [(articles[i], distances[0][i]) for i in indices[0]]

# Add title after page config
st.title("AI-Powered News Summarizer and Analyzer")

# Sidebar with enhanced category selection
st.sidebar.header("News Options")
categories = {
    "Politics": "politics",
    "Sports": "sports",
    "Technology": "technology",
    "Business": "business",
    "Entertainment": "entertainment",
    "Health": "health",
    "Science": "science"
}

category = st.sidebar.selectbox(
    "Select News Category",
    list(categories.keys()),
    help="Choose a category to fetch related news articles"
)

# Sports subcategories
sports_types = None
if category == "Sports":
    sports_types = st.sidebar.selectbox(
        "Select Sport Type",
        list(sports_categories.keys()),
        help="Choose a specific sport to filter news"
    )
    
    # Display description of selected sport
    if sports_types in sports_categories:
        st.sidebar.markdown(f"*{sports_categories[sports_types]}*")

# Politics subcategories
politics_filter = None
if category == "Politics":
    politics_filter = st.sidebar.selectbox(
        "Select Politics Type",
        ["All Politics", "International", "National", "Elections", "Policy", "Diplomacy"],
        help="Choose a specific political topic to filter news"
    )

# Main content
if "fetch_state" not in st.session_state:
    st.session_state.fetch_state = False

if st.button("Fetch News"):
    st.session_state.fetch_state = True

if st.session_state.fetch_state:
    with st.spinner("Fetching latest news..."):
        if category == "Sports" and sports_types != "All Sports":
            articles = fetch_news(categories[category], sports_types)
        elif category == "Politics" and politics_filter != "All Politics":
            articles = fetch_news(categories[category], politics_filter)
        else:
            articles = fetch_news(categories[category])

    if articles:
        with st.spinner("Analyzing articles with GPT-2..."):
            summarized_articles = summarize_and_analyze(articles)
            index, embeddings = create_faiss_index(summarized_articles)

        # Display summarized articles with category context
        st.write(f"### Latest {category} News")
        if sports_types and sports_types != "All Sports":
            st.write(f"Showing news for: {sports_types}")
        elif politics_filter and politics_filter != "All Politics":
            st.write(f"Showing news for: {politics_filter}")

        for article in summarized_articles:
            with st.expander(article["title"]):
                st.write(f"**Summary:** {article['summary']}")
                st.write(f"**Sentiment:** {article['sentiment']}")
                st.write(f"[Read full article]({article['url']})")

        # Search functionality
        st.write("### Search Articles")
        query = st.text_input("Enter your search query:")
        if query:
            st.write(f"### Search Results for '{query}':")
            search_results = search_faiss(query, index, summarized_articles)
            for result, distance in search_results:
                relevance_score = 1/(1+distance)
                with st.expander(f"{result['title']} (Relevance: {relevance_score:.2f})"):
                    st.write(f"**Summary:** {result['summary']}")
                    st.write(f"**Sentiment:** {result['sentiment']}")
                    st.write(f"[Read full article]({result['url']})")
    else:
        st.error("No articles found. Please try a different category or check back later.")
else:
    st.info("Click 'Fetch News' to get the latest articles.")

# Add footer with model attribution
st.sidebar.markdown("---")
st.sidebar.info("Powered by GPT-2 for intelligent summarization")