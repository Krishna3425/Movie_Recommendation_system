import streamlit as st
import os
import pickle
from src.recommender import MovieRecommender
from src.sentiment_analyzer import SentimentAnalyzer

# Configuration
DATA_DIR = r"C:\Users\Krishna\OneDrive\Desktop\movie recomendation system\data"
MODELS_DIR = r"C:\Users\Krishna\OneDrive\Desktop\movie recomendation system\models"

st.set_page_config(page_title="MovieInsight - Professional Recommendation System", layout="wide")

# Custom CSS for a classic and attractive UI (No emojis)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #b20710;
        color: white;
    }
    h1, h2, h3 {
        color: #e50914;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #1c1c1c;
    }
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper to load models
@st.cache_resource
def load_resources():
    recommender = MovieRecommender()
    movies_df = recommender.load_data()
    # Keep a copy for filtering before preprocessing strips columns
    filter_df = movies_df.copy()
    processed_df = recommender.preprocess()
    
    # Process filter_df to have clean categories and directors
    filter_df['genres'] = filter_df['genres'].apply(recommender.convert_ast)
    filter_df['director'] = filter_df['crew'].apply(recommender.fetch_director)
    filter_df['director'] = filter_df['director'].apply(lambda x: x[0] if len(x) > 0 else "Unknown")
    
    similarity_path = os.path.join(MODELS_DIR, 'similarity.pkl')
    if os.path.exists(similarity_path):
        with open(similarity_path, 'rb') as f:
            similarity = pickle.load(f)
    else:
        # Initial training if model doesn't exist
        similarity = recommender.train(processed_df)
        
    return recommender, processed_df, similarity, filter_df

recommender, processed_df, similarity, filter_df = load_resources()
sentiment_analyzer = SentimentAnalyzer()

# Sidebar Navigation
st.sidebar.title("MovieInsight Navigation")
page = st.sidebar.selectbox("Select Page", ["Movie Recommender", "Movie Filter", "Sentiment Analysis"])

st.sidebar.markdown("---")
st.sidebar.info("MovieInsight is a professional platform for movie discovery, behavior analysis, and sentiment tracking. It uses the TMDB 5000 dataset for precise results.")

# Page: Movie Filter
if page == "Movie Filter":
    st.title("Advanced Movie Filter")
    st.write("Filter movies based on category, rating, or director.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Get unique genres
        all_genres = set()
        for genres in filter_df['genres']:
            all_genres.update(genres)
        selected_genre = st.selectbox("Select Category", ["All"] + sorted(list(all_genres)))
        
    with col2:
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 0.0, 0.5)
        
    with col3:
        # Get unique directors
        directors = sorted(filter_df['director'].unique())
        selected_director = st.selectbox("Select Director", ["All"] + directors)
        
    # Apply filters
    filtered_results = filter_df.copy()
    
    if selected_genre != "All":
        filtered_results = filtered_results[filtered_results['genres'].apply(lambda x: selected_genre in x)]
        
    if selected_director != "All":
        filtered_results = filtered_results[filtered_results['director'] == selected_director]
        
    filtered_results = filtered_results[filtered_results['vote_average'] >= min_rating]
    
    st.markdown("---")
    st.subheader(f"Found {len(filtered_results)} Movies")
    
    if not filtered_results.empty:
        for idx, row in filtered_results.head(20).iterrows():
            with st.expander(f"{row['title']} (Rating: {row['vote_average']})"):
                st.write(f"**Director:** {row['director']}")
                st.write(f"**Genres:** {', '.join(row['genres'])}")
                st.write(f"**Overview:** {row['overview']}")
                st.write(f"**Popularity:** {row['popularity']}")
    else:
        st.warning("No movies match your criteria.")

# Page 1: Movie Recommender
elif page == "Movie Recommender":
    st.title("Strategic Content Recommendation")
    st.write("Enter a movie name to see its details and get similar content recommendations.")
    
    search_query = st.text_input("Search for a movie", placeholder="e.g. Inception")
    
    if st.button("Search and Recommend"):
        if search_query:
            # Case-insensitive search
            match = processed_df[processed_df['title'].str.lower() == search_query.lower()]
            
            if not match.empty:
                movie_title = match.iloc[0]['title']
                st.header(f"Results for: {movie_title}")
                
                # Get full details from filter_df
                details = filter_df[filter_df['title'] == movie_title].iloc[0]
                
                # Display Major Details
                st.subheader("Movie Overview & Tags")
                st.write(f"**Genres:** {', '.join(details['genres'])}")
                st.info(match.iloc[0]['tags'].capitalize())
                
                st.markdown("---")
                
                # Generate Recommendations
                recommendations = recommender.recommend(movie_title, processed_df, similarity)
                
                if recommendations:
                    st.subheader(f"Recommended for you based on '{movie_title}':")
                    for movie in recommendations:
                        rec_details = filter_df[filter_df['title'] == movie].iloc[0]
                        with st.expander(f"{rec_details['title']} (Rating: {rec_details['vote_average']})"):
                            st.write(f"**Director:** {rec_details['director']}")
                            st.write(f"**Genres:** {', '.join(rec_details['genres'])}")
                            st.write(f"**Overview:** {rec_details['overview']}")
                else:
                    st.warning("No similar movies found for recommendations.")
            else:
                st.error(f"Movie '{search_query}' not found. Please check the spelling and try again.")
        else:
            st.info("Please enter a movie name to begin.")

# Page 2: Sentiment Analysis
elif page == "Sentiment Analysis":
    st.title("Audience Sentiment Tracker")
    st.write("Analyze reviews and descriptions to understand audience perception.")
    
    user_review = st.text_area("Enter a movie review or description to analyze sentiment", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_review:
            sentiment, score = sentiment_analyzer.analyze_sentiment(user_review)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment Category", sentiment)
            with col2:
                st.metric("Polarity Score", score)
                
            if sentiment == "Positive":
                st.success("The audience opinion is favorable.")
                
                st.markdown("---")
                st.subheader("Movies Matching Your Positive Description")
                st.write("Based on your input, here are some movies you might like:")
                
                # Simple keyword matching
                keywords = set(user_review.lower().split())
                # Filter out very short words
                keywords = {w for w in keywords if len(w) > 3}
                
                def get_match_score(tags):
                    movie_tags = set(tags.split())
                    return len(keywords.intersection(movie_tags))
                
                match_results = processed_df.copy()
                match_results['match_score'] = match_results['tags'].apply(get_match_score)
                match_results = match_results[match_results['match_score'] > 0]
                match_results = match_results.sort_values(by='match_score', ascending=False).head(5)
                
                if not match_results.empty:
                    for _, match in match_results.iterrows():
                        # Get full details from filter_df
                        details = filter_df[filter_df['title'] == match['title']].iloc[0]
                        with st.expander(f"{details['title']} (Rating: {details['vote_average']})"):
                            st.write(f"**Director:** {details['director']}")
                            st.write(f"**Genres:** {', '.join(details['genres'])}")
                            st.write(f"**Overview:** {details['overview']}")
                else:
                    st.info("No specific movie matches found for those keywords, but your sentiment is great!")
                    
            elif sentiment == "Negative":
                st.error("The audience opinion is unfavorable.")
            else:
                st.warning("The audience opinion is neutral.")
        else:
            st.info("Please enter some text to begin analysis.")

