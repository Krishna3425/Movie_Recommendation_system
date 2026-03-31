# MovieInsight: Professional Movie Recommendation and Analysis System

MovieInsight is a comprehensive data science platform tailored for the entertainment and media industry. It provides advanced solutions for content discovery, audience sentiment tracking, and strategic metadata filtering through a high-performance Python-based architecture.

## Problem Domain
In an era of content abundance, users often struggle to find relevant media that aligns with their specific tastes or emotional states. MovieInsight solves this by offering structured discovery tools that leverage historical metadata and natural language processing to deliver precise, context-aware results.

## Core Features
- Strategic Movie Recommender: A search-based discovery engine that analyzes movie metadata (genres, keywords, cast, and crew) to find similar content. It provides deep insights into searched titles before suggesting mathematically similar alternatives.
- Advanced Movie Filter: A professional filtering interface allowing users to narrow down the vast TMDB library based on multiple concurrent criteria, including categories (genres), minimum user ratings, and specific directors.
- Audience Sentiment Tracker: A natural language analysis tool that evaluates user reviews or descriptions. If a positive sentiment is detected, the system intelligently extracts keywords from the input to proactively suggest matching films.

## Technical Architecture
The system employs a modular, scalable architecture:
- app.py: The primary application entry point, featuring a cinematic dark-themed Streamlit interface and dynamic navigation.
- src/recommender.py: Contains the core recommendation logic, using TF-IDF vectorization and Cosine Similarity to compute relationships between films.
- src/sentiment_analyzer.py: Implements sentiment classification using the TextBlob library for behavioral analysis and review tracking.
- data/: Directory for the TMDB 5000 dataset (movie_dataset.csv and movie_credits.csv).
- models/: Storage for serialized similarity matrices to ensure fast runtime performance.

## Dataset Information
The system utilizes the TMDB 5000 Movie Dataset for its robust metadata:
- movie_dataset.csv: Primary metadata including budget, genres, revenue, and vote averages.
- movie_credits.csv: Comprehensive cast and crew information essential for director-based filtering and actor-based similarity.

## Installation and Deployment

1. Environment Configuration:
   Ensure Python 3.8+ is installed. Install required libraries using:
   pip install -r requirements.txt

2. Model Initialization (Optional):
   The application will automatically train the similarity model on its first run, but it can be initialized manually via:
   python src/recommender.py

3. Execution:
   Launch the interactive dashboard with:
   streamlit run app.py

## Design Philosophy
MovieInsight adheres to a strict professional aesthetic. The user interface and all project documentation are designed to be high-signal and free of emojis or decorative clutter, ensuring a classic and focused presentation suitable for industry analysis.

## Source and Acknowledgments
Data is sourced from The Movie Database (TMDB) and serves as the foundation for this research into content-based filtering and sentiment-driven discovery.
