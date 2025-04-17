import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load data
with open('recommender_data_links.pkl', 'rb') as f:
    data = pickle.load(f)

movies = data['movies_with_links']
embeddings = data['embeddings']
indices = data['indices']

# TMDB API key
TMDB_API_KEY = st.secrets["api_key"]

def fetch_movie_details(tmdb_id):
    if np.isnan(tmdb_id):
        return None

    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return {
            'title': data.get('title'),
            'poster': f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get('poster_path') else None,
            'overview': data.get('overview'),
            'rating': data.get('vote_average'),
            'release_date': data.get('release_date', '')[:4],
            'genres': ', '.join([g['name'] for g in data.get('genres', [])])
        }
    return None

def recommend(title, top_n=6):
    title = title.lower().strip()
    normalized_indices = {k.lower().strip(): v for k, v in indices.items()}

    if title not in normalized_indices:
        return []
    
    idx = normalized_indices[title]
    query_embedding = embeddings[idx].reshape(1, -1)
    similarities = cosine_similarity(query_embedding, embeddings).flatten()
    similar_indices = similarities.argsort()[::-1][1:top_n+1]

    recommended = []
    for i in similar_indices:
        movie = movies.iloc[i]
        details = fetch_movie_details(movie['tmdbId'])
        if details:
            recommended.append(details)

    return recommended

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.markdown("<h1 style='text-align: center;'>🎬 Movie Recommender System</h1>", unsafe_allow_html=True)

movie_list = movies['title'].values
selected_movie = st.selectbox("🎥 Choose a movie you like:", movie_list)

if st.button("✨ Show Recommendations"):
    with st.spinner("Fetching your personalized movie picks..."):
        recommendations = recommend(selected_movie)

    if recommendations:
        st.subheader("💡 You might also like:")

        # Decide number of columns dynamically
        num_cols = min(len(recommendations), 3)  # Show up to 3 columns for responsiveness
        rows = [recommendations[i:i + num_cols] for i in range(0, len(recommendations), num_cols)]

        for row in rows:
            cols = st.columns(len(row))
            for i, movie in enumerate(row):
                with cols[i]:
                    if movie['poster']:
                        st.image(movie['poster'], use_container_width=True)
                    else:
                        st.write("🎞️ No poster available")
                    st.markdown(f"**{movie['title']} ({movie['release_date']})**")
                    st.caption(f"⭐ {movie['rating']} | {movie['genres']}")
                    st.write(movie['overview'][:150] + "...")
    else:
        st.error("😞 Sorry! No recommendations found.")
