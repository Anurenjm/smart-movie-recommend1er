import streamlit as st
import pandas as pd
import ast
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# ----------------------------
# Load and Prepare TMDB Movies
# ----------------------------
@st.cache_resource
def load_movie_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    movies = movies[['title', 'genres', 'vote_average', 'overview']].dropna()
    movies['genres'] = movies['genres'].apply(lambda x: [d['name'] for d in ast.literal_eval(x)])
    return movies

movies = load_movie_data()

# Encode genres + rating
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(movies['genres'])
X_movies = pd.DataFrame(genres_encoded, columns=mlb.classes_)
X_movies['rating'] = movies['vote_average'].values

# KNN model
knn = NearestNeighbors(n_neighbors=6, metric='cosine')
knn.fit(X_movies)

# ----------------------------
# Typing Suggestion (Improved)
# ----------------------------
def get_movie_suggestions(prefix):
    prefix = prefix.lower()
    return [title for title in movies['title'] if prefix in title.lower()][:5]

# ----------------------------
# Movie Recommendation
# ----------------------------
def recommend_movies(title):
    idx = movies[movies['title'].str.lower() == title.lower()].index
    if not len(idx):
        return []
    distances, indices = knn.kneighbors(X_movies.iloc[idx[0]].values.reshape(1, -1))
    return movies.iloc[indices[0][1:]][['title', 'overview']].values.tolist()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🎬 TMDB Movie Recommender")
st.title("🎬 Smart Movie Recommender (TMDB Edition)")
st.markdown("Start typing a movie name to get smart suggestions and content-based recommendations.")

query = st.text_input("🎯 Start typing a movie name:")

if query:
    suggestions = get_movie_suggestions(query)

    if suggestions:
        selected = st.selectbox("🎬 Select a movie from suggestions:", suggestions)

        if selected:
            st.subheader(f"📽️ Recommendations for **{selected}**")
            recommended = recommend_movies(selected)

            for i, (movie, overview) in enumerate(recommended, 1):
                st.markdown(f"**{i}. {movie}**  \n{overview[:200]}...")
    else:
        st.warning("❌ No matching titles found . Try a different keyword.")
