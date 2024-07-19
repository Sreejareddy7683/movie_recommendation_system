import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import base64
import os

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Load and prepare data
@st.cache_data
def load_data():
    ratings = pd.read_csv(r'C:\Users\P Kondal Reddy\OneDrive\Desktop\movierecs\ratings.csv')
    movies = pd.read_csv(r'C:\Users\P Kondal Reddy\OneDrive\Desktop\movierecs\movies.csv')
    return ratings, movies

@st.cache_data
def prepare_data(ratings, movies):
    # Merge the ratings and movies datasets
    data = pd.merge(ratings, movies, on='movieId')

    # Create a user-item interaction matrix
    user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

    # Replace NaN values with 0
    user_movie_matrix.fillna(0, inplace=True)

    return user_movie_matrix, data

def get_similar_movies(selected_movie, movies_data, num_recommendations=5):
    if selected_movie not in movies_data['title'].values:
        return []

    # Get the genre of the selected movie
    selected_movie_genre = movies_data[movies_data['title'] == selected_movie]['genres'].values[0]

    # Find movies with similar genres
    similar_movies = movies_data[movies_data['genres'] == selected_movie_genre]
    similar_movies = similar_movies[similar_movies['title'] != selected_movie]
    recommended_movies = similar_movies.head(num_recommendations)
    return recommended_movies['title'].tolist()

# Streamlit app
def main():
    # Define the background image path relative to the script
    background_image_path = r'C:\Users\P Kondal Reddy\OneDrive\Desktop\movierecs\image.jpeg'

    # Convert image to base64
    if os.path.isfile(background_image_path):
        base64_image = image_to_base64(background_image_path)
    else:
        base64_image = None

    # Apply background image if the base64 image is correctly converted
    if base64_image:
        st.markdown(
            f"""
            <style>
            .main {{
                background-image: url('data:image/jpeg;base64,{base64_image}');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                height: 100vh;
                width: 100vw;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 0;
                padding: 0;
                overflow: hidden;
            }}
            .css-1h6b5t2 {{
                background-color: transparent;
                color: #000000;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.write("Background image not found. Check the path and file.")

    st.title('Movie Recommendation System')
    
    # Load data
    ratings, movies = load_data()
    user_movie_matrix, data = prepare_data(ratings, movies)
    
    # Dropdown for movie selection
    selected_movie = st.selectbox('Select a Movie:', movies['title'].unique())
    
    if st.button('Get Recommendations'):
        recommendations = get_similar_movies(selected_movie, movies)
        
        if recommendations:
            st.write('Recommended Movies:')
            for movie in recommendations:
                st.write(movie)
        else:
            st.write('No recommendations available for the selected movie.')

if __name__ == "__main__":
    main()
