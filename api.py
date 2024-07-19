from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load and prepare data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    return movies

movies = load_data()

def get_similar_movies(selected_movie, num_recommendations=5):
    if selected_movie not in movies['title'].values:
        return []

    selected_movie_genres = movies[movies['title'] == selected_movie]['genres'].values[0]
    selected_movie_genres = eval(selected_movie_genres)  # Convert string to list
    selected_movie_genres = [genre['name'] for genre in selected_movie_genres]

    def has_similar_genres(genres_str):
        genres = eval(genres_str)  # Convert string to list
        genres = [genre['name'] for genre in genres]
        return any(genre in selected_movie_genres for genre in genres)

    similar_movies = movies[movies['genres'].apply(has_similar_genres)]
    similar_movies = similar_movies[similar_movies['title'] != selected_movie]
    recommended_movies = similar_movies.head(num_recommendations)
    return recommended_movies['title'].tolist()

@app.route('/recommend', methods=['GET'])
def recommend():
    movie = request.args.get('movie')
    recommendations = get_similar_movies(movie)
    result = [{'title': title} for title in recommendations]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
