import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Load the datasets
ratings = pd.read_csv(r'C:\Users\P Kondal Reddy\OneDrive\Desktop\movierecs\ratings.csv')
movies = pd.read_csv(r'C:\Users\P Kondal Reddy\OneDrive\Desktop\movierecs\movies.csv')

# Display the first few rows of each dataset to understand their structure
print("First few rows of the ratings dataset:")
print(ratings.head())
print("\nFirst few rows of the movies dataset:")
print(movies.head())

# Basic statistics of the ratings dataset
print("\nSummary statistics of the ratings dataset:")
print(ratings.describe())

# Distribution of movie genres
print("\nDistribution of movie genres:")
print(movies['genres'].value_counts())

# Plot the distribution of movie ratings
plt.figure(figsize=(10, 5))
sns.histplot(ratings['rating'], bins=5, kde=False)
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# Plot the number of ratings each user has given
ratings_per_user = ratings.groupby('userId')['rating'].count()
plt.figure(figsize=(10, 5))
sns.histplot(ratings_per_user, bins=50, kde=False)
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.show()

# Plot the number of ratings each movie has received
ratings_per_movie = ratings.groupby('movieId')['rating'].count()
plt.figure(figsize=(10, 5))
sns.histplot(ratings_per_movie, bins=50, kde=False)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.show()

# Merge the ratings and movies datasets
data = pd.merge(ratings, movies, on='movieId')

# Check for missing values in the merged dataset
print("\nMissing values in the dataset:")
print(data.isnull().sum())

# Create a user-item interaction matrix
user_movie_matrix = data.pivot_table(index='userId', columns='title', values='rating')

# Replace NaN values with 0 in the user-item interaction matrix
user_movie_matrix.fillna(0, inplace=True)

# Calculate the cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# Function to get movie recommendations for a user
def recommend_movies(user_id, num_recommendations=5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:num_recommendations+1]
    similar_users_ratings = user_movie_matrix.loc[similar_users]
    recommended_movies = similar_users_ratings.mean(axis=0).sort_values(ascending=False).index[:num_recommendations]
    return recommended_movies

# Example: Get recommendations for user with ID 1
print("Recommended movies for user 1:")
print(recommend_movies(user_id=1))

# Split the data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Build user-item interaction matrix for the training data
train_user_movie_matrix = train_data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Calculate user similarity for the training data
train_user_similarity = cosine_similarity(train_user_movie_matrix)
train_user_similarity_df = pd.DataFrame(train_user_similarity, index=train_user_movie_matrix.index, columns=train_user_movie_matrix.index)

# Function to predict a user's rating for a specific movie
def predict_rating(user_id, movie_title):
    similar_users = train_user_similarity_df[user_id].sort_values(ascending=False).index[1:11]
    similar_users_ratings = train_user_movie_matrix.loc[similar_users, movie_title]
    predicted_rating = similar_users_ratings.mean()
    return predicted_rating

# Evaluate the model using RMSE
test_data['predicted_rating'] = test_data.apply(lambda x: predict_rating(x['userId'], x['title']), axis=1)
rmse = np.sqrt(mean_squared_error(test_data['rating'], test_data['predicted_rating']))
print(f'Root Mean Squared Error (RMSE): {rmse}')
