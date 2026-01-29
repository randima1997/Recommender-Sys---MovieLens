import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utilities import train_test_split_userwise


ratings = pd.read_csv('Movie Lens/ml-32m/ratings.csv')
#movies = pd.read_csv('Movie Lens/ml-32m/movies.csv')

ratings = ratings.iloc[0:50000]                         # Using only 50,000 rows for testing



n_users = ratings.userId.nunique()                      
n_movies = ratings.movieId.nunique()                    # Number of unique users and movies
sparsity = len(ratings)/(n_users*n_movies)

print(n_users, n_movies, sparsity, len(ratings))


# Make a User-Item matrix, with Users on one axis and Movies on the other
user_item = ratings.pivot(
    index= 'userId',
    columns= 'movieId',
    values= 'rating'
)


# The following figures provide a unpersonalized average rating figure to be treated as a baseline
# Performance worse than the below suggests that there's something wrong with the model
global_mean = ratings['rating'].mean()

movie_stats = (ratings.groupby('movieId')
               .agg(mean_rating = ('rating', 'mean'), count = ('rating', 'count')))


# Split the data as train and test

train_ratings, test_ratings = train_test_split_userwise(df = ratings, test_size= 0.2)


# User-Item matrix for training
user_item_train = train_ratings.pivot(
    index= 'userId',
    columns= 'movieId',
    values= 'rating'
)

user_means = user_item_train.mean(axis=1)           # Returns a series with the average rating given by each user
                                                    # Subtracts each element of a row by the respective row average. Thus centering.
user_item_centered = user_item_train.sub(user_means, axis=0).fillna(0)


# Returns a Numpy array of shape n_users x n_users. Each element is a representation of similarity between users
# and varies between -1 and 1
user_similarity = cosine_similarity(user_item_centered)     







#print(user_item.iloc[0:10, 0:10], "\n \n", movie_stats)
#print(n_users, n_movies, '\n \n', ratings_2.head())

