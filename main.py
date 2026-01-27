import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Function splits the test and train sets accordingly
def train_test_split_userwise(df, test_size = 0.2):
    train, test = [], []
    for user, group in df.groupby("userId"):            # Returns interator which at each point give a tuple of key and subgroup
        if len(group) >= 5:                             # Makes sure users that haven't rated much do not get included in test set
                        # Line below splits each set of ratings by a user into test and train sets
            tr, te = train_test_split(group, test_size= test_size, random_state= 42)
            train.append(tr)
            test.append(te)
        else:
            train.append(group)                         # Users with fewer than 5 ratings simply added here for training
    
    return pd.concat(train), pd.concat(test)            # Concatenates the list elements into a single pandas dataframe


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



print(user_item.iloc[0:10, 0:10], "\n \n", movie_stats)
#print(n_users, n_movies, '\n \n', ratings_2.head())

