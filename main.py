import pandas as pd

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

print(user_item.iloc[0:10, 0:10])

#print(n_users, n_movies, '\n \n', ratings_2.head())

