import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


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



# Function returns a recommendation, when given similarity matrix, user means, and respective user and movie IDs
def user_CF_recommendation(user_similarity, user_means, user_item_centered, userId, movieId):

    mean = user_means[userId]                           # Stores the mean value for the User
    similarity_series = user_similarity.loc[userId].sort_values(ascending=True)[1:21]       # For the user in question, obtains the top 20 similarity values
    top_sim_user_indices = similarity_series.index                                          # Extracts the indices of the top 20 similarity values
    centered_values = user_item_centered[movieId].loc[top_sim_user_indices]                 # Extracts respective movie rating series using top_sim_user_indices

    # The calculation below predicts the rating 
    pred_rating = mean + ((similarity_series.to_numpy().reshape(1,20))@(centered_values.to_numpy().reshape(20,1)))/(similarity_series.sum())

    return pred_rating