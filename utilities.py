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