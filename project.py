import json
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np

import math

from collaborative_filtering import user_based_collab_filtering
from content_based import bernoulli_bayes

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 240)

def load_data(path):
    """
        Load events from files and convert to dataframe.
        @orginal-author: zhanglemei and peng
        @author: trygve nerland
    """
    map_lst=[]
    for f in tqdm(os.listdir(path)):
        file_name=os.path.join(path,f)
        if os.path.isfile(file_name):
            for line in open(file_name):
                obj = json.loads(line.strip())
                if not obj is None:
                    map_lst.append(obj)
    return pd.DataFrame(map_lst)


def binarize_categories(df):
    print('Binarizing categories')
    categories_df = df['category'].str.split('|', expand=True)
    categories = pd.unique(categories_df.values.ravel('K'))
    categories = categories[categories != None]  # Remove none from categories
    for i in tqdm(categories):
        df['category_' + i] = np.where(i in categories_df, 1, 0)
    df = df.drop(['category'], axis=1)
    return df


def pre_processing(df):
    # Remove homepage events and events without an article
    print('Removing homepage hits from datatset')
    df.dropna(subset=["documentId"], inplace=True)

    print("Removing duplicates")
    df.drop_duplicates(subset=["userId", "documentId"], inplace=True)

    print("Sort by user and time")
    df = df.sort_values(by=["userId", "time"])

    print("Adding transaction ID")
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId': item_ids, 'tid': range(len(item_ids))})
    df = pd.merge(df, new_df, on='documentId', how='outer').sort_values(by=["userId", "time"])

    print("Resetting index")
    df = df.reset_index().drop(columns="index")

    # remove incorrectly scraped events COMBINED WITH THE FRONTPAGE ARTICLES
    # print('Removing incorrectly events.')
    # df = df.replace(to_replace='None', value=np.nan).dropna(subset=['documentId'])

    # Print final dataframe
    print('Printing final dataframe')
    print(df)
    return df

def get_ratings_matrix(df):
    num_users = df["userId"].nunique()
    num_items = df["documentId"].nunique()

    ### Note that this part is equal to the implementation in the project example ###
    ratings = np.zeros((num_users, num_items))
    new_user = df['userId'].values[1:] != df['userId'].values[:-1]
    new_user = np.r_[True, new_user]
    df['uid'] = np.cumsum(new_user)
    df_ext = df[['uid', 'tid']]

    for row in df_ext.itertuples():
        ratings[row[1]-1, row[2]-1] = 1.0
    print(ratings)
    return ratings

def train_test_split(ratings, item_based=False):
    if item_based:
        ratings = ratings.T

    test_set = np.zeros_like(ratings)
    train_set = np.copy(ratings)

    for i, user in enumerate(ratings):
        num_test_docs = math.ceil(len(ratings[i].nonzero()[0]) * 0.2)
        test_docs = np.random.choice(ratings[i].nonzero()[0], size=num_test_docs, replace=False)
        train_set[i, test_docs] = 0.
        test_set[i, test_docs] = 1.
    return train_set, test_set

def evaluate(recommendations, test_set):
    recall = 0
    ARHR = 0
    CTR = 0
    for user in range(recommendations.shape[0]):
        correct_recommendations = np.intersect1d(recommendations[user], test_set[user].nonzero()[0])
        recall += len(correct_recommendations) / len(recommendations[user])

        if len(correct_recommendations) > 0:
            CTR += 1

        for correct_recommendation in correct_recommendations:
            ARHR += 1. / (recommendations[user].tolist().index(correct_recommendation) + 1)

    recall = recall / recommendations.shape[0]
    CTR = CTR / recommendations.shape[0]
    ARHR = ARHR / recommendations.shape[0]
    print(recall)
    print(ARHR)
    print(CTR)

    
if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    print('Loading dataset...')
    df = load_data("active1000")
    df = pre_processing(df)

    # recommendations, test_sets = bernoulli_bayes(df)
    ratings = get_ratings_matrix(df)
    train_set, test_sets = train_test_split(ratings, item_based=True)
    recommendations = user_based_collab_filtering(train_set, 20)
    evaluate(recommendations, test_sets)
