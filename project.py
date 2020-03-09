import json
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import pandas as pd
import numpy as np
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
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId': item_ids, 'tid': range(1, len(item_ids) + 1)})
    df = pd.merge(df, new_df, on='documentId', how='outer')
    df_ext = df[['uid', 'tid']]

    for row in df_ext.itertuples():
        ratings[row[1]-1, row[2]-1] = 1.0
    print(ratings)
    return ratings

def train_test_split(ratings):
    test_set = np.zeros_like(ratings)
    train_set = np.copy(ratings)

    for i, user in enumerate(ratings):
        num_test_docs = int(len(ratings[i].nonzero()[0]) * 0.2)
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
        recall += len(correct_recommendations) / len(test_set[user].nonzero()[0])
        CTR += len(correct_recommendations)

        for correct_recommendation in correct_recommendations:
            ARHR += 1. / (recommendations[user].tolist().index(correct_recommendation) + 1)

    recall = recall / recommendations.shape[0]
    CTR = CTR / recommendations.shape[0]
    ARHR = ARHR / recommendations.shape[0]
    print(recall)
    print(ARHR)
    print(CTR)



def user_based_collab_filtering(ratings, k):
    cosine_sim = cosine_similarity(ratings, ratings)  # Cosine sim instead of linear kernel to get normalized values
    ind = np.flip(np.argsort(cosine_sim, axis=1), 1)  # Get the indices of the most similar users for each users sorted
    # print(ind)
    #sim = np.take_along_axis(cosine_sim, ind, axis=1)  # Get the sim values for the most similar users
    ind_k = np.delete(ind, 0, 1)  # The first row corresponds to the user itself so remove it
    ind_k = np.delete(ind_k, slice(k, None), 1)  # We are only looking at the k most similar, so remove to less similar users
    # print(ind_k)

    recommendations = np.empty((ratings.shape[0], k))
    for user in range(ratings.shape[0]):  # Find recommendations for every user
        relevant_documents = []
        for document in range(ratings.shape[1]):  # Look at every document to see how relevant it is
            if ratings[user, document] == 0:  # If the user has seen the document before, it should not be recommended
                value = 0
                for sim_user in ind_k[user]:  # Look at the k most similar users
                    if ratings[sim_user, document] == 1:  # If the similar user has seen the document it is likely relevant to the current user
                        value += cosine_sim[user, sim_user]  # Add the similarity value between the users as the value used to rank the document
                relevant_documents.append((document, value))  # Append both the document index and the value for the document
        relevant_documents = sorted(relevant_documents, key=lambda x: x[1], reverse=True)  # Sort the documents based on the value
        relevant_documents = relevant_documents[:k]  # Keep only the k most relevant
        relevant_documents = [i for i, j in relevant_documents]  # Drop the value to keep only the documents
        recommendations[user] = np.array(relevant_documents)

    print(recommendations)
    return recommendations
    
if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    print('Loading dataset...')
    df = load_data("active1000")

    df = pre_processing(df)
    ratings = get_ratings_matrix(df)
    train_set, test_set = train_test_split(ratings)
    recommendations = user_based_collab_filtering(train_set, 20)
    evaluate(recommendations, test_set)
