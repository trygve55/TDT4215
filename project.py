import json
import os

from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split as sklean_train_test_split

from collaborative_filtering import collaborative_filtering
from content_based import bernoulli_bayes, DocumentRanker

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
        df['category_' + i] = np.where(i == categories_df[0], 1, np.where(i == categories_df[1], 1, 0))
    df = df.drop(['category'], axis=1)
    return df


def pre_processing(df):
    # Remove homepage events and events without an article
    print('Removing homepage hits from datatset')
    df.dropna(subset=["documentId"], inplace=True)
    print('Finding duplicates and setting same the documentId')
    df = df.sort_values(['title', 'publishtime'])

    title = None
    publishtime = None
    documentId = None
    title_col = df.columns.get_loc('title')
    publishtime_col = df.columns.get_loc('publishtime')
    documentId_col = df.columns.get_loc('documentId')

    for i in tqdm(range(df.shape[0])):
        temp_title = df.iat[i, title_col]
        temp_publishtime = df.iat[i, publishtime_col]
        if publishtime != temp_publishtime or temp_title != title:
            title = temp_title
            publishtime = temp_publishtime
            documentId = df.iat[i, documentId_col]
            continue

    print('Finding duplicates and setting same the documentId')
    df = df.sort_values(['title', 'publishtime'])

    title = None
    publishtime = None
    documentId = None
    title_col = df.columns.get_loc('title')
    publishtime_col = df.columns.get_loc('publishtime')
    documentId_col = df.columns.get_loc('documentId')

    for i in tqdm(range(df.shape[0])):
        temp_title = df.iat[i, title_col]
        temp_publishtime = df.iat[i, publishtime_col]
        if publishtime != temp_publishtime or temp_title != title:
            title = temp_title
            publishtime = temp_publishtime
            documentId = df.iat[i, documentId_col]
            continue

        df.iat[i, documentId_col] = documentId

    df = df.sort_index()

    print("Removing duplicates events")
    df.drop_duplicates(subset=["userId", "documentId"], inplace=True)

    print("Sort by user and time")
    df = df.sort_values(by=["userId", "time"])

    print("Adding transaction ID")
    item_ids = df['documentId'].unique().tolist()
    new_df = pd.DataFrame({'documentId': item_ids, 'tid': range(len(item_ids))})
    df = pd.merge(df, new_df, on='documentId', how='outer').sort_values(by=["userId", "time"])

    print("Resetting index")
    df = df.reset_index().drop(columns="index")

    # Print final dataframe
    print('Printing final dataframe')

    #Filling NaN from categories
    print('Filling NaN from categories')
    df['category'] = df['category'].fillna("")
    print(df.shape)
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


def train_test_split(ratings):
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
    print("Recall: {}".format(recall))
    print("ARHR: {}".format(ARHR))
    print("CTR: {}".format(CTR))

def evaluate_nearest_neighbor_classifier(recommendations, test_set):
    recall = 0
    ARHR = 0
    CTR = 0
    for userId in recommendations.keys():
        correct_recommendations = list(set(recommendations[userId]).intersection(test_set[userId]))
        recall += len(correct_recommendations) / len(recommendations[userId])
        if len(correct_recommendations) > 0:
            CTR += 1

        for correct_recommendation in correct_recommendations:
            ARHR += 1. / (recommendations[userId].index(correct_recommendation) + 1)
    recall = recall / len(recommendations)
    CTR = CTR / len(recommendations)
    ARHR = ARHR / len(recommendations)
    print("Recall: {}".format(recall))
    print("ARHR: {}".format(ARHR))
    print("CTR: {}".format(CTR))


def get_unique_documents(df):
    print('Adding count')
    df = df.set_index('documentId')
    per_document = df.pivot_table(index=['documentId'], aggfunc='size')
    per_document.rename({0: 'count'}, inplace=True, axis='columns')
    df = df.loc[~df.index.duplicated(keep='first')]
    df['count'] = per_document

    df_documents = df #df_documents = df.set_index('documentId')
    df_documents = df_documents.loc[~df_documents.index.duplicated(keep='first')]
    df_documents = df_documents.drop(columns=['eventId', 'activeTime', 'userId', 'time'])
    return df_documents


if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    print('Loading dataset...')
    df = load_data("active1000")
    df = pre_processing(df)

    ### Naive Bayes
    print("Naive Bayes")
    recommendations, test_sets = bernoulli_bayes(df)
    evaluate(recommendations, test_sets)

    ### User and item based collaborative filtering
    # To change between user and item based set the flag when calling collaborative_filtering()
    print("collab user / item based")
    ratings = get_ratings_matrix(df)
    train_set, test_sets = train_test_split(ratings)
    recommendations = collaborative_filtering(train_set, 20, item_based=False)
    evaluate(recommendations, test_sets)

    ### Content based nearest neighbor classifier
    print("content based nearest neighbor classifier")
    df_documents = get_unique_documents(df)

    document_ranker = DocumentRanker()

    userIds = df.userId.unique()

    predicted_documentIds = {}
    test_documentIds = {}

    for userId in tqdm(userIds):
        document_category_cosine = None
        document_title_cosine = None

        user_read_documents = df.loc[df['userId'] == userId]['documentId'].tolist()

        train_documentIds, test_documentIds[userId] = sklean_train_test_split(user_read_documents, test_size=0.2)

        for documentId in train_documentIds:
            if document_category_cosine is None:
                document_category_cosine = document_ranker.rank_documents_category_cosine(df_documents, documentId)
                document_title_cosine = document_ranker.rank_documents_title_cosine(df_documents, documentId)
            else:
                document_category_cosine += document_ranker.rank_documents_category_cosine(df_documents, documentId)
                document_title_cosine += document_ranker.rank_documents_title_cosine(df_documents, documentId)

        document_count_rank = document_ranker.rank_documents_count(df_documents)
        document_skip_rank = document_ranker.rank_document_skip(df_documents, train_documentIds)

        top_hits = np.argsort((document_title_cosine + document_category_cosine) * document_count_rank * document_skip_rank)[::-1][:len(test_documentIds[userId])]

        predicted_documentIds[userId] = df_documents.iloc[top_hits].index.tolist()

    evaluate_nearest_neighbor_classifier(predicted_documentIds, test_documentIds)
