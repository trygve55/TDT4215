from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import mean_squared_error, recall_score

def get_document_attribute_matrix(df):
    # Creates an M x N matrix with M unique documents and N features
    df_docs = df.copy()
    df_docs.drop_duplicates(["tid"], inplace=True)
    return CountVectorizer(ngram_range=(1, 2)).fit_transform(df_docs["category"]).toarray()

def train_test_split_bernoulli(df, k=20):
    users = df["userId"].unique()
    users_df = df.groupby(["userId"])

    categories = CountVectorizer(ngram_range=(1, 2)).fit(df["category"]).get_feature_names()
    attribute_vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=dict(zip(categories, range(len(categories)))))

    df_unique_docs = df.copy()
    df_unique_docs.drop_duplicates(["tid"], inplace=True)

    recommendations = np.empty((users.shape[0], k))
    test_sets = np.empty((users.shape[0], df_unique_docs.shape[0]))

    for idx, user in enumerate(tqdm(users)):
        user_df = users_df.get_group(user)  # Every observation from the current user
        num_train = round(user_df.shape[0] * 0.8)  # Use 80% of observations for training
        user_df_train = user_df[:num_train]
        test_set = np.zeros((df_unique_docs.shape[0],))
        test_set_docs = user_df[num_train:]["tid"].to_numpy()
        test_set[test_set_docs] = 1


        # Randomly choose articles that the users hasn't seen and add them to the training set as entries for the 0-class
        # This is an assumption that adds bias, look at alternative ways of a) choosing the entries, and b) determining size of 0-class elements compared to seen articles
        # This also mean that the unseen articles used cannot be recommended.
        df_unseen = df_unique_docs[~df_unique_docs["tid"].isin(user_df["tid"])]  # The observations of unique articles the user hasn't seen
        zero_class_sample = df_unseen.sample(n=user_df.shape[0])  # A random sample of these observations (as many as articles the user has seen).
        zero_class_train_sample = zero_class_sample[:num_train]

        user_df_train = pd.concat([user_df_train, zero_class_train_sample])  # Combining the 1 and 0 classes
        user_df_test = pd.concat([df_unseen, zero_class_train_sample]).drop_duplicates(keep=False)  # Drops observations seen in both df_unseen and zero_class_train_sample

        X_train = attribute_vectorizer.fit_transform(user_df_train["category"]).toarray()
        X_test = attribute_vectorizer.fit_transform(user_df_test["category"]).toarray()

        y_train = np.zeros(X_train.shape[0])
        y_train[:num_train] = 1

        model = BernoulliNB()
        model.fit(X_train, y_train)

        predicted_values = model.predict_proba(X_test)[:, 1]
        predicted_values = np.array([(i, predicted_values[i]) for i in range(len(predicted_values))])
        predicted_values = sorted(predicted_values, key=lambda x: x[1], reverse=True)[:k]
        predicted_values = np.array([i for i, j in predicted_values])

        recommendations[idx] = predicted_values
        test_sets[idx] = test_set
    return recommendations, test_sets



def bernoulli_bayes(df):
    # train_attributes, train_class = pre_process(df)

    # Bli gjort i preprocessing i starten av programmet i ny commit
    df['category'] = df['category'].fillna("")


    return train_test_split_bernoulli(df)
