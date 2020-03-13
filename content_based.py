import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import BernoulliNB

def train_test_split_bernoulli(df, sim):
    users = df["userId"].unique()
    users_df = df.groupby(["userId"])

    #TODO Countvectorizer på categorianj aktiv bruker har lest. Normaliser for å få tall mellom 0 og 1. Finn similarity og legg te dem som e mest ulik

    all_categories_vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(df["category"])
    # all_categories_vectorizer.fit(df["category"])
    categories = all_categories_vectorizer.get_feature_names()
    attribute_vectorizer = CountVectorizer(ngram_range=(1, 2), vocabulary=dict(zip(categories, range(len(categories)))))

    for user in users:
        user_df = users_df.get_group(user)
        num_train = round(user_df.shape[0] * 0.8)
        user_df_train = user_df[:num_train]
        user_df_test = user_df[num_train:]


        # Randomly choose articles that the users hasn't seen and add them to the training set as entries for the 0-class
        # This is an assumption that adds bias, look at alternative ways of a) choosing the entries, and b) determining size of 0-class elements comparet to seen articles
        df_unseen = df[~df["tid"].isin(user_df["tid"])]
        zero_class_sample = df_unseen.sample(n=user_df.shape[0])
        zero_class_train_sample = zero_class_sample[:num_train]
        zero_class_test_sample = zero_class_sample[num_train:]

        print(user_df.shape)
        print(zero_class_train_sample.shape)
        user_df_train = pd.concat([user_df_train, zero_class_train_sample])
        user_df_test = pd.concat([user_df_test, zero_class_test_sample])
        # print(user_df.shape)

        X_train = attribute_vectorizer.fit_transform(user_df_train["category"]).toarray()
        print(attribute_vectorizer.get_feature_names())
        X_test = attribute_vectorizer.fit_transform(user_df_test["category"]).toarray()
        print(attribute_vectorizer.get_feature_names())
        # print(X.shape)
        # print(X)
        # print(X.toarray())
        #
        # X = X.toarray()
        y_train = np.zeros(X_train.shape[0])
        y_train[:num_train] = 1

        y_test = np.zeros(X_test.shape[0])
        y_test[:user_df.shape[0] - num_train] = 1


        model = BernoulliNB()
        model.fit(X_train, y_train)
        # score = model.score(X_test, y_test)
        # print(score)
        predicted_values = model.predict_proba(X_test)
        print(predicted_values)

        # Perhaps useful later
        # X = np.sum(X, axis=0)
        # X = X / np.max(X)
        # print(X.shape)
        # print(X)



def bernoulli_bayes(df):
    # train_attributes, train_class = pre_process(df)

    df['category'] = df['category'].fillna("")
    df_item = df[['tid', 'category']].drop_duplicates(inplace=False)
    df_item.sort_values(by=['tid', 'category'], ascending=True, inplace=True)

    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
    tfidf_matrix = tf.fit_transform(df_item['category'])
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)

    train_test_split_bernoulli(df, cosine_similarity)

    print(df_item)

    # vectorizer = HashingVectorizer(n_features=169)
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(df_item["category"])
    print(X.shape)
    print(X)
    model = BernoulliNB()
    model.fit(X, [1 for _ in range(X.shape[0])])
    # model.predict()
    print(model)


def pre_process(df):
    df_categories = df["category"]
    df_categories["tid"] = pd.Series([range(1, len(df))])
    print(df_categories)

    train_attributes, train_class = 0, 0
    return train_attributes, train_class
