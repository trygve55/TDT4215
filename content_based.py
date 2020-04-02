from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.naive_bayes import BernoulliNB
from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer("norwegian")

stop_words_norwegian = ['alle', 'andre', 'arbeid', 'at', 'av', 'bare', 'begge', 'ble', 'blei', 'bli', 'blir', 'blitt',
                       'bort', 'bra', 'bruke', 'både', 'båe', 'da', 'de', 'deg', 'dei', 'deim', 'deira', 'deires',
                       'dem', 'den', 'denne', 'der', 'dere', 'deres', 'det', 'dette', 'di', 'din', 'disse', 'ditt',
                       'du', 'dykk', 'dykkar', 'då', 'eg', 'ein', 'eit', 'eitt', 'eller', 'elles', 'en', 'ene',
                       'eneste', 'enhver', 'enn', 'er', 'et', 'ett', 'etter', 'folk', 'for', 'fordi', 'forsûke', 'fra',
                       'få', 'før', 'fûr', 'fûrst', 'gjorde', 'gjûre', 'god', 'gå', 'ha', 'hadde', 'han', 'hans', 'har',
                       'hennar', 'henne', 'hennes', 'her', 'hjå', 'ho', 'hoe', 'honom', 'hoss', 'hossen', 'hun', 'hva',
                       'hvem', 'hver', 'hvilke', 'hvilken', 'hvis', 'hvor', 'hvordan', 'hvorfor', 'i', 'ikke', 'ikkje',
                       'ingen', 'ingi', 'inkje', 'inn', 'innen', 'inni', 'ja', 'jeg', 'kan', 'kom', 'korleis', 'korso',
                       'kun', 'kunne', 'kva', 'kvar', 'kvarhelst', 'kven', 'kvi', 'kvifor', 'lage', 'lang', 'lik',
                       'like', 'makt', 'man', 'mange', 'me', 'med', 'medan', 'meg', 'meget', 'mellom', 'men', 'mens',
                       'mer', 'mest', 'mi', 'min', 'mine', 'mitt', 'mot', 'mye', 'mykje', 'må', 'måte', 'navn', 'ned',
                       'nei', 'no', 'noe', 'noen', 'noka', 'noko', 'nokon', 'nokor', 'nokre', 'ny', 'nå', 'når', 'og',
                       'også', 'om', 'opp', 'oss', 'over', 'part', 'punkt', 'på', 'rett', 'riktig', 'samme', 'sant',
                       'seg', 'selv', 'si', 'sia', 'sidan', 'siden', 'sin', 'sine', 'sist', 'sitt', 'sjøl', 'skal',
                       'skulle', 'slik', 'slutt', 'so', 'som', 'somme', 'somt', 'start', 'stille', 'så', 'sånn', 'tid',
                       'til', 'tilbake', 'tilstand', 'um', 'under', 'upp', 'ut', 'uten', 'var', 'vart', 'varte', 'ved',
                       'verdi', 'vere', 'verte', 'vi', 'vil', 'ville', 'vite', 'vore', 'vors', 'vort', 'vår', 'være',
                       'vært', 'vöre', 'vört', 'å'#]
                        , '000', '10', '10 000', '100', '100 000', '1000', '11', '12', '120', '13', '14',
                        '15', '15 000', '16', '17', '18', '19', '20', '200', '2000', '21', '22', '23', '24', '25',
                        '250', '26', '27', '28', '29', '30', '300', '32', '35', '38', '39', '40', '400', '42', '43',
                        '45', '50', '500', '60', '600', '70', '700', '80', '90', '-']


def bernoulli_bayes(df, k=20):
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


def rank_documents_popularity(df):
    pass


def rank_documents_title_cosine(df, documentId):
    tf = TfidfVectorizer(
        analyzer='word',
        tokenizer=tokenize,
        ngram_range=(1, 2),
        min_df=10,
        #token_pattern='(?u)\b[A-Za-z]+\b',
        stop_words=stop_words_norwegian)

    tfidf_matrix = tf.fit_transform(df['title'])
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(tf.get_feature_names())
    print(tfidf_matrix.shape)

    document_index = df.index.get_loc(documentId)

    return cosine_similarity[:, document_index]


def rank_documents_category_cosine(df_documents, documentId):
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0)
    print(df_documents)
    tfidf_matrix = tf.fit_transform(df_documents['category'])
    cosine_similarity = linear_kernel(tfidf_matrix, tfidf_matrix)
    print(tf.get_feature_names())
    print(tfidf_matrix.shape)

    document_index = df_documents.index.get_loc(documentId)

    return cosine_similarity[:, document_index]


def rank_documents_count(df_documents):
    return np.log(df_documents['count'])/10 + 1


def tokenize(text):
    return [stemmer.stem(word) for word in text.split(' ')]
