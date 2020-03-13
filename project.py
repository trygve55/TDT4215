import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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


def prepare_data(df):
    # Remove homepage events
    print('Removing homepage hits from datatset.')
    df = df[df.url != 'http://adressa.no']

    # remove incorrectly scraped events
    print('Removing incorrectly events.')
    df = df.replace(to_replace='None', value=np.nan).dropna(subset=['documentId'])

    '''
    # binarize categories
    print('Binarizing categories')
    categories_df = df['category'].str.split('|', expand=True)
    categories = pd.unique(categories_df.values.ravel('K'))
    categories = categories[categories != None]  # Remove none from categories
    for i in tqdm(categories):
        df['category_' + i] = np.where(i in categories_df, 1, 0)
    df = df.drop(['category'], axis=1)
    df = df.fillna(1.0)
    '''

    return df


if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    print('Loading dataset...')
    df=load_data("active1000")
    df=prepare_data(df)


    #Print final dataframe
    print('Printing final dataframe')
    print(df)

    print(df.dtypes)

    df = df.set_index('documentId')
    per_document = df.pivot_table(index=['documentId'], aggfunc='size')
    per_document.rename({0:'count'}, inplace=True, axis='columns')
#    df = df.drop_duplicates(subset='documentId', keep="first")
    df = df.loc[~df.index.duplicated(keep='first')]

    df['count'] = per_document
    print(per_document)
    print(df)

    tfidf = TfidfVectorizer()
    df['category'] = df['category'].fillna('').str.split('|').astype('str')
    tfidf_matrix = tfidf.fit_transform(df['category'])
    print(tfidf.get_feature_names())
    print(tfidf_matrix.shape)

    print(np.argmax(cosine_similarity(tfidf_matrix)[1:,0]))

    #df_cat_only = df.drop(['documentId', 'title', 'url', 'eventId', 'publishtime', 'userId', 'activeTime', 'time'], axis=1)
