import json
import os
from tqdm import tqdm
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


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
    df = pd.DataFrame(map_lst)
    df.index.name = 'documentId'
    return df


if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    print('Loading dataset...')
    df=load_data("active1000")

    #Remove homepage events
    print('Removing homepage hits from datatset.')
    df = df[df.url != 'http://adressa.no']

    #remove incorrectly scraped events
    print('Removing incorrectly events.')
    df = df.replace(to_replace='None', value=np.nan).dropna(subset=['documentId'])

    #binarize categories
    print('Binarizing categories')
    categories_df = df['category'].str.split('|', expand=True)
    categories = pd.unique(categories_df.values.ravel('K'))
    categories = categories[categories != None] #Remove none from categories
    for i in tqdm(categories):
        df['category_' + i] = np.where(i == categories_df[0], 1, np.where(i == categories_df[1], 1, 0))
    df = df.drop(['category'], axis=1)

    #Print final dataframe
    print('Printing final dataframe')

    df_cat_only = df.drop(['documentId', 'title', 'url', 'eventId', 'publishtime', 'userId', 'activeTime', 'time'], axis=1)

    print(df_cat_only)

    kmeans = KMeans(n_clusters=3).fit(df_cat_only)
    centroids = kmeans.cluster_centers_
    print(centroids)

    pca = PCA(n_components=2).fit(df_cat_only)

    print(pca)

    pca_d = pca.transform(df_cat_only)
    centroids_pca = pca.transform(centroids)

    print(pca_d)

    plt.scatter(pca_d[:, 0], pca_d[:, 1], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    plt.show()



