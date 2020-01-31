import json
import os
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

def set_category_row(row, categories_in):
    def label_race(row):
        if row['eri_hispanic'] == 1:
            return 'Hispanic'
        if row['eri_afr_amer'] + row['eri_asian'] + row['eri_hawaiian'] + row['eri_nat_amer'] + row['eri_white'] > 1:
            return 'Two Or More'
        if row['eri_nat_amer'] == 1:
            return 'A/I AK Native'
        if row['eri_asian'] == 1:
            return 'Asian'
        if row['eri_afr_amer'] == 1:
            return 'Black/AA'
        if row['eri_hawaiian'] == 1:
            return 'Haw/Pac Isl.'
        if row['eri_white'] == 1:
            return 'White'
        return 'Other'
    
if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    print('Loading dataset...')
    df=load_data("active1000")

    #Remove homepage events
    print('Removing homepage hits from datatset.')
    df = df[df.url != 'http://adressa.no']

    #remove 404s
    print('Removing 404 articles.')
    df = df.replace(to_replace='None', value=np.nan).dropna(subset=['documentId'])

    #binarize categories
    print('Binarizing categories')
    categories_df = df['category'].str.split('|', expand=True)
    categories = pd.unique(categories_df.values.ravel('K'))
    categories = categories[categories != None] #Remove none from categories
    for i in tqdm(categories):
        df['category_' + i] = np.where(i in categories_df, 1, 0)
    df = df.drop(['category'], axis=1)

    #Print final dataframe
    print('Printing final dataframe')
    print(df)
    
    
    
    
    
    
    
    
    
    
    
