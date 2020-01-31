import json
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 160)

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

    
if __name__ == '__main__':
    #Load dataset into a Pandas dataframe
    df=load_data("active1000")
    #Remove homepage events
    df = df[df.url != 'http://adressa.no']
    #remove 404s
    df = df.replace(to_replace='None', value=np.nan).dropna(subset=['documentId'])
    print(df)
    
    
    
    
    
    
    
    
    
    
    
