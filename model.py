import pandas as pd
import numpy as np
df=pd.read_csv('Datasets/imdb_cleaned.csv')

df.drop(['Unnamed: 0','gross'],axis=1,inplace=True)
df['imdb_score']=df['imdb_score'].fillna(np.mean(df['imdb_score'])).round(1)
df['votes']=df['votes'].fillna(np.mean(df['votes']))
df['runtime']=df['runtime'].fillna(np.mean(df['runtime']))

df['rating']=df['rating'].fillna(method='ffill')
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['description'])

tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

data = pd.Series(df['description'], index=df.index)
data = pd.DataFrame(data)

import re

class ItemRecommender:
    def __init__(self):
        self.data = data
        self.cosine_sim = cosine_sim
        
    def recommendation(self, keyword):
        index = self.data[self.data['description'].str.contains(keyword, flags=re.IGNORECASE, regex=True)].index[0]
        sim_score = list(enumerate(self.cosine_sim[index]))    
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)

        sim_score = sim_score[1:6]
        final_index = [i[0] for i in sim_score]
        return final_index
    
    def predict(self,ram):
        idx = self.recommendation(ram)
        b=pd.DataFrame()
        b['Title']=df['movie_name'].iloc[idx]
        b['rating']=df['imdb_score'].iloc[idx]
        return b
    
idx=ItemRecommender()

import pickle
pickle.dump(idx,open('model.pkl','wb'))