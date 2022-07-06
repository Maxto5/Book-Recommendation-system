#!/usr/bin/env python
# coding: utf-8

# ## Hybrid Book Recommendation System

# **The notebook structure guideline**
#  - import python libraries needed
#  - import the datasets
#  - preprocessing
#  - Exploratory Data Analysis 
#  - Hybrid training
#    - collaborative
#    - item-based
#  - test the model

import numpy as np
import pandas as pd

books = pd.read_csv('book.csv', dtype='unicode')
users = pd.read_csv('Users.csv')
ratings = pd.read_csv('Ratings.csv')

print('drop 0 rated books')
ratings = ratings[ratings['Book-Rating'] > 0]

books.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'],inplace=True)



book_ratings = pd.merge(books,ratings,on='ISBN').drop_duplicates()
book_user_ratings = pd.merge(book_ratings,users,on='User-ID').drop_duplicates()


book_user_ratings.dropna(inplace=True)


print(book_user_ratings['Book-Title'].nunique(),'unique books' ,"\n" , book_user_ratings['User-ID'].nunique(), "unique users")


book_user_ratings = book_user_ratings.drop_duplicates(['User-ID'])
book_user_ratings


print("Split dataset")
test,train = np.split(book_user_ratings.sample(frac=1,random_state=42),[int(.8*len(book_user_ratings))])
test.shape,train.shape


print("create pivot table")
test_pivot = test.pivot(columns='User-ID',index=['Book-Title'],  values='Book-Rating').fillna(0)
test_pivot.head()


print("create sparse matrix")
from scipy.sparse import csr_matrix
test_csr =  csr_matrix(test_pivot)


print("apply knn")
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='euclidean',algorithm='auto')
model_knn.fit(test_csr)



print("Query away")
title = list(test_pivot.index)
# collaborative filtering
def collab(query):
    output = []
    dist = []
    query_index=title.index(query)
    distances, indices = model_knn.kneighbors(test_pivot.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 6 )
    for i in range(1,len(indices.flatten())):
        if i == 0:
            print("Recommendations for {0}:\n".format(test_pivot.index[indices.flatten()[i]]))
        if (i != 0):
            output.append(test_pivot.index[indices.flatten()[i]])
            dist.append(distances.flatten()[i])
            out = pd.DataFrame({'knn-Distance':dist,'Book-Title':output})
    return out

# content-based filtering

books_title = test['Book-Title']
books_title = books_title.drop_duplicates()
book_title = books_title.reset_index(drop=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

tf = TfidfVectorizer(analyzer = 'word', ngram_range = (1,2), min_df = 0, stop_words = 'english')
tfidf_matrix = tf.fit_transform(books_title)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim

indices = pd.Series(books_title.index,index = book_title)

buks=pd.DataFrame()
buks['Book-Title'] = book_title
serie = pd.DataFrame()
serie['series']=indices
fin = pd.merge(buks,serie,on='Book-Title')

def content(query):
    idx = int(fin[fin['Book-Title']== query].index.values)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key = lambda x:x[1], reverse = True)
    sim_scores = sim_scores[1:5+1]
    book_indices = [i[0] for i in sim_scores]

    x=pd.DataFrame(sim_scores)
    z=pd.DataFrame(buks.iloc[book_indices]).values
    out=pd.DataFrame()
    out['cosine_sim'],out['title'] = x[1],z
    return out
    
