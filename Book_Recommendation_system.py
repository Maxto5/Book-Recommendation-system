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

#books.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'],inplace=True)



book_ratings = pd.merge(books,ratings,on='ISBN').drop_duplicates()
book_user_ratings = pd.merge(book_ratings,users,on='User-ID').drop_duplicates()


book_user_ratings.dropna(inplace=True)


print(book_user_ratings['Book-Title'].nunique(),'unique books' ,"\n" , book_user_ratings['User-ID'].nunique(), "unique users")


book_user_ratings = book_user_ratings.drop_duplicates(['User-ID'])
book_user_ratings


print("Split dataset")
train,validate,test = np.split(book_user_ratings.sample(frac=1,random_state=42),[int(.6*len(book_user_ratings)),int(.8*len(book_user_ratings))])
train.shape,validate.shape,test.shape
train


print("create pivot table")
train_pivot = train.pivot(columns='User-ID',index=['Book-Title'],  values='Book-Rating').fillna(0)
train_pivot.head()


print("create sparse matrix")
import scipy
from scipy.sparse import csr_matrix
train_csr =  csr_matrix(train_pivot)


print("apply knn")
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='euclidean',algorithm='auto')
model_knn.fit(train_csr)



print("Query away")
title = list(train_pivot.index)

def collector(query):
    output = []
    dist = []
    query_index=title.index(query)
    distances, indices = model_knn.kneighbors(train_pivot.iloc[query_index,:].values.reshape(1,-1), n_neighbors = 6 )
    for i in range(1,len(indices.flatten())):
        if i == 0:
            print("Recommendations for {0}:\n".format(train_pivot.index[indices.flatten()[i]]))
        if (i != 0):
            output.append(train_pivot.index[indices.flatten()[i]])
            dist.append(distances.flatten()[i])
            out = pd.DataFrame({'knn-Distance':dist,'Book-Title':output})
    return out


