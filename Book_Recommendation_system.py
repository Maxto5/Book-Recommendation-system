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



books = pd.read_csv('books.csv',on_bad_lines='skip',dtype='unicode')
users = pd.read_csv('Users.csv',on_bad_lines='skip')
ratings = pd.read_csv('Ratings.csv',on_bad_lines='skip')

print('drop 0 rated books')
ratings = ratings[ratings['Book-Rating'] > 0]

#books.drop(columns=['Image-URL-S','Image-URL-M','Image-URL-L'],inplace=True)


book_ratings = pd.merge(books,ratings,on='ISBN').drop_duplicates()
book_user_ratings = pd.merge(book_ratings,users,on='User-ID').drop_duplicates()


book_user_ratings.dropna(inplace=True)


print(book_user_ratings['Book-Title'].nunique(),'unique books' ,"\n" , book_user_ratings['User-ID'].nunique(), "unique users")


book_user_ratings = book_user_ratings.drop_duplicates(['User-ID'])
book_user_ratings

import matplotlib.pyplot as plt
import seaborn as sns

print('visualization')

fig = plt.figure()
dis = sns.displot(book_user_ratings['Book-Rating'])
fig.savefig('rating.png')


fig3=plt.figure()
count=book_user_ratings.groupby('Book-Title')['Book-Rating'].count()
avg=book_user_ratings.groupby('Book-Title')['Book-Rating'].mean()
ax=sns.jointplot(x =count,y=avg)
ax.set_axis_labels('number of ratings','rating')
fig3.savefig('rater.png')

fig4 = plt.figure()
pp = sns.pairplot(book_user_ratings)
fig4.savefig('rating.png')

fig2 = plt.figure()
users.Age.hist(bins=[0,10,20,30,40,50,60,70,100])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
fig2.savefig('age.png')

print("Split dataset")
train,test = np.split(book_user_ratings.sample(frac=1,random_state=42),[int(.8*len(book_user_ratings))])
train.shape,test.shape


print("create pivot table")
train_pivot = train.pivot(columns='User-ID',index=['Book-Title'],values='Book-Rating').fillna(0)
train_pivot.head()


print("create sparse matrix")
from scipy.sparse import csr_matrix
train_csr =  csr_matrix(train_pivot)


print("apply knn")
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric='euclidean',algorithm='auto')
model_knn.fit(train_csr)



print("Query away")
title = list(train_pivot.index)
# collaborative filtering
def collab(query):
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
            out = pd.DataFrame({'knn-Distance':dist,'title':output})
            

    return out

# content-based filtering

books_title = train['Book-Title']
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
    out = out[out['cosine_sim'] > 0]
    return out
    


