import streamlit as st
import pandas as pd
import Book_Recommendation_system
st.set_page_config(layout="wide")
st.title("Book Recommendation System")
st.subheader("Collaborative filtering using KNN")

title = st.text_input('Book TItle')

button=st.button("submit")

col1,col2 = st.columns((1,1))

with col1:
    st.write("Book-titles")
    st.dataframe(Book_Recommendation_system.train_pivot.index.unique())
with col2:
    if title and button:
       output = Book_Recommendation_system.collector(title)
       st.write("Recommendation for:\t",title)
       #st.dataframe(Book_Recommendation_system.book_user_ratings)
       #out = pd.DataFrame(output,columns=['Book-Title'])
       st.dataframe(output)
       
   




