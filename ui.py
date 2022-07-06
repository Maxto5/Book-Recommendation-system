import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import Book_Recommendation_system
st.set_page_config(layout="wide")
st.title("Book Recommendation System")
st.subheader("Hybrid Collaborative & Content filtering")

title = st.text_input('Book TItle')

button=st.button("submit")

col1,col2 = st.columns((1,1))

with col1:
    st.write("Book-titles")
    st.dataframe(Book_Recommendation_system.book_title)
with col2:
    if title and button:
       colab_output = Book_Recommendation_system.collab(title)
       cont_output = Book_Recommendation_system.content(title)
       st.write("Recom based on Similar user ratings:\t",title)
       st.dataframe(colab_output)
       st.write("Recom based on Title similarity:\t",title)
       st.dataframe(cont_output)
components.html("""<center> BCSC01/0043/2018 &nbsp Barasa Mathews &nbsp &copy 2022 <center>""")

       
   




