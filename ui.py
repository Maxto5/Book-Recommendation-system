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
col3,col4 = st.columns((9,1))
col5,col6 = st.columns((1,1))
with col1:
    st.write("Book-titles")
    st.dataframe(Book_Recommendation_system.book_title)
    
#final content dataframe
buku = Book_Recommendation_system.train.drop_duplicates(subset='Book-Title',keep='first')
buku = buku.drop(['User-ID','Age','Location','Image-URL-M','Image-URL-L'],axis=1)

if title and button:
    colab_output = Book_Recommendation_system.collab(title)
    cont_output = Book_Recommendation_system.content(title)
    reco_list=list(colab_output['title']) + list(cont_output['title'])
    
        
    with col2:
        #collaborative filtering
        st.write("Recom based on Similar user ratings:\t",title)
        st.dataframe(colab_output)
    
        #content-based filtering
        st.write("Recom based on Title similarity:\t",title)
        st.dataframe(cont_output)
    
    with col3:
    #output dataframe
        st.write('Detailed Recommendation')
        fin_df = buku[buku['Book-Title'].isin(reco_list)]
        st.dataframe(fin_df.reset_index(drop=True))
        
    
with col5:
    st.write('Rating distribution')
    st.pyplot(Book_Recommendation_system.dis)

    
with col6:
    st.write('pairplot graphs')
    st.pyplot(Book_Recommendation_system.pp)
    
components.html("""<center> BCSC01/0043/2018 &nbsp Barasa Mathews &nbsp &copy 2022 <center>""")

   
