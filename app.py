import streamlit as st
import pickle as pk

import numpy as np
import pandas as pd

from predict import text_processing

pca = pk.load(open("pca.pkl",'rb'))
gnb = pk.load(open("gnb.pkl",'rb'))
vec = pk.load(open("vectorizer.pkl",'rb'))

st.title('Email Classification')


user_input = st.text_area(label='Enter some text')


# option = st.selectbox('NLP Service',('Classification','Email-Classification'))

if st.button('Predict'):
    d= pd.DataFrame()
    d['text'] = [user_input]
    st.write('processing text')
    processed_text = text_processing(d)
    tf_idf_text = vec.transform(d['text']).toarray()
    st.write('text processed')
    pca_reduced_text = pca.transform(tf_idf_text)

    label = gnb.predict(pca_reduced_text)
    if label ==1:
        st.success('Email is not SPAM')
    else:
        st.success('Email is SPAM')