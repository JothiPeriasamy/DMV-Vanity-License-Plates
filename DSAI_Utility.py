# Copyright (c) DeepSphere.AI 2021

# All rights reserved

# We are sharing this partial code for learning and research, and the idea behind us sharing the source code is to stimulate ideas #and thoughts for the learners to develop their MLOps.

# Author: # DeepSphere.AI | deepsphere.ai | dsschoolofai.com | info@deepsphere.ai

# Release: Initial release

""

import streamlit as st
from PIL import Image

def CSS_Property(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def All_Initialization():
    image = Image.open('DMV_Logo.png')
    st.image(image)
    st.markdown("<h1 style='text-align: center; color: black; font-size:25px;'>DMV - API Client for Vanity License Plate Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("""
    <hr style="width:100%;height:3px;background-color:gray;border-width:10">
    """, unsafe_allow_html=True)
    choice1 =  st.sidebar.selectbox(" ",('Home','About Us'))
    choice2 =  st.sidebar.selectbox(" ",('Libraries in Scope','SQLAlchemy','Psycopg2', 'Keras','Tensorflow','Transformers','Pandas','Streamlit'))
    choice3 =  st.sidebar.selectbox(" ",('Models Implemented','Naive Bayes Classifier', 'Random Forest Classifier', 'LSTM RNN-Deep Learning','BERT Model','XLM-Roberta Model'))
    menu = ["Google Cloud Services in Scope","Cloud Storage", "Bigquery", "Cloud Run", "Cloud Function", "Pubsub", "Vertex AI", "Secret Manager"]
    choice = st.sidebar.selectbox(" ",menu)
    st.sidebar.write('')
    st.sidebar.write('')
    href = f'<a style="color:black;text-align: center;" href="" class="button" target="_self">Clear/Reset</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)
