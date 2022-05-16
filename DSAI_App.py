import requests
import streamlit as st
import json
import traceback
import sys
from PIL import Image
from DSAI_Utility import All_Initialization,CSS_Property
from DSAI_DMV_API import API_Validation

if __name__=='__main__':
    
    
    img = Image.open('DMV_Logo.png')
    st.set_page_config(page_title="DMV Vanity Plate Analyzer", layout="wide",page_icon=img)
    try:
        # Applying CSS properties for web page
        CSS_Property("style.css")
        # Initializing Basic Componentes of Web Page
        All_Initialization()
        
        API_Validation()
    
        
        
    except BaseException as e:
        col1, col2, col3 = st.columns([1.15,7.1,1.15])
        with col2:
            st.write('')
            st.error('In Error block - '+str(e))
            traceback.print_exception(*sys.exc_info())
