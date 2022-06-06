import requests
import streamlit as st
import json
import traceback
import sys
from PIL import Image

from Utility.DMV_UT_Streamlit import All_Initialization,CSS_Property
from User_Interface.DMV_API import API_Validation

if __name__=='__main__':
    
    
    img = Image.open('Utility/DMV_UT_Logo.png')
    st.set_page_config(page_title="DMV Vanity Plate Analyzer", layout="wide",page_icon=img)
    try:
        # Applying CSS properties for web page
        CSS_Property("Utility/DMV_UT_Style.css")
        # Initializing Basic Componentes of Web Page
        All_Initialization()
        
        API_Validation()
    
        
        
    except BaseException as e:
        col1, col2, col3 = st.columns([1.5,11,1.5])
        with col2:
            st.write('')
            st.error('In Error block - '+str(e))
            traceback.print_exception(*sys.exc_info())
