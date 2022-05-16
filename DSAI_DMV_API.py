import streamlit as st
import requests
import json
from google.cloud import storage
import datetime


def API_Validation():

    col1, col2, col23, col4,col5 = st.columns([1.5,4,1,4,1.5])
    with col2:
        st.write('')
        st.write('')
        st.subheader('API Request URL')
        st.write('')
        st.write('')
        st.subheader('ELP Configuration')
        st.write('')
        st.write('')
        st.write('')
        st.subheader('Select Model')
        st.write('')
        st.write('')
        st.subheader('Vehicle Identification Number')
    with col4:
        vAR_request_url = st.text_input('Request URL','https://dsai-dmv-elp-classification-ht2tn6zcxa-uc.a.run.app')
        vAR_input_text = st.text_input('Enter input','',max_chars=7).upper().strip()
        vAR_model = st.selectbox('',('Select Model','BERT','RNN'))
        vAR_vin = st.text_input('Enter vin','456').strip()
    if len(vAR_input_text)>0 and vAR_model!='Select Model' and len(vAR_vin)>0 and vAR_request_url=='https://dsai-dmv-elp-classification-ht2tn6zcxa-uc.a.run.app':
        vAR_payload = {"message":vAR_input_text,"model":vAR_model,"vin":vAR_vin}

        vAR_headers = {'content-type': 'application/json'}
        col1, col2, col3 = st.columns([1.15,7.1,1.15])
        with col2:
            st.write('')
            st.write('')
            with st.spinner('Sending Request and Waiting for Response'):
                vAR_request = requests.post(vAR_request_url, data=json.dumps(vAR_payload),headers=vAR_headers)
        
        print(vAR_request.text)
        
        with col2:
            vAR_result = vAR_request.text
            st.write('')
            st.write('')
            st.json(vAR_result)
        
            Upload_Response_GCS(vAR_result)
    else:
        col1, col2, col3 = st.columns([1.15,7.1,1.15])
        with col2:
            st.write('')
            st.warning('Please enter correct input details')


def Upload_Response_GCS(vAR_result):
    with st.spinner('Saving Json Response to Cloud Storage'):
        vAR_bucket_name = 'dsai_saved_models'
        vAR_bucket = storage.Client().get_bucket(vAR_bucket_name)
        # define a dummy dict
        vAR_json_object = vAR_result
        vAR_utc_time = datetime.datetime.utcnow()
        blob = vAR_bucket.blob('DSAI_DMV_API_RESULTS/dmv_api_result_'+vAR_utc_time.strftime('%Y%m%d %H%M%S')+'.json')
        blob.upload_from_string(data=json.dumps(vAR_json_object),content_type='application/json')
        st.write('')
        st.success('API Response successfully saved into cloud storage')
