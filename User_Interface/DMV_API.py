import streamlit as st
import requests
import json
from google.cloud import storage,bigquery
import datetime
import pandas as pd
from random import randint
from io import StringIO 
import boto3
import os


def API_Validation():

    col1, col2, col23, col4,col5 = st.columns([1.5,5,1,5,1.5])
    vAR_output = pd.DataFrame()
    vAR_batch_input = None
    vAR_api_request_url = 'https://dsai-dmv-elp-classification-ht2tn6zcxa-uc.a.run.app'
    with col2:
        st.write('')
        st.write('')
        st.subheader('API Request URL')
        st.write('')
        st.write('')
        st.write('')
        st.subheader('ELP Batch Request Source')
        
        
    with col4:
        vAR_request_url = st.text_input('Request URL',vAR_api_request_url)
        st.write('')
        # vAR_batch_input = st.file_uploader('Upload Batch Configuration',type='csv')
        # st.write('')
        vAR_source = st.selectbox('',('Select Request Source','Upload From Your System','S3'))
        
    col20, col21, col22, col23,col24 = st.columns([1.5,5,1,5,1.5])
    if vAR_source=='Upload From Your System':
        with col21:
            st.write('')
            st.write('')
            st.write('')
            st.subheader('ELP Configuration')
        with col23:
            st.write('')
            vAR_batch_input = st.file_uploader('Upload Batch Configuration',type='csv')
    elif vAR_source=='S3':
        with col21:
            st.write('')
            st.write('')
            st.write('')
            st.subheader('ELP Configuration')
            st.write('')
        with col23:
            st.write('')
            vAR_batch_input = st.selectbox('',('s3://s3-us-west-2-elp/batch/simpligov/new/ELP_Project_Request/ELP_Batch_Input_Small.csv','s3://s3-us-west-2-elp/batch/simpligov/new/ELP_Project_Request/ELP_Batch_Input.csv','s3://s3-us-west-2-elp/batch/simpligov/new/ELP_Project_Request/ELP_Batch_Input2.csv'),help="select anyone")
    if vAR_batch_input is not None:
        vAR_headers = {'content-type': 'application/json'}
        vAR_batch_elp_configuration = pd.read_csv(vAR_batch_input)
        vAR_number_of_configuration = len(vAR_batch_elp_configuration)
        col6, col7, col8 = st.columns([1.5,11,1.5])
        with col7:
            st.write('')
            with st.expander("Preview Uploaded ELP Configuration"):
                st.dataframe(vAR_batch_elp_configuration)
        col9, col10, col11 = st.columns([1.5,11,1.5])
        with col10:
            st.write('')
            st.write('')
            vAR_submit_button = st.button('Upload ELP Configuration File to GCS For Batch Processing')
        if vAR_submit_button:
            if vAR_request_url==vAR_api_request_url:
                vAR_file_path = Upload_Request_GCS(vAR_batch_elp_configuration)
                col1, col2, col3 = st.columns([1.5,11,1.5])
                with col2:
                    st.write('')
                    st.write('')
                    with st.spinner('Sending Request and Waiting for the Response, It may take some time!!!'):
                        for elp_idx in  range(vAR_number_of_configuration):
                            configuration = vAR_batch_elp_configuration['CONFIGURATION'][elp_idx].replace('/','')
                            vAR_model = "RNN"
                            vAR_payload = {"message":configuration,"model":vAR_model,"vin":str(vAR_batch_elp_configuration['VIN'][elp_idx])}

                            vAR_request = requests.post(vAR_request_url, data=json.dumps(vAR_payload),headers=vAR_headers)
                            
                            vAR_result = vAR_request.text #Getting response as str
                            vAR_result = json.loads(vAR_result) #converting str to dict
                            if "Error Message" in vAR_result.keys():
                                st.error('Below Error in Order Id - '+str(vAR_batch_elp_configuration['ORDER ID'][elp_idx]))
                                st.json(vAR_result)
                            else:
                                print('Order Id - '+str(vAR_batch_elp_configuration['ORDER ID'][elp_idx])+' Successfully processed')
                                vAR_response_dict = Process_API_Response(vAR_result,vAR_batch_elp_configuration['REQUEST DATE'][elp_idx],vAR_batch_elp_configuration['ORDER DATE'][elp_idx],vAR_batch_elp_configuration['CONFIGURATION'][elp_idx],vAR_batch_elp_configuration['ORDER ID'][elp_idx],vAR_batch_elp_configuration['VIN'][elp_idx],vAR_model)
                                vAR_output = vAR_output.append(vAR_response_dict,ignore_index=True) 
                vAR_output_copy = vAR_output.copy(deep=True)

                vAR_output = vAR_output.to_csv()

                # save response into gcs
                Upload_Response_GCS(vAR_output)
                
                vAR_request_id = randint(10001, 50000)
                
                # Bigquey table insert
                Insert_Response_to_Bigquery(vAR_output_copy,vAR_request_id)
                
                # save response into s3 bucket
                Upload_Response_To_S3(vAR_output_copy,vAR_request_id)

                col6, col7, col8 = st.columns([1.5,11,1.5])
                with col7:
                    st.write('')
                    st.download_button(
         label="Download response as CSV",
         data=vAR_output,
         file_name='ELP Response.csv',
         mime='text/csv',
     )
            else:
                col1, col2, col3 = st.columns([1.5,11,1.5])
                with col2:
                    st.write('')
                    st.warning('Please enter all the input details')
                               
    else:
        col1, col2, col3 = st.columns([1.5,11,1.5])
        with col2:
            st.write('')
            st.warning('Please enter all the input details')


def Upload_Response_GCS(vAR_result):
    
    col1, col2, col3 = st.columns([1.5,11,1.5])
    with col2:
        with st.spinner('Saving API Response to Cloud Storage'):
            vAR_bucket_name = 'dsai_saved_models'
            vAR_bucket = storage.Client().get_bucket(vAR_bucket_name)
            # define a dummy dict
            vAR_utc_time = datetime.datetime.utcnow()
            client = storage.Client()
            bucket = client.get_bucket(vAR_bucket_name)
            bucket.blob('DSAI_DMV_API_RESULTS/dmv_api_result_'+vAR_utc_time.strftime('%Y%m%d %H%M%S')+'.csv').upload_from_string(vAR_result, 'text/csv')
            st.success('API Response successfully saved into cloud storage')
            
def Upload_Request_GCS(vAR_request):
    vAR_request = vAR_request.to_csv()
    col1, col2, col3 = st.columns([1.5,11,1.5])
    with col2:
        with st.spinner('Saving ELP configuration Request to Cloud Storage'):
            vAR_bucket_name = 'dsai_saved_models'
            vAR_bucket = storage.Client().get_bucket(vAR_bucket_name)
            # define a dummy dict
            vAR_utc_time = datetime.datetime.utcnow()
            client = storage.Client()
            bucket = client.get_bucket(vAR_bucket_name)
            vAR_file_path = 'DSAI_DMV_API_REQUESTS/'+vAR_utc_time.strftime('%Y%m%d')+'/dmv_api_request_'+vAR_utc_time.strftime('%H%M%S')+'.csv'
            bucket.blob(vAR_file_path).upload_from_string(vAR_request, 'text/csv')
            st.write('')
            st.write('')
            st.success('ELP Configuration Request successfully saved into cloud storage')
            return vAR_file_path

def Insert_Response_to_Bigquery(vAR_df,vAR_request_id):
    vAR_df.rename(columns = {'REQUEST DATE':'REQUEST_DATE','ORDER DATE':'ORDER_DATE','ORDER ID':'ORDER_ID','DIRECT PROFANITY':'DIRECT_PROFANITY',
'DIRECT PROFANITY MESSAGE':'DIRECT_PROFANITY_MESSAGE','RULE-BASED CLASSIFICATION':'RULE_BASED_CLASSIFICATION','RULE-BASED CLASSIFICATION MESSAGE':'RULE_BASED_CLASSIFICATION_MESSAGE','SEVERE TOXIC':'SEVERE_TOXIC','IDENTITY HATE':'IDENTITY_HATE','OVERALL PROBABILITY':'OVERALL_PROBABILITY'
}, inplace = True)
    vAR_request_ids = []
    created_at = []
    created_by = []
    updated_at = []
    updated_by = []
    df_length = len(vAR_df)
    vAR_request_ids += df_length * [vAR_request_id]
    created_at += df_length * [datetime.datetime.utcnow()]
    created_by += df_length * ['Streamlit-User']
    updated_by += df_length * ['']
    updated_at += df_length * ['']
    vAR_df['REQUEST_ID'] = vAR_request_ids
    vAR_df['CREATED_AT'] = created_at
    vAR_df['CREATED_BY'] = created_by
    vAR_df['UPDATED_AT'] = updated_at
    vAR_df['UPDATED_BY'] = updated_by
    col1, col2, col3 = st.columns([1.5,11,1.5])
    with col2:
        with st.spinner('Saving API Request&Response to Bigquery table'):

            # Load client
            client = bigquery.Client(project='flydubai-338806')

            # Define table name, in format dataset.table_name
            table = 'DSAI_DMV_DATASET.DSAI_DMV_API_RESULT'
            job_config = bigquery.LoadJobConfig(schema=[bigquery.SchemaField("ORDER_ID", bigquery.enums.SqlTypeNames.INTEGER),bigquery.SchemaField("VIN", bigquery.enums.SqlTypeNames.INTEGER),],write_disposition="WRITE_APPEND",)
            # Load data to BQ
            job = client.load_table_from_dataframe(vAR_df, table,job_config=job_config)

            job.result()  # Wait for the job to complete.
            table_id = 'flydubai-338806.DSAI_DMV_DATASET.DSAI_DMV_API_RESULT'
            table = client.get_table(table_id)  # Make an API request.
            print(
                "Loaded {} rows and {} columns to {}".format(
                    table.num_rows, len(table.schema), table_id
                )
            )
            st.write('')
            st.write('')
            st.success('API Request&Response successfully saved into Bigquery table')
            
            
def Upload_Response_To_S3(vAR_result,vAR_request_id):
    
    col1, col2, col3 = st.columns([1.5,11,1.5])
    with col2:
        with st.spinner('Saving API Response to AWS S3'):
            vAR_bucket_name = os.environ['BUCKET_NAME']
            vAR_csv_buffer = StringIO()
            vAR_result.to_csv(vAR_csv_buffer)
            vAR_s3_resource = boto3.resource('s3',aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'])
            vAR_utc_time = datetime.datetime.utcnow()
            vAR_s3_resource.Object(vAR_bucket_name, 'batch/simpligov/new/ELP_Project_Response/'+vAR_utc_time.strftime('%Y%m%d')+'/ELP_Response_'+str(vAR_request_id)+'_'+vAR_utc_time.strftime('%H%M%S')+'.csv').put(Body=vAR_csv_buffer.getvalue())
            st.write('')
            st.success('API Response successfully saved into S3 bucket')
        
def Process_API_Response(vAR_api_response,vAR_request_date,vAR_order_date,vAR_configuration,vAR_order_id,vAR_vin,vAR_model):
    # vAR_api_response as dict
    vAR_data = {}
    vAR_data['REQUEST DATE'] = vAR_request_date
    vAR_data['ORDER DATE'] = vAR_order_date
    vAR_data['CONFIGURATION'] = vAR_configuration
    vAR_data['ORDER ID'] = vAR_order_id
    vAR_data['VIN'] = vAR_vin
    if vAR_api_response['1st Level(Direct Profanity)']['Is accepted']:
        vAR_data['DIRECT PROFANITY'] = 'APPROVED'
        vAR_data['DIRECT PROFANITY MESSAGE'] = 'Not falls under any of the profanity word'
    if not vAR_api_response['1st Level(Direct Profanity)']['Is accepted']:
        vAR_data['DIRECT PROFANITY'] = 'DENIED'
        vAR_data['DIRECT PROFANITY MESSAGE'] = vAR_api_response['1st Level(Direct Profanity)']['Message']
    if vAR_api_response['2nd Level(Denied Pattern)']['Is accepted']:
        vAR_data['RULE-BASED CLASSIFICATION'] = 'APPROVED'
        vAR_data['RULE-BASED CLASSIFICATION MESSAGE'] = 'Not falls under any of the denied patterns'
    if not vAR_api_response['2nd Level(Denied Pattern)']['Is accepted']:
        vAR_data['RULE-BASED CLASSIFICATION'] = 'DENIED'
        vAR_data['RULE-BASED CLASSIFICATION MESSAGE'] = vAR_api_response['2nd Level(Denied Pattern)']['Message']
    vAR_data['MODEL'] = vAR_model
    vAR_data['TOXIC'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Toxic']
    vAR_data['SEVERE TOXIC'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Severe Toxic']
    vAR_data['OBSCENE'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Obscene']
    vAR_data['IDENTITY HATE'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Identity Hate']
    vAR_data['INSULT'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Insult']
    vAR_data['THREAT'] = vAR_api_response['3rd Level(Model Prediction)']['Profanity Classification'][0]['Threat']
    vAR_data['OVERALL PROBABILITY'] = vAR_api_response['3rd Level(Model Prediction)']['Sum of all Categories']
    
    return vAR_data
