import re
import pandas as pd
from google.cloud import bigquery



def Pattern_Denial(input_text):
    
    # Below configurations needs to be update in bigquery table
    
    # 1.3 numbers, 2 letters 000AA-000E0 RESERVED Between AA-E0
    # 2.4 numbers 1111 PREV. ASSIGNED Check R60 4E/4S
    # 3.5 numbers 11111 PREV. ASSIGNED Check R60 4E/4S
    vAR_regex_pattern_data = Read_Bigquery_Data()
    # vAR_regex_pattern_data = pd.read_csv('DSAI_Model_Implementation_Sourcecode/DSAI_Regex_Pattern.tsv',sep='\t')
    
    vAR_pattern = None
    
    for vAR_index, vAR_row in vAR_regex_pattern_data.iterrows():
        vAR_result = re.findall(vAR_row['REGEX_CONFIGURATION'],input_text)
        if len(vAR_result)==0:
            pass
        else:
            vAR_pattern = vAR_row['DENIAL_PATTERN']
            # st.write('')
            # st.error("Denied - Similar to " +vAR_pattern+ " Pattern")
            return False,vAR_pattern
        
    return True,vAR_pattern
    
    
    
def Read_Bigquery_Data():

    bqclient = bigquery.Client()

    # Download query results.
    query_string = """
    SELECT * FROM `flydubai-338806.DSAI_DMV_DATASET.DSAI_DMV_DENIED_PATTERN`
    """

    dataframe = (
        bqclient.query(query_string)
        .result()
        .to_dataframe(
            # Optionally, explicitly request to use the BigQuery Storage API. As of
            # google-cloud-bigquery version 1.26.0 and above, the BigQuery Storage
            # API is used by default.
            create_bqstorage_client=True,
        )
    )
    return dataframe