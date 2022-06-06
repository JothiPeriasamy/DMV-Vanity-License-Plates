# Load Huggingface transformers
from transformers import TFBertModel,  BertConfig, BertTokenizerFast, TFAutoModel

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
from google.cloud import bigquery
import tensorflow as tf
import re
import numpy as np
from sklearn.model_selection import train_test_split



def read_bq_data():
    vAR_bqclient = bigquery.Client()
    vAR_query_string = """
        select * from `flydubai-338806.DSAI_DMV_DATASET.DSAI_DMV_TOXIC_COMMENTS`
        """
    vAR_dataframe = (
            vAR_bqclient.query(vAR_query_string)
            .result()
            .to_dataframe(
                create_bqstorage_client=True,
            )
        )
    return vAR_dataframe

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

try:
    # df=pd.read_csv('/home/jupyter/DSAI_DMV_Text_Analyzer/DSAI_Dataset/jigsaw-toxic-comment-train.csv')
    
    df = read_bq_data()
    df = df.head(100)
    print(df)
    df['comment_text'] = df['comment_text'].map(lambda x : clean_text(x))
    train_sentences = df["comment_text"].values
    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train_y = df[list_classes].values

    # Name of the BERT model to use
    model_name = 'bert-base-uncased'

    # Max length of tokens
    max_length = 128

    # Load transformers config and set output_hidden_states to False
    config = BertConfig.from_pretrained(model_name)
    #config.output_hidden_states = False

    # Load BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)
    bert = TFAutoModel.from_pretrained(model_name)


    input_ids = Input(shape=(max_length,), name='input_ids', dtype='int32')
    attention_mask = Input(shape=(max_length,), name='attention_mask', dtype='int32') 
    # input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    # attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    x = bert.bert(inputs)

    #x2 =Dense(512, activation='relu')(x[1])
    x2 = GlobalAveragePooling1D()(x[0])
    #x3 = Dropout(0.5)(x2)
    y =Dense(len(list_classes), activation='sigmoid', name='outputs')(x2)

    model = Model(inputs=inputs, outputs=y)
    #model.layers[2].trainable = False

    # Take a look at the model
    model.summary()


    optimizer = Adam(lr=4e-5, decay=1e-6)
    model.compile(loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])


    # Tokenize the input 
    x = tokenizer(
    text=list(train_sentences),
    add_special_tokens=True,
    max_length=max_length,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

    history = model.fit(
    x={'input_ids': x['input_ids'], 'attention_mask': x['attention_mask']},
    #x={'input_ids': x['input_ids']},
    y={'outputs': train_y},
    validation_split=0.1,
    batch_size=32,
    epochs=1)
    print('Model training completed-------------------------------')
    print('Model Accuracy - ',history.history)
    model.save('/home/jupyter/BERT_MODEL_64B_4e5LR_3E_Test/')
    print('Model saved successfully--------------------------------')

except Exception as e:
    print('Error Block - ',str(e))
