import pandas as pd
import tempfile
from keras import models
import tensorflow_hub as hub
import boto3
import re
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

client = boto3.client('s3')
bucket_name = 'essayfeedback'
prefix = 'data/test/'

# Output Categories
label_categories = {0:"Ineffective", 1:"Adequate", 2:"Effective"}

def get_model():
    # Create the S3 object
    response_data = client.get_object(
        Bucket = bucket_name,
        Key = 'models/use_final.h5'
    )

    model_name='use_final.h5'
    response_data=response_data['Body']
    response_data=response_data.read()
    #save byte file to temp storage
    with tempfile.TemporaryDirectory() as tempdir:
        with open(f"{tempdir}/{model_name}", 'wb') as my_data_file:
            my_data_file.write(response_data)
            #load byte file from temp storage into variable
            gotten_model=models.load_model(f"{tempdir}/{model_name}",custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
    return gotten_model

def read_data(essay_id: str) -> pd.DataFrame:

    file_keys =[]
    response = client.get_object(Bucket=bucket_name, Key="data/test.csv")
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    response_text = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    for obj in response_text['Contents']:
        file_keys.append(obj['Key'])

    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        df = pd.read_csv(response.get("Body"))
    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")

    essay_dict = {}
    for file_key in file_keys:
        obj = client.get_object(Bucket=bucket_name, Key=file_key)
        file_body = obj['Body'].read().decode('utf-8')
        filename = file_key.split('test/')[1].split('.txt')[0]
        essay_dict[filename] = file_body

    essay_data = pd.DataFrame.from_dict([essay_dict]).T.reset_index()
    essay_data.columns = ["essay_id", "essay_text"]
    df = pd.merge(df,essay_data,left_on = 'essay_id', right_on ='essay_id', how ='left')
    requested_df = df[df.essay_id == essay_id]
    print("S3 data read:",requested_df)
    processed_data = data_preprocessing(requested_df)
    return processed_data
 

def cleanup_text(text:str) -> str:
    words = re.sub(pattern = '[^a-zA-Z]',repl = ' ', string = text)
    words = words.lower()
    return words

def text_to_char_idx(full_text:str, substring:str):
    try:
        words = substring.split()
        essay_token = full_text.split()
        substring_text = " ".join(words)
        essay_text = " ".join(essay_token)
        essay_len = len(essay_text)
        start_char = essay_text.find(substring_text)
        end_char = start_char + len(substring_text)
    except Exception as e:
        print(e, full_text, substring)
    return start_char, end_char, essay_len


def data_preprocessing(test_data:pd.DataFrame) -> pd.DataFrame:
    test_data['processed_discourse'] = test_data['discourse_text'].apply(cleanup_text)
    test_data['processed_essay'] = test_data['essay_text'].apply(cleanup_text)
    test_data[['discourse_start','discourse_end','essay_len']]= test_data.apply(lambda x: text_to_char_idx( x['processed_essay'] ,x['processed_discourse']),\
                                                                axis=1 , result_type='expand')
    # df.drop(columns =['essay_text','processed_essay'], inplace = True)
    return test_data

def predict_data(df:pd.DataFrame) -> pd.DataFrame:
    model = get_model()
    model.compile() 
    predicted_output = model.predict(df['processed_discourse'], verbose=1)
    df = df.reset_index()
    df['predicted_prob']= list(predicted_output.argmax(1))
    df["predicted_label_class"] = df["predicted_prob"].map(label_categories)
    print("predicted_data:",df.shape)
    return df
