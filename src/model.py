from keras.models import load_model
import tensorflow_hub as hub
import h5py
import pandas as pd
import glob
import re

path_models = "./models/"
# Universal Sentence Encoder
path_use = path_models + 'use_final.h5'
use_model = load_model(h5py.File(path_use),custom_objects={'KerasLayer':hub.KerasLayer}, compile=False)
use_model.compile()

#fille path
path = '/Users/shringa/FourthBrain/ML project/nlp_project/data/'

# Output Categories
label_categories = {0:"Ineffective", 1:"Adequate", 2:"Effective"}

def read_data(essay_id: str) -> pd.DataFrame:
    if glob.glob(path+"test.csv"):
        df = pd.read_csv(path+"test.csv")
    
    essay_dict = {}
    for file in glob.glob(path+'test/*.txt'): 
        with open(file, "r") as file_open:
            filename = file.split('test/')[1].split('.txt')[0]
            essay_dict[filename] = file_open.read()
        
    essay_data = pd.DataFrame.from_dict([essay_dict]).T.reset_index()
    essay_data.columns = ["essay_id", "essay_text"]
    df = pd.merge(df,essay_data,left_on = 'essay_id', right_on ='essay_id', how ='left')
    requested_df = df[df.essay_id == essay_id]
    requested_df = data_preprocessing(requested_df)
    return requested_df

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


def data_preprocessing(df:pd.DataFrame) -> pd.DataFrame:
    df['processed_discourse'] = df['discourse_text'].apply(cleanup_text)
    df['processed_essay'] = df['essay_text'].apply(cleanup_text)
    df[['discourse_start','discourse_end','essay_len']]= df.apply(lambda x: text_to_char_idx( x['processed_essay'] ,x['processed_discourse']),\
                                                                axis=1 , result_type='expand')
    # df.drop(columns =['essay_text','processed_essay'], inplace = True)
    return df

def predict_data(df:pd.DataFrame) -> pd.DataFrame:

    predicted_output = use_model.predict(df['processed_discourse'], verbose=1)
    df['predicted_prob']=pd.Series(predicted_output.argmax(1))
    df["predicted_label_class"] = df["predicted_prob"].map(label_categories)
    return df