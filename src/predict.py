import pandas as pd
from model import read_data,predict_data
from output_data import output_data_format

def predict(essay_id:str) -> dict:
    try:
        processed_output = read_data(essay_id)
        predicted_data = predict_data(processed_output)
        final_output = output_data_format(predicted_data)
        return final_output
    except Exception as e:
        print(e)
 
predict('A8445CABFECE')

