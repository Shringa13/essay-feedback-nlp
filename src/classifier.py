
import pandas as pd
import re
from text_preprocessing import data_preprocessing


# Output Categories
label_categories = {0:"Ineffective", 1:"Adequate", 2:"Effective"}

def get_classifications(df:pd.DataFrame,essay_text:str,model=None) -> pd.DataFrame:
    print(model.summary())
    processed_data = data_preprocessing(df,essay_text)
    print("Running Classifier...")
    predicted_output = model.predict(df['processed_discourse'], verbose=1)
    print(predicted_output)
    df = processed_data.reset_index()
    df['predicted_prob']= list(predicted_output.argmax(1))
    df["predicted_label_class"] = df["predicted_prob"].map(label_categories)
    df.drop(columns = ['discourse_text'], inplace = True)
    return df





