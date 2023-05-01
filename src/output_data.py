import pandas as pd
import numpy as np
from model import cleanup_text



def calculate_gap_len(df:pd.DataFrame) -> pd.DataFrame:
    #initialize column
    df['gap_length'] = np.nan
    #set the first one
    df.loc[0, 'gap_length'] = 0

    #loop over rest
    for i in range(1, len(df)):
        #gap if difference is not 1 within an essay
        if ((df.loc[i, "essay_id"] == df.loc[i-1, "essay_id"])\
            and (df.loc[i, "discourse_start"] - df.loc[i-1, "discourse_end"] > 1)):
            df.loc[i, 'gap_length'] = df.loc[i, "discourse_start"] - df.loc[i-1, "discourse_end"] - 2
            #minus 2 as the previous end is always -1 and the previous start always +1
        #gap if the first discourse of an new essay does not start at 0
        elif ((df.loc[i, "essay_id"] != df.loc[i-1, "essay_id"])\
            and (df.loc[i, "discourse_start"] != 0)):
            df.loc[i, 'gap_length'] = df.loc[i, "discourse_start"] -1


    #is there any text after the last discourse of an essay?
    last_ones = df.drop_duplicates(subset="essay_id", keep='last')
    last_ones['gap_end_length'] = np.where((last_ones.discourse_end < last_ones.essay_len),\
                                        (last_ones.essay_len - last_ones.discourse_end),\
                                        np.nan)

    cols_to_merge = ['essay_id', 'gap_end_length']
    test_data_w_gap_len = df.merge(last_ones[cols_to_merge], on = ["essay_id"], how = "left")
    return test_data_w_gap_len

def add_gap_rows(df: pd.DataFrame, essay_id: str) -> pd.DataFrame:
    cols_to_keep = ['discourse_start', 'discourse_end', 'discourse_type', 'gap_length', 'gap_end_length','predicted_label_class']
    df_essay = df[df.essay_id == essay_id][cols_to_keep].reset_index(drop = True)
    #index new row
    insert_row = len(df_essay)
   
    for i in range(1, len(df_essay)):          
        if df_essay.loc[i,"gap_length"] >0:
            if i == 0:
                start = 0 #as there is no i-1 for first row
                end = df_essay.loc[0, 'discourse_start'] -1
                disc_type = "Nothing"
                gap_end = np.nan
                gap = np.nan
                pred_label = ""
                df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, pred_label]
                insert_row += 1
            else:
                start = df_essay.loc[i-1, "discourse_end"] + 1
                end = df_essay.loc[i, 'discourse_start'] -1
                disc_type = "Nothing"
                gap_end = np.nan
                gap = np.nan
                pred_label = ""
                df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, pred_label]
                insert_row += 1

    df_essay = df_essay.sort_values(by = "discourse_start").reset_index(drop=True)

    #add gap at end
    if df_essay.loc[(len(df_essay)-1),'gap_end_length'] > 0:
        start = df_essay.loc[(len(df_essay)-1), "discourse_end"] + 1
        end = start + df_essay.loc[(len(df_essay)-1), 'gap_end_length']
        disc_type = "Nothing"
        gap_end = np.nan
        gap = np.nan
        pred_label = ""
        df_essay.loc[insert_row] = [start, end, disc_type, gap, gap_end, pred_label]

    df_essay.loc[df_essay.discourse_type.isna(),'discourse_type'] = 'Nothing'
    return df_essay


def output_data_format(df:pd.DataFrame) -> dict:
    df_gap_length = calculate_gap_len(df)
    essay_id = df_gap_length['essay_id'].unique()
    df_essay = add_gap_rows(df_gap_length, essay_id[0])
    essay_text = df_gap_length['processed_essay'].unique()[0]
    ents = []
    for i, row in df_essay.iterrows():
        ents.append({
                        'entity': row['discourse_type'] + ":: " +row['predicted_label_class'],
                        'start': int(row['discourse_start']), 
                         'end': int(row['discourse_end']), 
                    })

#     clean_data = cleanup_text(essay_text)
    tokens_essay = clean_data.split()
    essay_text = " ".join(tokens_essay)
    doc = {"text": essay_text, "entities": ents}
    return doc





