import pandas as pd
import numpy as np

MAX_LEN =1024

# print(model.summary())

def get_preds_inference(txt=None, id=None, tokenizer=None, model=None, verbose=True):

    all_predictions = []
    target_map_rev = {0:'Lead', 1:'Position', 2:'Evidence', 3:'Claim', 4:'Concluding Statement',
             5:'Counterclaim', 6:'Rebuttal', 7:'blank'}

    test_tokens = np.zeros((len([1]),MAX_LEN), dtype='int32')
    test_attention = np.zeros((len([1]),MAX_LEN), dtype='int32')

    tokens = tokenizer.encode_plus(txt, max_length=MAX_LEN, padding='max_length',
                                   truncation=True, return_offsets_mapping=True)
                                   
    test_tokens[0,] = tokens['input_ids']
    test_attention[0,] = tokens['attention_mask']
    print("Running NER predictions...")
    p = model.predict([test_tokens, test_attention], batch_size=16, verbose=2)
    
    test_preds = np.argmax(p,axis=-1)        
    off = tokens['offset_mapping']


    # GET WORD POSITIONS IN CHARS
    w = []
    blank = True
    for i in range(len(txt)):
        if (txt[i]!=' ')&(txt[i]!='\n')&(txt[i]!='\xa0')&(txt[i]!='\x85')&(blank==True):
            w.append(i)
            blank=False
        elif (txt[i]==' ')|(txt[i]=='\n')|(txt[i]=='\xa0')|(txt[i]=='\x85'):
            blank=True
    w.append(1e6)
        
    # MAPPING FROM TOKENS TO WORDS
    word_map = -1 * np.ones(MAX_LEN,dtype='int32')
    w_i = 0
    for i in range(len(off)):
        if off[i][1]==0: continue
        while off[i][0]>=w[w_i+1]: w_i += 1
        word_map[i] = int(w_i)
    
    # CONVERT TOKEN PREDICTIONS INTO WORD LABELS
    ### KEY: ###
    # 0: LEAD_B, 1: LEAD_I
    # 2: POSITION_B, 3: POSITION_I
    # 4: EVIDENCE_B, 5: EVIDENCE_I
    # 6: CLAIM_B, 7: CLAIM_I
    # 8: CONCLUSION_B, 9: CONCLUSION_I
    # 10: COUNTERCLAIM_B, 11: COUNTERCLAIM_I
    # 12: REBUTTAL_B, 13: REBUTTAL_I
    # 14: NOTHING i.e. O
    ### NOTE THESE VALUES ARE DIVIDED BY 2 IN NEXT CODE LINE
    pred = test_preds[0,]/2

    i = 0
    all_predictions = []
    while i<MAX_LEN:
        prediction = []
        start = pred[i]
        if start in [0,1,2,3,4,5,6,7]:
            prediction.append(word_map[i])
            i += 1
            if i>=MAX_LEN: break
            while pred[i]==start+0.5:
                if not word_map[i] in prediction:
                    prediction.append(word_map[i])
                i += 1
                if i>=MAX_LEN: break
        else:
            i += 1
        prediction = [x for x in prediction if x!=-1]
        if len(prediction)>4:
            #sentence_pred=tokenizer.decode(tokens["input_ids"][prediction[0]:prediction[-1]+1] )
            all_predictions.append( (id, target_map_rev[int(start)], 
                            ' '.join([str(x) for x in prediction]) , ' '.join(txt.split()[prediction[0]:prediction[-1]+1])  ))
              
    # MAKE DATAFRAME
    df = pd.DataFrame(all_predictions)
    df.columns = ['id','discourse_type','predictionstring', 'discourse_text']
    return df