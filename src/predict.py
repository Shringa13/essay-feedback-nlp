import pandas as pd
from ner_model import get_preds_inference
from classifier import get_classifications
from output_data import output_data_format
import logging 

logger = logging.getLogger(__name__)
def predict(essay_text:str,tokenizer, ner_model, clf_model) -> dict:
    try:
        ner_output = get_preds_inference(txt=essay_text, id='001', tokenizer=tokenizer, model=ner_model)
        predicted_data = get_classifications(df=ner_output, essay_text=essay_text, model=clf_model)
        final_output = output_data_format(predicted_data, essay_text)
        logger.info(final_output)
        return final_output
    except Exception as e:
        logger.info(e)


