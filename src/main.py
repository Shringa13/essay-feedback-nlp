from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict
from utils import ner_tokenizer, get_model, get_models_files
from examples import example_1, example_2
import gradio as gr
import logging 
import os
from pathlib import Path

# get root logger
logger = logging.getLogger(__name__)

# ------------------------------------------------  Reading files from S3 ----------------------------------------------------
try:
    lonformer_path = 'models/longformer'
    classifiers_path = 'models/classifiers'

    if not (os.path.exists(lonformer_path) and os.path.exists(lonformer_path)):
        logger.info(f" Fetching models")
        ner_model_files_path = get_models_files(lonformer_path)
        classifiers_files_path = get_models_files(classifiers_path)
    else:
        ner_model_files_path = lonformer_path
        classifiers_files_path = classifiers_path
except Exception as error:
    logger.error(f" Error occured while fetching models :::{str(error)}")
    os.exit(1)

# ------------------------------------------------  Loading models ----------------------------------------------------
tokenizer = ner_tokenizer(ner_model_files_path)
ner_model = get_model(ner_model_files_path,"ner")
clf_model = get_model(classifiers_files_path,"classifier")

# ------------------------------------------------  Fast API ----------------------------------------------------
app = FastAPI()
class Request(BaseModel):
    essay_text: str

class Result(Request):
    output_data: dict

@app.post("/predict", status_code=200)
async def predict_api(request: Request):
    try:
        result = predict(request.essay_text, tokenizer, ner_model, clf_model)
    except ValueError as e:
        return HTTPException(status_code=422, detail="Please provide input.")
    if result == None:
        raise HTTPException(status_code=400, detail="Model not found.")
    return result

# ------------------------------------------------  Gradio Demo ----------------------------------------------------
def post_predictions(essay_text:str):
    # if len(essay_id) == 0:
    #     raise gr.Error("Please provide input values.")
    # else:
    results = predict(essay_text, tokenizer, ner_model, clf_model)
    return results

feedback = gr.Text(
                    label='Feedback',
                    show_label=False,
                    max_lines=3,
                    placeholder='Enter your feedback',
                    elem_id='prompt-text-input',
                ).style(container=False)

demo = gr.Interface(post_predictions,
            inputs = [gr.Textbox(placeholder="Enter sentence here...", lines = 20, label ="Enter Essay Text")],
            outputs = [gr.HighlightedText()],
            allow_flagging="manual",
            examples=[[example_1], [example_2]],
            flagging_options=["Adequate", "Effective", "Ineffective"],
            )

app = gr.mount_gradio_app(app, demo, path="/ui")





    
    
