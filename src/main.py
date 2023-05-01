from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import predict
from model import get_model
import gradio as gr
import logging 

# get root logger
logger = logging.getLogger(__name__)

app = FastAPI()
model = get_model()
model.compile()

class Request(BaseModel):
    essay_id: str
    essay_text: str

class Result(Request):
    output_data: dict

@app.post("/predict", status_code=200)
async def predict_api(request: Request):
    try:
        result = predict(request.essay_id, model)
    except ValueError as e:
        return HTTPException(status_code=422, detail="Please provide input.")
    
    if result == None:
        raise HTTPException(status_code=400, detail="Model not found.")
    return result

# ------------------------------------------------  Gradio Demo ----------------------------------------------------
def post_predictions(essay_id:str, essay_text:str):
    # if len(essay_id) == 0:
    #     raise gr.Error("Please provide input values.")
    # else:
    results = predict(essay_id, model)
    return results

demo = gr.Interface(post_predictions,
            inputs = [gr.Textbox(placeholder="Enter sentence here...", lines = 1, label ="Enter Essay Id"),gr.Textbox(placeholder="Enter sentence here...", lines = 20, label ="Enter Essay Text")],
            outputs = gr.HighlightedText())

app = gr.mount_gradio_app(app, demo, path="/")



    
    
