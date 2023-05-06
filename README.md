<img align=right src="/data/fourthbrain.png" width="128"/>

# Essay Writing Feedback

Georgia State University and the Learning Agency Lab are hosted a Kaggle Competition aimed at improving the persuasive writing skills of students. The current education system has not given enough importance to this aspect, which may hinder the students' development of critical thinking. The competition's goal is to create automated essay writing feedback tool.

## Objective 

* Build a classifier model to classify argumentative elements in student writing as “effective”, “adequate” or “ineffective”.
* Create a front-end interface for teachers to help evaluate students writings.

## Summary

The classifier was built using pre-trained [sentence transformer](https://huggingface.co/sentence-transformers) embedding model which was further fine-tuned and ran XGB boost classifier to classify the arguments. I have also incorporated a [baseline NER model](https://huggingface.co/allenai/longformer-base-4096) to identify arguments to make this an end-to-end solution. The model is performing well to classify "Adequate" and "Ineffective" labels, but the recall rate for "Effective" label is low.

**Data:** 

Information of the data used can be found here: https://www.kaggle.com/competitions/feedback-prize-effectiveness/data

**Approach:**

Examined both Word2Vec and Transformers-based embedding methods that can comprehend the context of an essay.
Fine-tuned various calssifiers (including NN models) to identify essay arguments.

**Evaluation Metrics**
Two metrics were used to evaluate the model:

1) F1-Score (per class & weighted average) - F1-Score was used to evaluate the prediction as the data is imbalance, thus accuracy will not be very representative.

2) Cross Entropy Loss - Since this is a multiclass classification problem and neural networks were used for modelling, a loss function that is differentiable is required, thus cross entropy loss were used.

# Service Setup 

1. Clone the Project

```
 git clone https://github.com/Shringa13/essay-feedback-nlp.git
 cd essay-feedback-nlp
```
2. Install required libraries
```
pip install -r requirement.txt
```
3. Run the service

```
cd src
python main.py
```
4. You can also start playaround with the service through below link

```
http://0.0.0.0:8000/ui
```
<img align = center src="/data/gradio output.png" width=100% height=100%/>
