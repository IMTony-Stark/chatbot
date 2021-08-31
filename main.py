import uvicorn
from fastapi import FastAPI, Body, Request, File, UploadFile, Form, Depends, BackgroundTasks
from fastapi.templating import Jinja2Templates

import json

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from keras.utils import np_utils

import CustomToken as tkn

import tokenization
import tensorflow_text as tf_text
from fastapi.responses import HTMLResponse


app = FastAPI(
    title="HealthSafety Model API",
    description="A simple API that use NLP model to predict the potential accident levels for industry workers",
    version="0.1",
)

loadmodelbert=tf.keras.models.load_model('modelBert4GoodOutput')
# loadmodelbert=tf.keras.models.load_model('sachinBERTmodel')


module_url = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)


bert_preprocess_model = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')


vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tkn.FullTokenizer(vocab_file, do_lower_case)


def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


import Preprocess as pp


def text_cleaning(text):
#    curatedWords = pp.preprocessor(text) 
#    return curatedWords
    return text


classNames = ['I','II','III','IV','V','VI','Accident Level Not Identified']
def finalOutput(result):
    if result > 5:
        return classNames[6]
    else:
        return classNames[result]   


def checkforGreetings(text):
    data_file = open('greetings.json').read()
    intents = json.loads(data_file)
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern == text:
                text = intent['responses']
    return text
    
    
templates = Jinja2Templates(directory="htmldirectory")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

#@app.get("/getChatBotResponse")
#def get_bot_response(msg: str):
#    return str(english_bot.get_response(msg))


# @app.get("/home")
#def write_home(request: Request):
 #   return templates.TemplateResponse("home.html", {"request": request})

@app.get("/getChatBotResponse")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content and predict the accident level of the content.
    :param review:
    :return: prediction
    """
    
    curatedReview = checkforGreetings(review)
    
    if curatedReview == review : 
        
        # clean the review
        cleaned_reviewPreprocessed = text_cleaning(curatedReview)

        cleaned_review = bert_encode(cleaned_reviewPreprocessed, tokenizer, max_len=160)
        # perform prediction
        prediction = loadmodelbert.predict([cleaned_review])
        result = finalOutput(prediction.argmax())

        return result
        
    else:
        return curatedReview