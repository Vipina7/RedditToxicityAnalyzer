import streamlit as st
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline
from src.exception import CustomException
import gensim

st.title('Reddit Toxicity Evaluation App')

st.write(
    "This app predicts whether a given text is toxic or not based on a trained GRU model. "
    "Simply input some text below, and the model will predict if the text is toxic."
)

# Input for the user
input_text = st.text_area("Enter your text here...")

if st.button("Evaluate Toxicity"):
    if input_text:
        try:
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(input_text)

            if prediction > 40:
                st.write(f"The content is likely **toxic** with a toxicity level of {prediction}%. Please review carefully.")
            else:
                st.write(f"The content appears to have a toxicity level of {prediction}%.")
        
        except Exception as e:
            st.error(f"Error occured: {str(e)}")

    else:
        st.error("Please enter some text to predict!")