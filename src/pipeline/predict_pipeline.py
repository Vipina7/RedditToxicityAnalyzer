import sys
import os
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import download_model_from_drive, preprocess_text, vectorize_text, load_object
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
import gdown

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifacts/gru_model.h5'
            vectorizer_path = 'artifacts/fasttest.pkl'

            model = load_model(model_path)
            logging.info("Successfully installed the model")

            fasttext_url = 'https://drive.google.com/uc?id=1jKlaaiD2EG-t66x584NY0S16UI3jzVWK'
            if not os.path.exists(vectorizer_path):
                download_model_from_drive(url=fasttext_url, output_path=vectorizer_path)
            else:
                fasttest_model = load_object(file_path=vectorizer_path)

            features = preprocess_text(features)
            features = features.split()
            features = np.array([vectorize_text(word, fasttest_model, embedding_dim=300) for word in features])
            features = pad_sequences(features, maxlen = 50, padding = 'post', truncating = 'post', dtype = 'float32')

            prediction = model.predict(features)[0][0] * 100

            return prediction
        
        except Exception as e:
            raise CustomException(e, sys)