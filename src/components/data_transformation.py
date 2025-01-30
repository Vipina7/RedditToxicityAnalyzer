import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import vectorize_text, preprocess_text
from src.logger import logging

from dataclasses import dataclass
import gensim
from gensim.models import word2vec, KeyedVectors
import gensim.downloader as api

@dataclass
class DataTransformationConfig():
    google_word2vec_file_path = os.path.join('artifacts', 'wv.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_vectorization_object(self):
        try:
            """This function imports google Word2Vec Model from gensim"""

            wv = api.load('word2vec-google-news-300')
            logging.info("Dowloaded google word2vec model")

            return wv
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data')

            train_df['body'] = train_df['body'].apply(lambda x: preprocess_text(x))
            test_df['body'] = test_df['body'].apply(lambda x: preprocess_text(x))
            logging.info('Preprocessing of data is complete')

            logging.info('Obtaining vectorizing model')
            vectorize_obj = self.get_vectorization_object()

            X_train = np.vstack(train_df['body'].apply(lambda x: vectorize_text(x, vectorize_obj, embedding_dim=300)))
            y_train = np.array(train_df['controversiality'])
            
            X_test = np.vstack(test_df['body'].apply(lambda x: vectorize_text(x, vectorize_obj, embedding_dim=300)))
            y_test = np.array(test_df['controversiality'])

            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            logging.info("Vectorizing the train and test data complete")

            logging.info("Saving the word2vec model")
            vectorize_obj.save('artifacts/wv')

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
    
    

        