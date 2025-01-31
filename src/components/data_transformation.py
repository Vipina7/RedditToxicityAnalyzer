import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import vectorize_text, preprocess_text, save_object
from src.logger import logging

from dataclasses import dataclass
import gensim
from gensim.models import FastText
import gensim.downloader as api
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

@dataclass
class DataTransformationConfig():
    fasttext_file_path = os.path.join('artifacts', 'fasttext.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_vectorization_object(self, corpus):
        try:
            """This builds the FastText model for vectorizing"""

            fasttext_model = FastText(sentences=corpus, vector_size=400, window=5, min_count=1)

            return fasttext_model
        
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

            corpus = train_df['body'].to_list() + test_df['body'].to_list()
            logging.info("Obtained the corpus for vector training")

            logging.info('Obtaining vectorizing model')
            vectorize_obj = self.get_vectorization_object(corpus=corpus)

            X_train = np.vstack(train_df['body'].apply(lambda x: vectorize_text(x, vectorize_obj, embedding_dim=400)))
            X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1])
            y_train = train_df['controversiality']
            
            X_test = np.vstack(test_df['body'].apply(lambda x: vectorize_text(x, vectorize_obj, embedding_dim=400)))
            X_test = X_test.reshape(X_test.shape[0],1,X_test.shape[1])
            y_test = test_df['controversiality']
            logging.info("Vectorizing the train and test data complete")

            logging.info("Saving the word2vec model")
            save_object(file_path= self.data_transformation_config.fasttext_file_path, obj=vectorize_obj)

            return(
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            raise CustomException(e, sys)
        