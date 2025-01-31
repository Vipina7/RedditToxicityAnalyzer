import pandas as pd
import numpy as np
import re
import dill
import gdown
import sys
import os
import lxml
from bs4 import BeautifulSoup
from src.exception import CustomException
from src.logger import logging

def preprocess_text(sentence):
    try:
        sentence = str(sentence).lower()
        sentence = re.sub('[^a-zA-z0-9]',' ',sentence)
        sentence = re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?','',sentence)
        sentence = BeautifulSoup(sentence,'lxml').get_text()

        return sentence
    
    except Exception as e:
         raise CustomException(e, sys)

def vectorize_text(sentence, vector_model, embedding_dim):
        try:
            words = str(sentence).split()
            valid_words = [vector_model.wv[word] for word in words if word in vector_model.wv]
            if valid_words:
                return np.array(valid_words)
            else:
                return np.zeros((1,embedding_dim))
        
        except Exception as e:
            raise CustomException(e, sys)
    
def download_model_from_drive(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False)
        logging.info(f"Model downloaded successfully from {url} to {output_path}")
    except Exception as e:
        logging.error(f"Failed to download model: {e}")
        raise CustomException(e, sys)
        
def save_object(file_path, obj):
    try:
          dir_path = os.path.dirname(file_path)
          os.makedirs(dir_path, exist_ok=True)

          with open(file_path, 'wb') as file_obj:
               dill.dump(obj, file_obj)

    except Exception as e:
         raise CustomException(e, sys)