import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from dataclasses import dataclass
from imblearn.over_sampling import RandomOverSampler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')

        try:
            df = pd.read_csv('notebook\data\kaggle_RC_2019-05.csv')
            logging.info('Read the dataset as dataframe')

            df_reduced = resample(
                df,
                replace=False,
                n_samples=50000,
                stratify=df['controversiality'],
                random_state=42
            )
            df_reduced = df_reduced.reset_index().drop(columns = ['index'])
            logging.info("Resampling the dataset is successful")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            logging.info("Initialized the oversampler")

            feature = df_reduced['body']
            target = df_reduced['controversiality']

            X, y = oversampler.fit_resample(feature.values.reshape(-1,1), target)

            df_balanced = pd.DataFrame({'body':X.flatten(), 'controversiality':y})
            logging.info("Resampling completed: Balanced dataset successfully generated.")

            logging.info('Train test split initiated')
            train_set, test_set = train_test_split(df_balanced, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('Data Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()