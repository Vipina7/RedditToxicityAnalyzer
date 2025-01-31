import os
import sys
from src.exception import CustomException
from src.logger import logging
import logging

from dataclasses import dataclass
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping

@dataclass
class ModelTrainerConfig:
    def __init__(self):
        self.trained_model_path = os.path.join('artifacts','model.h5')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test
            logging.info("Read the train and test sets for model training and evaluation")

            logging.info("Initializing the model")
            model = Sequential()
            model.add(GRU(64, return_sequences = False, input_shape = (50,300)))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation = 'sigmoid'))

            model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
            logging.info("Model compilation completed")

            early_stopping = EarlyStopping(
                monitor = 'val_loss',
                patience = 5,
                restore_best_weights = True)
            logging.info("Early Stopping defined")

            model.fit(
                X_train, y_train, epochs = 100, batch_size = 64, validation_split = 0.2, callbacks = early_stopping
            )
            logging.info("Model training complete")

            model.save('artifacts/gru_model.h5')
            logging.info("Save model successful")

            prediction = model.predict(X_test)
            prediction_labels = (prediction > 0.5).astype('int')

            return classification_report(y_test, prediction_labels)
        
        except Exception as e:
            raise CustomException(e, sys)