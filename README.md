# ToxiScan: Reddit Toxicity Evaluator

ToxiScan is a Streamlit-based web application designed to predict the toxicity level of Reddit comments using a GRU (Gated Recurrent Unit) model. This project focuses on identifying controversial or toxic content in online discussions by analyzing text data. The system leverages NLP techniques and deep learning to provide real-time predictions and insights into comment toxicity.

---

## Project Structure
The project is organized as follows:

Reddit-Toxicity-Evaluation/
|
|-- src/
| |-- components/
| | |-- data_ingestion.py # Loads/processes raw Reddit comment data
| | |-- data_transformation.py# Handles text preprocessing & FastText embeddings
| | |-- model_trainer.py # Trains/validates GRU model
| | -- predict_pipeline.py # Manages inference pipeline 
| |
| |-- utils.py # Text cleaning/vectorization utilities
| |-- exception.py # Custom error handling
| |-- logger.py # Logging configuration
|
|-- artifacts/ # Saved models/preprocessed data
|-- notebooks/ # Jupyter notebooks for EDA/model prototyping
|-- app.py # Main Streamlit application
|-- requirements.txt # Python dependencies
|-- README.md # Project documentation
|
-- data/ |-- kaggle_RC_2019-05.csv


---

## Dataset
The model was trained on a sample(100k) from Kaggle dataset containing 1M+ Reddit comments with metadata.  
**Key Attributes**:
- `body`: Comment text content
- `controversiality`: Binary label (1=toxic, 0=non-toxic)

---

## Technology Stack
**Backend**:  
- Python 3.10+
- TensorFlow/Keras (GRU implementation)  
- Gensim (Word2Vec embeddings)  

**Frontend**:  
- Streamlit (Web interface)  

**MLOps**:  
- Scikit-learn (Data resampling)  
- Pandas/NumPy (Data processing)  
- Imbalanced-learn (Oversampling) 

**Infrastructure**:  
- Google Colab (Model prototyping)  
- Git/GitHub (Version control)  

---

## Key Features
1. **Automated Data Pipeline**  
   - Balanced dataset handling (RandomOverSampler)
   - Text preprocessing (special char removal, lowercasing, URL stripping)
   - FastText sentence vectorization

2. **GRU Neural Network**  
   - Two layers of GRU with 64-units and 32-Units along with dropout regularization
   - Sigmoid output for binary classification
   - Early stopping to prevent overfitting

3. **Real-Time Predictions**  
   - Web interface for text input
   - Toxicity probability visualization
   - Evaluated the percentage of Toxicity/Controversiality in the comment

4. **Modular Architecture**  
   - Separation of concerns (ingestion/transformation/training)
   - Reusable components for easy maintenance

---
## Results

- Model achieved 74% accuracy on the test set.
- Addressed potential overfitting concerns.

---
## Future Improvements

- Implement hyperparameter tuning to improve accuracy.
- Explore pretrained FastText embeddings for better word representations.
- Introduce attention mechanisms to enhance model interpretability.

---
## How to Use

**Prerequisites**
- Ensure you have the following installed:
- Python 3.10+
- pip (Python package installer)

**Running the Application**
To start the Flask app, run:
streamlit run app.py
This will launch the application on http://127.0.0.1:5000. You can use the web interface to input data and see predictions.