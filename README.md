Bank Customer Churn Prediction ML Pipeline
A complete MLOps pipeline for predicting bank customer churn using Logistic Regression, with orchestration via Prefect and a Flask web interface for predictions.
Overview
This project implements an end-to-end machine learning workflow that:

Processes bank customer data through an ETL pipeline
Trains a Logistic Regression model to predict customer churn
Provides a web interface for making predictions on new data

Features

Orchestrated ML Pipeline: Uses Prefect for workflow orchestration and logging
Data Processing: Automated feature engineering, imputation, encoding, and scaling
Model Training: Logistic Regression with feature selection using chi-squared test
Model Persistence: Saves trained models using skops for secure serialization
Web API: Flask application for real-time predictions via CSV input

Project Structure
.
├── ml_workflow.py        # Main ML pipeline with Prefect orchestration
├── app.py               # Flask web application for predictions
├── bank_model.skops     # Saved trained model (generated after training)
└── train.csv            # Training data (not included)
Requirements
pandas
scikit-learn
prefect
skops
flask
Install dependencies:
bashpip install pandas scikit-learn prefect skops flask
Usage
Training the Model

Prepare your dataset as train.csv with the following expected columns:

id (index column)
CustomerId, Surname (will be dropped)
Geography, Gender (categorical features)
Numerical features: CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
Exited (target variable: 1 for churned, 0 for retained)


Run the training pipeline:

bashpython ml_workflow.py
The pipeline will:

Extract and load the first 1000 rows from the CSV
Handle missing values (categorical: most frequent, numerical: median)
Encode categorical variables and scale numerical features
Split data (70% train, 30% test)
Train a Logistic Regression model with feature selection
Evaluate performance (accuracy and F1 score)
Save the model to bank_model.skops

Running the Prediction API

Ensure bank_model.skops exists in the same directory
Start the Flask server:

bashpython app.py

Navigate to http://127.0.0.1:5000 in your browser
Paste CSV data in the text area with properly formatted features (matching training data columns)
Click "Predict" to get churn predictions

API Endpoint
POST /predict

Input: CSV data as form data (key: csv_data)
Output: JSON with predictions array
Example response:

json{
  "predictions": [0, 1, 0, 1]
}
Model Details

Algorithm: Logistic Regression
Feature Selection: SelectKBest with chi-squared test
Preprocessing:

Missing value imputation (median for numerical, most frequent for categorical)
Ordinal encoding for categorical variables
MinMax scaling for numerical features


Evaluation Metrics: Accuracy and Macro F1 Score

Pipeline Workflow
The Prefect-orchestrated pipeline consists of:

ETL Flow:

extract_data: Load CSV data
transform_data: Clean, encode, and scale features
load_data: Return processed data


ML Flow:

data_split: Split into train/test sets
train_model: Feature selection and model training
get_prediction: Generate predictions
evaluate_model: Calculate performance metrics
save_model: Persist trained model



Notes

The current implementation processes 1000 rows (nrows=1000) for demonstration purposes
Update the file path in ml_workflow() to match your data location
Ensure input data for predictions matches the feature structure used during training
The model uses random_state=125 for reproducibility

Future Enhancements

Add model versioning and experiment tracking
Implement automated retraining pipelines
Add data validation and drift detection
Expand API with batch prediction endpoints
Add authentication and rate limiting to the web service
