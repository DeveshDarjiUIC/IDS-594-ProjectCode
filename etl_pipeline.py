import pandas as pd
import skops.io as sio
from prefect import flow, task, get_run_logger
from datetime import timedelta
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

@task
def extract_data(filename: str):
    logger = get_run_logger()
    logger.info("Extracting data...")
    bank_df = pd.read_csv(filename, index_col="id", nrows=1000)
    logger.info("Data extraction complete.")
    return bank_df

@task
def transform_data(bank_df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Transforming data...")
    bank_df = bank_df.drop(["CustomerId", "Surname"], axis=1)
    bank_df = bank_df.sample(frac=1)
    
    cat_col = [1, 2]
    num_col = [0, 3, 4, 5, 6, 7, 8, 9]

    cat_impute = SimpleImputer(strategy="most_frequent")
    bank_df.iloc[:, cat_col] = cat_impute.fit_transform(bank_df.iloc[:, cat_col])

    num_impute = SimpleImputer(strategy="median")
    bank_df.iloc[:, num_col] = num_impute.fit_transform(bank_df.iloc[:, num_col])

    cat_encode = OrdinalEncoder()
    bank_df.iloc[:, cat_col] = cat_encode.fit_transform(bank_df.iloc[:, cat_col])

    scaler = MinMaxScaler()
    bank_df.iloc[:, num_col] = scaler.fit_transform(bank_df.iloc[:, num_col])
    
    logger.info("Data transformation complete.")
    return bank_df

@task
def load_data(bank_df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Loading data...")
    return bank_df

@flow
def etl_pipeline(filename: str):
    raw_data = extract_data(filename)
    transformed_data = transform_data(raw_data)
    return load_data(transformed_data)

@task
def data_split(bank_df: pd.DataFrame):
    logger = get_run_logger()
    logger.info("Splitting data...")
    X = bank_df.drop(["Exited"], axis=1)
    y = bank_df.Exited
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)
    logger.info("Data splitting complete.")
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train, X_test, y_train):
    logger = get_run_logger()
    logger.info("Training model...")
    KBest = SelectKBest(chi2, k="all")
    X_train = KBest.fit_transform(X_train, y_train)
    X_test = KBest.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=125)
    model.fit(X_train, y_train)
    logger.info("Model training complete.")
    return model

@task
def get_prediction(X_test, model: LogisticRegression):
    logger = get_run_logger()
    logger.info("Getting predictions...")
    return model.predict(X_test)

@task
def evaluate_model(y_test, prediction: pd.DataFrame):
    logger = get_run_logger()
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average="macro")
    logger.info("Evaluation results - Accuracy: {}%, F1: {}".format(round(accuracy, 2) * 100, round(f1, 2)))

@task
def save_model(model: LogisticRegression):
    logger = get_run_logger()
    logger.info("Saving model...")
    sio.dump(model, "bank_model.skops")
    logger.info("Model saved.")

@flow(log_prints=True)
def ml_workflow(filename: str = "C:\\Users\\arpit\\Python projects\\MLOps\\train.csv"):
    data = etl_pipeline(filename)
    X_train, X_test, y_train, y_test = data_split(data)
    model = train_model(X_train, X_test, y_train)
    predictions = get_prediction(X_test, model)
    evaluate_model(y_test, predictions)
    save_model(model)

if __name__ == "__main__":
    ml_workflow()
