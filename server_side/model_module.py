#!/usr/bin/env python3

## Import necessary libraries 
import os, re, time, joblib, pickle
from argparse import ArgumentParser

from typing import Dict
from datetime import date

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from server_side.logger_module import update_predict_log, update_train_log

import warnings
warnings.filterwarnings('ignore')

PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '.', "models")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), '.', "training_data")
LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "server_logs")
MODEL_PARAM_GRID = {'model__n_estimators' : [90, 100, 110, 120, 130] ,
                    'model__max_depth'    : [None, 5, 10, 15]        ,
                    'model__criterion'    : ['squared_error']        }

                
def model_train(preprocessed_data_dir: str, country: str, model, 
                model_param_grid: dict, model_scaler, mode: str) -> Dict:
    """
    Trains a machine learning model on a specified dataset, evaluates its performance, and saves the model.
    This function performs the following steps:
    1. Prepares data for training.
    2. Tunes model hyperparameters using grid search.
    3. Evaluates the model's performance on a test set using RMSE and MAPE metrics.
    4. Retrains the model using the entire dataset.
    5. Saves the trained model to the filesystem.
    Parameters:
        - clean_data_dir   : The directory containing cleaned data for model training.
        - country          : The country for which the model is being trained.
        - version          : The version number of the model.
        - model            : The machine learning model to be trained (e.g., RandomForestRegressor).
        - model_param_grid : The hyperparameter grid for tuning the model.
        - model_scaler     : The scaler to preprocess input features.
        - mode             : A string indicating the training mode (e.g., 'development', 'production').
    Returns   : A dictionary containing details of the training process
    """
    
    # Start timer for runtime
    model_train_time_start = time.time()
    
    # Construct the path to the preprocessed data
    preprocessed_data_file_name = f'{country}_preprocessed_data.csv'
    preprocessed_data_file_path = os.path.join(preprocessed_data_dir, preprocessed_data_file_name)

    try:
        preprocessed_df = pd.read_csv(preprocessed_data_file_path)
    except FileNotFoundError:
        raise Exception("Country not found")

    # Construct the model version name
    version = pd.to_datetime(preprocessed_df['date']).iloc[-1]
    year = version.year
    month = version.month
    version = f"v{month:02d}_{year}"
    
    print(f"\n... Performing training on train set\n    Version: {version}")

    # Exclude the last 30 days to ensure target reliability
    preprocessed_df = preprocessed_df.iloc[:-30]

    # Prepare the data
    features = ['previous_30_days_views','previous_year_revenue','previous_7_days_revenue',
                'previous_14_days_revenue','previous_28_days_revenue','previous_70_days_revenue','revenue']
    
    X     = preprocessed_df[features]
    y     = preprocessed_df['target_next_30_days_revenue']
    dates = preprocessed_df['date']

    # Define the date range for logging
    max_date = dates.iloc[-1]
    min_date = dates.iloc[0]
    date_range = f"[{min_date}, {max_date}]"
    
    # Perform a train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1, shuffle=True)

    # Create a pipeline with scaling and a random forest model
    pipe = Pipeline([('model_scaler', model_scaler), ('model', model)])
    
    # Tune the hyperparameter
    print("\n... Tuning the model hyperparameters ")
    grid = GridSearchCV(pipe, param_grid=model_param_grid, cv=5, n_jobs=2)

    # Fit the model on train set
    try:
        grid.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training on train set: {e}")
        return None

    # Make predictions on test set 
    y_pred = grid.predict(X_test)
    
    ## Evaluate the model on test set
    # Calculate RMSE
    eval_rmse = round(mean_squared_error(y_test, y_pred)**0.5)
    # Calculate MAPE
    eval_mape = round( mean_absolute_percentage_error(y_test, y_pred) * 100 , 1)
    # Define the date range for logging
    eval_metrics   = {"RMSE":eval_rmse,"MAPE":eval_mape}
                  
    # Retrain using all data
    print("\n... Retraining model on all data")
    try:
        grid.fit(X, y)
    except Exception as e:
        print(f"Error during model retraining on all data: {e}")
        return None

    ### Extract the best model
    best_fitted_model = grid.best_estimator_
    
    ## Extract the best model parameters
    best_model_params = best_fitted_model.get_params()
    # Access the best parameters for max_depth and n_estimators
    best_max_depth    = best_model_params["model__max_depth"]
    best_n_estimators = best_model_params["model__n_estimators"]
    best_criterion    = best_model_params["model__criterion"]
    best_grid_parameters = {"max_depth": best_max_depth,
                            "n_estimators": best_n_estimators,
                            "criterion": best_criterion}    
    
    ## make the model name more system compatible and file-friendly when saving the model
    best_fitted_model_name = best_fitted_model.named_steps['model'].__class__.__name__
    scaler_name = best_fitted_model.named_steps['model_scaler'].__class__.__name__

    # Create model directory 
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Define the file path to save the model
    saved_model = os.path.join(MODEL_DIR,
                               f"{mode}-{country}-{best_fitted_model_name}-{scaler_name}-{eval_metrics}-{version}.joblib")

    # Save the fitted model
    joblib.dump(best_fitted_model, saved_model)
    
    # Create directory to store training data
    if not os.path.isdir(TRAINING_DATA_DIR):
        os.makedirs(TRAINING_DATA_DIR)
        
    # Define the file path to save the training data
    saved_training_data = os.path.join(TRAINING_DATA_DIR, 
                                       f"{mode}-{country}-{best_fitted_model_name}-{scaler_name}-{eval_metrics}-{version}.pickle")

    # Save the training data
    with open(saved_training_data, 'wb') as tmp:
            pickle.dump({'dates': dates, 'X': X, 'y': y }, tmp)

    print(f"\nModel '{mode}-{country}-{best_fitted_model_name}-{scaler_name}-{eval_metrics}-v{version}' trained")
    print(f"\nIts corresponding training data : f'{mode}-{country}-{best_fitted_model_name}-{scaler_name}-{eval_metrics}-v{version}'.pickle")    
    
    # Calculate runtime duration
    m, s = divmod(time.time() - model_train_time_start, 60)
    h, m = divmod(m, 60)
    model_train_runtime = "%02d:%02d:%02d" % (h, m, s)
    print(f'\nModel training runtime : {model_train_runtime}\n')

    # update training log
    update_train_log(mode, country, date_range, 
                     best_fitted_model_name, scaler_name, best_grid_parameters, 
                     eval_metrics, version, model_train_runtime)

    training_details = {
        "mode": mode,
        "country": country,
        "version": version,
        "scaler": scaler_name,
        "model": best_fitted_model_name,
        "model parameters grid": model_param_grid,
        "best model parameters" : best_grid_parameters,
        "evaluation metrics": eval_metrics,
        "training runtime": model_train_runtime,
        "status": "training completed successfully"
    }
    
    # Return details of the training process
    return training_details


def model_load(country: str, mode: str) -> dict:
    """
    Load trained models from disk.
    Parameters:
        - country        : The country for which models are being loaded.
        - mode           : The mode of the model (e.g., 'train', 'test').    
    Returns   : 
            - A dictionary of loaded models keyed by model name.
    """
    
    # Initialize return value
    models_dict = {}

    # List all files in MODEL_DIR and print for debugging
    all_files = os.listdir(MODEL_DIR)
    
    # Retrieve all model filenames matching the specified mode and country
    models = [filename for filename in all_files 
              if (f"{mode}-{country}" in filename) and filename.endswith('.joblib')]

    # Check if any models were found
    if not models:
        print(f"\nNo model found for {country} in {mode} mode. Have you trained it yet?\n")
        return models_dict
        
    # Load models into a dictionary, keyed by model name without file extension
    models_dict = {
                   model.replace('.joblib', '') : joblib.load(os.path.join(MODEL_DIR, model)) 
                   for model in models
                  }
    
    return models_dict


def nearest_date(dates: pd.Series, target_date: date) -> pd.Timestamp:

    """Find the nearest date to the given pivot date.
    Parameters:
        - dates : A list of date strings in 'YYYY-MM-DD' format.
        - target_date : The date to which the nearest date is sought.
    Returns   : 
        - The nearest date from the items list to the pivot date.
    """
    dates = pd.to_datetime(dates)
    target_date = pd.to_datetime(target_date)
    
    # Find the absolute difference in days and return the nearest date
    return dates.iloc[(dates - target_date).abs().argmin()]



def model_predict(preprocessed_data_dir: str, country: str, year: int, month: int, day: int
                  , mode: str, models_dict: dict = None) -> int:
    """
    Predict revenue for a specified date using the best available model.
    This function loads the relevant model, checks the target date, and makes a revenue prediction.
    It logs the prediction results along with runtime information.
    Parameters:
        - preprocessed_data_dir : Directory path containing cleaned data.
        - country        : The country for which the prediction is being made.
        - year           : The year of the target date.
        - month          : The month of the target date.
        - day            : The day of the target date.
        - mode           : The mode of the model (e.g., 'development', 'production').
        - models_dict    : Optional, a dictionary of loaded models (if already loaded).
    Returns    :
        - The prediction
    """
    
    ## Start timer for runtime
    model_predict_time_start = time.time()
    
    # Validate Date Components
    for date_component in [year, month, day]:
        if re.search(r"\D", str(date_component)):  
            raise Exception("ERROR (model_predict) - invalid year, month, or day")

    ## Load the preprocessed data 
    preprocessed_data_file_path = os.path.join(preprocessed_data_dir, f'{country}_preprocessed_data.csv')
    
    features = ['previous_30_days_views', 'previous_year_revenue', 'previous_7_days_revenue', 
                'previous_14_days_revenue', 'previous_28_days_revenue', 'previous_70_days_revenue', 'revenue']
    
    try:
        preprocessed_df = pd.read_csv(preprocessed_data_file_path,
                                      usecols = ['date','target_next_30_days_revenue'] + features )
    except FileNotFoundError:
        raise Exception("Country not found")

    dates = preprocessed_df['date']

    # Define the date range for logging
    max_date = dates.iloc[-1]
    min_date = dates.iloc[0]
    date_range = f"[{min_date}, {max_date}]"
    
    # Check the target date
    target_date = f'{year}-{str(month).zfill(2)}-{str(day).zfill(2)}'
    print(f"\n... Checking if {target_date} is in the range.")

    # Validate Target Date and Find Nearest Date if Out of Range
    if target_date not in dates.astype(str).values:  
        
        print(f"{target_date} not in range {date_range}")
              
        nearest_date_ = nearest_date(preprocessed_df['date'], date.fromisoformat(target_date))
        print(f"The nearest target date is {nearest_date_}. Use it to predict the revenue over the next 30 days.")
        
        # Update predict log
        update_predict_log(mode, country, date_range, target_date, None, None, None, None, None)
        
        return (preprocessed_df['date'].iloc[0], preprocessed_df['date'].iloc[-1], nearest_date_)
        
    else:
        print("Target date is in the range.")
        
    ## Load model if needed
    print(f"\n... Loading models")    
    if models_dict is None:
        models_dict = model_load(country, mode=mode)
    
    # Finding the best model based on latest version and min RMSE
    best_model_key = min( models_dict.keys(),
                          key=lambda k: (
                            -int(k.split("-v")[1].split("_")[1]),  
                            -int(k.split("-v")[1].split("_")[0]),  
                            int(k.split("{'RMSE': ")[1].split(",")[0])
                                        )
                          )
    best_model = models_dict[best_model_key]
    print(f"\nBest Model trained for {country}: {best_model_key}")  
    print(best_model)  

    best_rmse = int(best_model_key.split("{'RMSE': ")[1].split(",")[0])
    model_version = best_model_key.split("-")[5].split(".")[0]

    # Get the index of the target_date
    target_date_indx = pd.DatetimeIndex(dates).get_loc(target_date)

    # Query the corresponding row
    query = preprocessed_df[features].iloc[[target_date_indx]]  

    ## Make prediction
    y_pred  = best_model.predict(query)
    y_pred_ = round(y_pred[0], 2)

    ## Add a probability to the prediction 
    y_proba = None
    if 'predict_proba' in dir(best_model) and getattr(best_model, 'probability', False):
        y_proba = best_model.predict_proba(query)
    
    # Calculate runtime duration
    m, s = divmod(time.time() - model_predict_time_start, 60)
    h, m = divmod(m, 60)
    model_predict_runtime = "%02d:%02d:%02d" % (h, m, s)
    print(f'\nPredicting runtime: {model_predict_runtime}\n')

    # Update predict log
    update_predict_log(mode, country, date_range, target_date, y_pred_,
                       y_proba, best_rmse, model_version, model_predict_runtime)
    
    print(f'\nStarting from {target_date}, the expected revenue over the next 30 days is {y_pred[0]:,.0f} £\n')

    return y_pred[0]
    
  
if __name__ == "__main__":
    
    ap = ArgumentParser()
    
    ap.add_argument('-model', '--model', choices=['rf', 'et'], default='et',
                    help='(rf) RandomForestRegressor or (et) ExtraTreesRegressor (default)')
    
    ap.add_argument('-s', '--scaler', choices=['ss', 'rs'], default='rs',
                    help='(ss) StandardScaler or (rs) RobustScaler (default)')
    
    ap.add_argument('-t', '--task', choices=['train', 'load', 'predict'], default='train',
                    help='choosing whether we are training a model, or predicting from an already-trained model,\
                        or just loading existing models')

    ap.add_argument('-m', '--mode', choices=['dev', 'prod'], default='prod',
                    help='Train either a development or production model. Omitting this implies loading an already-trained model')
    
    ap.add_argument('-c', '--country', default='World',
                    help="The country to predict revenue for (default: 'World')")

    args    = ap.parse_args()
    
    ################################################################
    
    TASK    = args.task
    MODE    = args.mode
    COUNTRY = args.country
    
    MODEL   = RandomForestRegressor(random_state=42) if args.model  == 'rf' else ExtraTreesRegressor(random_state=42)
    SCALER  = StandardScaler()                       if args.scaler == 'ss' else RobustScaler()
   
    # variables for predictions
    YEAR    = 2019
    MONTH   = 9
    DAY     = 1
    
    ################################################################
    
    # train models
    if TASK == 'train':
        
        print(f"\n{TASK}ing a {SCALER} + {MODEL} model for {COUNTRY} in {MODE} mode")
        print(f"\nThe parameters grid : {MODEL_PARAM_GRID}")
        
        training_details = model_train(PREPROCESSED_DATA_DIR, COUNTRY, MODEL, MODEL_PARAM_GRID, SCALER, MODE)

        print(f'Training Summary :\n{training_details}\n')
    # load models
    elif TASK == 'load':
        
            DEV_MODELS,  _  = model_load(PREPROCESSED_DATA_DIR, COUNTRY, with_data=False, mode='dev')
            PROD_MODELS, _  = model_load(PREPROCESSED_DATA_DIR, COUNTRY, with_data=False, mode='prod')

            print(f"\n{TASK}ing {len(DEV_MODELS)} {COUNTRY} models fitted in development mode:\n\n{DEV_MODELS}\n")
            print("-------------------------------------------------------------------------")
            print(f"\n{TASK}ing {len(PROD_MODELS)} {COUNTRY} models fitted in production mode\n\n{PROD_MODELS}\n")
                        
    # predict from models
    elif TASK == 'predict':
        
        print(f"\n{TASK}ing revenue over the 30 next days starting from {DAY}-{MONTH}-{YEAR}")
        print(f"for {COUNTRY} in {MODE} mode")

        model_predict(PREPROCESSED_DATA_DIR, COUNTRY, YEAR, MONTH, DAY, MODE, models_dict= None)