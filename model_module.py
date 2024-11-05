## Import necessary libraries 
import os, re, time, joblib
from typing import tuple
from datetime import date

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from preprocessing_module import engineer_features
from logger_module import update_predict_log, update_train_log

import warnings
warnings.filterwarnings('ignore')


def model_train(clean_data_dir: str, country: str, version: int,
                model, model_param_grid: dict, model_scaler, mode: str) -> None:
      """
    Trains a machine learning model on a specified dataset, evaluates its performance, and saves the model.

    This function performs the following steps:
    1. Prepares data for training.
    2. Tunes model hyperparameters using grid search.
    3. Evaluates the model's performance on a test set using RMSE and MAPE metrics.
    4. Retrains the model using the entire dataset.
    5. Saves the trained model to the filesystem.

    Parameters:
    clean_data_dir   : The directory containing cleaned data for model training.
    country          : The country for which the model is being trained.
    version          : The version number of the model.
    model            : The machine learning model to be trained (e.g., RandomForestRegressor).
    model_param_grid : The hyperparameter grid for tuning the model.
    model_scaler     : The scaler to preprocess input features.
    mode             : A string indicating the training mode (e.g., 'development', 'production').

    Returns: The function saves the trained model to a specified directory.
    """
    
    print("\n... Perform training on train set")

    # start timer for runtime
    model_train_time_start = time.time()
    
    # prepare the data
    X, y, dates = engineer_features(clean_data_dir, country, training=True)

    # Execute this block only if ensemble of decision trees model
    if isinstance(model, (RandomForestRegressor, ExtraTreesRegressor)):
        X = X.dropna()  
        y = y[X.index]
        dates = dates[X.index]

    # Define the date range for logging
    max_date = dates.iloc[-1].strftime('%Y-%m-%d')
    min_date = dates.iloc[0].strftime('%Y-%m-%d')
    date_range = f"{min_date}:{max_date}"
    
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
    eval_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    # Define the date range for logging
    eval_metrics = f"[RMSE={eval_rmse},MAPE={eval_mape:.1f}%]"

    # Retrain using all data
    print("\n... Retraining model on all data")
    try:
        grid.fit(X, y)
    except Exception as e:
        print(f"Error during model retraining on all data: {e}")
        return None

    # Best model
    fitted_model = grid.best_estimator_
    # make the model name more system compatible and file-friendly when saving the model
    model_name = fitted_model.named_steps['model'].__class__.__name__
    scaler_name = fitted_model.named_steps['model_scaler'].__class__.__name__
    
    # Create model directory 
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model directory: {MODEL_DIR}")        
    
    # Define the file path 
    saved_model = os.path.join(MODEL_DIR, f"{mode}-{country}-{model_name}-{scaler_name}-{eval_metrics}-v{version}.joblib")
    # Save the fitted model
    print(f"\n... Saving model version as : {saved_model}\n")
    joblib.dump(fitted_model, saved_model)

    print(f'\nModel "{mode}-{country}-{model_name}-{scaler_name}-{eval_metrics}-v{version}" trained')

    # Calculate runtime duration
    m, s = divmod(time.time() - model_train_time_start, 60)
    h, m = divmod(m, 60)
    model_train_runtime = "%02d:%02d:%02d" % (h, m, s)
    print(f'\nModel training runtime : {model_train_runtime}\n')

    # update training log
    update_train_log(country, date_range, eval_metrics, model_train_runtime, version, mode, model_name, scaler_name)
    

def model_load(clean_data_dir: str, country: str, with_data: bool, mode: str) -> tuple[dict, dict]:
    """
    Load trained models from disk and optionally prepare the training data.

    This function retrieves all models matching the specified criteria and can also load the relevant 
    feature and target data for the specified country.

    Parameters:
    ----------
    clean_data_dir : Directory path containing cleaned data.
    country : The country for which models are being loaded.
    with_data : Flag indicating whether to load the training data.
    mode : The mode of the model (e.g., 'train', 'test').

    Returns: A tuple containing:
        - A dictionary of loaded models keyed by model name.
        - A dictionary of training data (features and target) if with_data is True, otherwise empty.
    """
    
    # Define model directory path using Path for portability
    model_dir = "models"

    # Initialize return values
    data_dict = {}
    models_dict = {}

    # Retrieve all model filenames matching the specified prefix and country
    models = [filename for filename in os.listdir(model_dir) 
              if (mode in filename) and (country in filename)]

    # Check if any models were found
    if not models:
        print(f"\nNo model found for {country}. Have you trained it yet?\n")
        return models_dict, data_dict

    # Load models into a dictionary, keyed by model name without file extension
    models_dict = {model.replace('.joblib', ''): joblib.load(os.path.join(model_dir, model)) for model in models}

    # Only prepare data if with_data is True
    if with_data:
        # Get the dates, the features and target variable
        X, y, dates = engineer_features(clean_data_dir, country, training=True)
        # Convert dates to string format for consistency
        dates = pd.to_datetime(dates).dt.strftime('%Y-%m-%d').tolist()
        # Compile the data into a dictionary for easy access
        data_dict = {"X": X, "y": y, "dates": dates}
    
    # If with_data is False, return only models_dict
    return models_dict, data_dict


def nearest_date(items: list, pivot: date) -> date:
    """Find the nearest date to the given pivot date.

    Parameters:
    items : A list of date strings in 'YYYY-MM-DD' format.
    pivot : The date to which the nearest date is sought.

    Returns: The nearest date from the items list to the pivot date.

    Raises: ValueError If the items list is empty.
    """
    if not items:
        raise ValueError("ERROR: items list is empty.")
    return min(items, key=lambda x: abs(date.fromisoformat(x) - pivot))


def model_predict(clean_data_dir: str, country: str, year: int, month: int, day: int
                  , mode: str, models_dict: dict = None) -> None:
    """
    Predict revenue for a specified date using the best available model.

    This function loads the relevant model, checks the target date, and makes a revenue prediction.
    It logs the prediction results along with runtime information.

    Parameters:
    clean_data_dir : Directory path containing cleaned data.
    country : The country for which the prediction is being made.
    year : The year of the target date.
    month : The month of the target date.
    day : The day of the target date.
    mode : The mode of the model (e.g., 'development', 'production').
    models_dict : Optional, a dictionary of loaded models (if already loaded).

    Returns: The prediction
    """
    
    ## Start timer for runtime
    model_predict_time_start = time.time()
    
    ## Load model if needed
    print(f"\n... Loading models")    
    if models_dict is None:
        models_dict, data_dict = model_load(clean_data_dir, country, mode=mode, with_data=True)

    # Input checks   
    if not any(f'{country}' in key for key in models_dict.keys()):
        return

    # Finding the model with the latest version
    best_model_key = min(models_dict.keys(), key=lambda k: int((k.split("[RMSE=")[1]).split(",")[0]))
    best_model = models_dict[best_model_key]
    print(f"\nBest Model trained for {country}: {best_model}")    
    best_rmse = int(best_model_key.split("[RMSE=")[1].split(",")[0])


    # Validate Date Components
    for d in [year, month, day]:
        if re.search(r"\D", str(d)):  
            raise Exception("ERROR (model_predict) - invalid year, month, or day")
        
    ## Load data
    data = data_dict

    # Convert dates to Datetime Format
    dates = pd.to_datetime(data['dates'])

    # Check the target date
    target_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    print(f"\n... Checking if {target_date} is in the range.")

    # Validate Target Date and Find Nearest Date if Out of Range
    if target_date not in dates.astype(str).values:  # Convert to string for comparison
        print(f"ERROR (model_predict) - date {target_date} not in range [ {data['dates'].iloc[0]} - {data['dates'].iloc[-1]} ]")
        target_date = nearest_date(data['dates'], date.fromisoformat(target_date))
        print(f"Nearest target date is {target_date}")
    else:
        print("Target date is in the range.")

    # Get the index of the target_date
    target_date_indx = dates.get_loc(target_date)  # Correct way to get the index

    # Query the corresponding row
    query = data['X'].iloc[[target_date_indx]]  

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
    update_predict_log(country, target_date, y_pred_, y_proba, model_predict_runtime, best_rmse, mode)

    return print(f'The expected revenue over the next 30 days is {y_pred[0]:,.0f} £')
    
    
    

if __name__ == "__main__":
    
    from argparse import ArgumentParser
    ap = ArgumentParser()
    
    ap.add_argument('-model', '--model', choices=['rf', 'et'], default='rf',
                    help='(rf) RandomForestRegressor or (et) ExtraTreesRegressor (default)')
    
    ap.add_argument('-s', '--scaler', choices=['ss', 'rs'], default='ss',
                    help='(ss) StandardScaler or (rs) RobustScaler (default)')
    
    ap.add_argument('-t', '--task', choices=['train', 'load', 'predict'], default='predict',
                    help='choosing whether we are training a model, or predicting from an already-trained model,\
                        or just loading existing models')

    ap.add_argument('-m', '--mode', choices=['dev', 'prod'], default='prod',
                    help='Train either a development or production model. Omitting this implies loading an already-trained model')
    
    ap.add_argument('-c', '--country', default='all_countries',
                    help="The country to predict revenue for (default: 'all_countries')")

    args    = ap.parse_args()
    
    TASK    = args.task
    MODE    = args.mode
    COUNTRY = args.country
    
    MODEL   = RandomForestRegressor(random_state=42) if args.model  == 'rf' else ExtraTreesRegressor(random_state=42)
    SCALER  = StandardScaler()                       if args.scaler == 'ss' else RobustScaler()
    
    MODEL_PARAM_GRID = { 'model__n_estimators' : [90, 100, 110, 120, 130] ,
                         'model__max_depth'    : [None, 5, 10, 15]        ,
                         'model__criterion'    : ['squared_error']        }

    CLEAN_DATA_DIR = os.path.join(os.path.dirname(__file__), '.', 'data', 'clean_data')
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '.', "models")
    LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "logs")
    
    VERSION = 1

    YEAR    = 2019
    MONTH   = 1
    DAY     = 7 
    
    # train models
    if TASK == 'train':
        print(f"\n{TASK}ing a {SCALER} + {MODEL} model for {COUNTRY} in {MODE} mode")
        model_train(CLEAN_DATA_DIR, COUNTRY, VERSION, MODEL, MODEL_PARAM_GRID, SCALER, MODE)
        
    # load models
    elif TASK == 'load':
            DEV_MODELS,  _  = model_load(CLEAN_DATA_DIR, COUNTRY, with_data=False, mode='dev' )
            PROD_MODELS, _  = model_load(CLEAN_DATA_DIR, COUNTRY, with_data=False, mode='prod')
            
            print(f"\n{TASK}ing {len(DEV_MODELS)} {COUNTRY} models fitted in development mode:\n\n{DEV_MODELS}\n")
            print("-------------------------------------------------------------------------")
            print(f"\n{TASK}ing {len(PROD_MODELS)} {COUNTRY} models fitted in production mode\n\n{PROD_MODELS}\n")
                        
    # predict from models
    elif TASK == 'predict':
        print(f"\n{TASK}ing for {COUNTRY} in {MODE} mode")
        model_predict(CLEAN_DATA_DIR, COUNTRY, YEAR, MONTH, DAY, MODE, models_dict= None)