#!/usr/bin/env python3

import os, time, uuid, glob
from datetime import date, datetime
import pandas as pd
from loguru import logger

LOG_DIR = os.path.join(os.path.dirname(__file__), '.', "server_logs")

# Ensure log directory exists
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

def write_header_if_needed(logfile: str, header: str) -> None:
    """
    Write the header to the log file if it doesn't already exist or is empty.

    Parameters:
        logfile (str): The path to the log file.
        header (str): The header string to write to the log file.
    """
    if not os.path.exists(logfile) or os.path.getsize(logfile) == 0:
        with open(logfile, 'w') as f:
            f.write(header + "\n")


def update_train_log(mode: str, country: str, date_range: str, 
                     model_name: str, scaler_name: str, best_grid_parameters: str, 
                     eval_metrics: str, version: str, runtime: float) -> None:
    """
    Append a new entry to the training log file with details about the model training run.
    Parameters:
        mode : The mode of the training (e.g., 'train').
        country : The country for which the model is trained.
        date_range : The date range for the training data.
        model_name : The name of the model being trained.
        scaler_name : The name of the scaler used.
        best_grid_parameters : The best parameters found after the grid search
        eval_metrics : The metric used for evaluation.
        version : The version number of the model.
        runtime : The time taken for the training process.
    """
    
    # Configure logger to log CSV-like structured entries
    train_logfile = os.path.join(LOG_DIR, f"{mode}-training-logfile.csv")
    logger.add(train_logfile , 
               format = "{message}" , 
               retention = None ,
               rotation = None)

    # Define the header
    header = "unique_id|date|timestamp|mode|country|date_range|model_name|scaler_name|best_model_parameters|eval_metrics|version|runtime"
              
    write_header_if_needed(train_logfile, header)

    # Prepare log data
    log_data = {
        "unique_id": str(uuid.uuid4())[:13],
        "date": date.today().strftime("%Y-%m-%d"),
        "timestamp": datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"),
        "mode": mode,
        "country": country,
        "date_range": date_range,
        "model_name": model_name,
        "scaler_name": scaler_name,
        "best_model_parameters": best_grid_parameters,
        "eval_metrics": eval_metrics,
        "version": version,
        "runtime": runtime}

    # Format data as CSV line and log it
    csv_line = "|".join(str(value) for value in log_data.values())
    logger.info(csv_line)

              
def update_predict_log(mode: str, country: str, date_range: str, target_date: str, 
                       y_pred: str, y_proba: list, rmse: int, version: str, runtime: str) -> None:  
    """
    Update the prediction log file with the prediction results.
    Parameters:
        mode (str): The mode of the prediction (e.g., 'predict').
        country (str): The country for which the prediction is made.
        date_range (str): The date range for the training data.
        target_date (str): The date of the prediction.
        y_pred (str): The predicted value.
        y_proba (list): The predicted probabilities.
        rmse (int): The root mean square error of the prediction.
        version : The version of the model.
        runtime (str): The time taken for the prediction process.
        y_actual_previous
    """

    # Configure logger to log CSV-like structured entries
    predict_logfile = os.path.join(LOG_DIR, f"{mode}-predictions-logfile.csv")
    logger.add(predict_logfile , 
               format = "{message}" , 
               retention = None ,
               rotation = None)
    # Define the header
    header = "unique_id|date|timestamp|mode|country|date_range|target_date|y_pred|y_proba|test_set_rmse|version|runtime"#|y_actual_previous"
              
    write_header_if_needed(predict_logfile, header)

    # Prepare log data
    log_data = {
        "unique_id": str(uuid.uuid4())[:13],
        "date":  date.today().strftime("%Y-%m-%d"),
        "timestamp": datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"),
        "mode": mode,
        "country": country,
        "date_range": date_range,
        "target_date": target_date,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "rmse": rmse,
        "version": version,
        "runtime": runtime,
        #"y_actual_previous": y_actual_previous
        }
    
    # Format data as CSV line and log it
    csv_line = "|".join(str(value) for value in log_data.values())
    logger.info(csv_line)
    
    
def _is_validate_date(year: int, month: int) -> bool:
    """Validate if the given year, month, and day form a valid date."""
    try:
        datetime(year, month)
        return True
    except ValueError:
        return False

def _parse_logs(logs) -> list:
    """Parse the log entries, expecting a list of CSV lines."""
    if logs is None or logs == [] or not isinstance(logs, list):
        return []
    return [line.split("|") for line in logs]

def get_logs(mode: str, task: str, year: int, month: int) -> list:
    """Retrieve logs based on the mode, task, and date.

    Parameters:
        mode (str): The mode of the log (e.g., 'train', 'predict').
        task (str): The task associated with the logs (e.g., 'training', 'prediction').
        year (int): The year of the logs to retrieve.
        month (int): The month of the logs to retrieve.

    Returns:
        list: A list of log entries, or an empty list if no logs are found.
    """
    
    target_date = f"{year}-{str(month).zfill(2)}"
    log_filename_pattern = os.path.join(LOG_DIR, f"{mode}-{task}*-logfile.csv")
    
    # find files matching the pattern
    log_filenames = glob.glob(log_filename_pattern)
    
    # Check if files were found
    if not log_filenames:
        print(f"\nNo log files found for {task} in mode: {mode}\n")
        return None

    # Get the file
    log_filename = log_filenames[0]
    log_filename_path = os.path.join(LOG_DIR, log_filename)

    # Check if the file exists and is a valid file
    if os.path.exists(log_filename_path) and os.path.isfile(log_filename_path):
        with open(log_filename_path, "r") as f:
            log_lines = f.read().splitlines()
            return _parse_logs(log_lines)
    else:
        print(f"\nThe log file {log_filename_path} does not exist or is not a valid file.\n")
        return None


def get_log(mode: str, task: str, year: int, month: int) -> list:
    
    # List all files in the log directory
    all_files = os.listdir(LOG_DIR)
    # Filter log files based on task and mode
    log_file_name = [filename for filename in all_files if f"{mode}-{task}" in filename][0]
    #
    log_file_path = os.path.join(os.path.dirname(__file__), LOG_DIR, log_file_name)
    # 
    print('\log_file:\n\n',log_file_path,'\n')

    log_file_df = pd.read_csv(log_file_path,
                                  sep='|')    
    
    target = f"{year:04d}-{month:02d}"
    
    # Filtrer les lignes où target est dans date_range
    filtered_df = log_file_df[
        log_file_df['date_range'].str.contains(target)
    ]
    
    return filtered_df


def update_training_data_drift_log(mode, country, version, 
                     outliers_X_threshold, wasserstein_X_threshold, wasserstein_y_threshold,
                     outliers_X_percentage, wasserstein_X, wasserstein_y) -> None: 
     
    # Configure logger to log CSV-like structured entries
    drift_logfile = os.path.join(LOG_DIR, f"{mode}-training-data-drift-logfile.csv")
    logger.add(drift_logfile , 
               format = "{message}" , 
               retention = None ,
               rotation = None)

    # Define the header
    header = "date|mode|country|version|outliers_X_threshold|wasserstein_X_threshold|wasserstein_y_threshold|outliers_X_percentage|wasserstein_X|wasserstein_y"
              
    write_header_if_needed(drift_logfile, header)

    # Prepare log data
    log_data = {
        "date": date.today().strftime("%Y-%m-%d"),
        "mode": mode,
        "country": country,
        "version": version,
        "outliers_X_threshold": outliers_X_threshold,
        "wasserstein_X_threshold": wasserstein_X_threshold,
        "wasserstein_y_threshold": wasserstein_y_threshold,
        "outliers_X_percentage": outliers_X_percentage,
        "wasserstein_X": wasserstein_X,
        "wasserstein_y": wasserstein_y,
    }

    # Format data as CSV line and log it
    csv_line = "|".join(str(value) for value in log_data.values())
    logger.info(csv_line)

if __name__ == "__main__":
    
    
    desired_log = get_log(mode='dev', task='train', year=2019, month=7)
    print(desired_log)
