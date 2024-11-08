import os
import time
import uuid
from datetime import date, datetime
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


def update_train_log(country: str, date_range: str, metric: str, runtime: float, 
                     version: int, mode: str, model_name: str, scaler_name: str, 
                     best_grid_parameters: str) -> None:
    """
    Append a new entry to the training log file with details about the model training run.

    Parameters:
        country : The country for which the model is trained.
        date_range : The date range for the training data.
        metric : The metric used for evaluation.
        runtime : The time taken for the training process.
        version : The version number of the model.
        mode : The mode of the training (e.g., 'train').
        model_name : The name of the model being trained.
        scaler_name : The name of the scaler used.
        best_grid_parameters : The best parameters found after the grid search
    """
    # Configure logger to log CSV-like structured entries
    today_str = date.today().strftime("%Y-%m-%d")
    train_logfile = os.path.join(LOG_DIR, f"{mode}-trained_on_{today_str}_logfile.csv")
    logger.add(train_logfile, format="{message}", rotation="00:00", retention="60 days")

    # Define the header
    header = "unique_id,timestamp,date_range,country,model_version,runtime,mode,model_name,scaler_name,best_model_parameters,metric"
    write_header_if_needed(train_logfile, header)

    # Prepare log data
    log_data = {
        "unique_id": str(uuid.uuid4())[:13],
        "timestamp": datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"),
        "date_range": date_range,
        "country": country,
        "model_version": f'v{version}',
        "runtime": runtime,
        "mode": mode,
        "model_name": model_name,
        "scaler_name": scaler_name,
        "best_model_parameters": best_grid_parameters,
        "metric": metric}

    # Format data as CSV line and log it
    csv_line = ",".join(str(value) for value in log_data.values())
    logger.info(csv_line)
    print(f"\nTraining log entry saved to {train_logfile}\n")


def update_predict_log(country: str, target_date: str, y_pred: str, y_proba: list, 
                       runtime: str, rmse: int, mode: str) -> None:
    """
    Update the prediction log file with the prediction results.

    Parameters:
        country (str): The country for which the prediction is made.
        target_date (str): The date of the prediction.
        y_pred (str): The predicted value.
        y_proba (list): The predicted probabilities.
        runtime (str): The time taken for the prediction process.
        rmse (int): The root mean square error of the prediction.
        mode (str): The mode of the prediction (e.g., 'predict').
    """
    # Configure logger to log CSV-like structured entries
    today_str = date.today().strftime("%Y-%m-%d")
    predict_logfile = os.path.join(LOG_DIR, f"{mode}-predicted_on_{today_str}_logfile.csv")
    logger.add(predict_logfile, format="{message}", rotation="00:00", retention="60 days")
    
    # Define the header
    header = "unique_id,timestamp,mode,country,target_date,y_pred,y_proba,test_set_rmse,runtime"
    write_header_if_needed(predict_logfile, header)

    # Prepare log data
    log_data = {
        "unique_id": str(uuid.uuid4())[:13],
        "timestamp": datetime.fromtimestamp(time.time()).strftime("%H:%M:%S"),
        "mode": mode,
        "country": country,
        "target_date": target_date,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "rmse": rmse,
        "runtime": runtime}
    
    # Format data as CSV line and log it
    csv_line = ",".join(str(value) for value in log_data.values())
    logger.info(csv_line)
    print(f"\nPrediction log entry saved to {predict_logfile}\n")
    
    
def _is_validate_date(year: int, month: int, day: int) -> bool:
    """Validate if the given year, month, and day form a valid date."""
    try:
        datetime(year, month, day)
        return True
    except ValueError:
        return False

def _parse_logs(logs) -> list:
    """Parse the log entries, expecting a list of CSV lines."""
    if logs is None or logs == [] or not isinstance(logs, list):
        return []
    return [line.split(",") for line in logs]

def get_log(mode: str, task: str, year: int, month: int, day: int) -> list:
    """Retrieve logs based on the mode, task, and date.

    Parameters:
        mode (str): The mode of the log (e.g., 'train', 'predict').
        task (str): The task associated with the logs (e.g., 'training', 'prediction').
        year (int): The year of the logs to retrieve.
        month (int): The month of the logs to retrieve.
        day (int): The day of the logs to retrieve.

    Returns:
        list: A list of log entries, or an empty list if no logs are found.
    """
    if not _is_validate_date(year, month, day):
        print(f"\nInvalid date: Year: {year}, Month: {month}, Day: {day}\n")
        return
    
    target_date = f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
    log_filename = f"{mode}-{task}ed_on_{target_date}_logfile.csv" 
    log_path = os.path.join(LOG_DIR, log_filename)

    if os.path.exists(log_path) and os.path.isfile(log_path):
        with open(log_path, "r") as f:
            log_lines = f.read().splitlines()
            return _parse_logs(log_lines)
    else:
        print(f"\nNo log found for {task} in mode: {mode} on date: {target_date}\n")
        return

