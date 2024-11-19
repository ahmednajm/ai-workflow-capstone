#!/usr/bin/env python3

## Import necessary libraries 
import os
from argparse import ArgumentParser
from ast import literal_eval
import pandas as pd
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data')
MODEL_DIR         = os.path.join(os.path.dirname(__file__), '.' , "models")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), '.' , "training_data")
LOG_DIR           = os.path.join(os.path.dirname(__file__), '.' , "server_logs")

TRAINING_RMSE_THRESHOLD_MULTIPLIER = 1.1

def reading_training_metrics(mode, country):
    
    # Acceding to the training log file
    training_log_file_name = f"{mode}-training-logfile.csv"
    training_log_file_path = os.path.join(os.path.dirname(__file__), LOG_DIR, training_log_file_name)
    print('\ntraining_log_file_name:\n\n',training_log_file_path,'\n')

    # Read CSV training log file
    training_log_df = pd.read_csv(training_log_file_path, 
                     index_col = False, 
                     usecols = ['country', 'date', 'version', 'date_range', 'runtime', 'eval_metrics'],
                     sep='|'
                     ).query(f"country == '{country}'")
    
    # Create separate RMSE and MAPE columns 
    training_log_df['rmse'] = training_log_df['eval_metrics'].apply(lambda x : literal_eval(x)['RMSE'])
    training_log_df['mape'] = training_log_df['eval_metrics'].apply(lambda x : literal_eval(x)['MAPE'])
    training_log_df.drop(columns=['eval_metrics','country'], inplace=True)
    training_log_df['runtime_sec'] = pd.to_timedelta(training_log_df['runtime']).dt.total_seconds()

    # Acceding to the drift log file
    drift_log_file_name = f"{mode}-training-data-drift-logfile.csv"
    drift_log_file_path = os.path.join(os.path.dirname(__file__), LOG_DIR, drift_log_file_name)
    print('\ndrift_log_file_path:\n\n',drift_log_file_path,'\n')

    # Read CSV drift log file
    drift_log_df = pd.read_csv(drift_log_file_path, 
                     index_col = False, 
                     sep='|'
                     ).query(f"country == '{country}'")
    
    drift_log_df.drop(columns=['country'], inplace=True)
    
    return training_log_df, drift_log_df

def reading_predictions_metrics(mode, country):

    # Acceding to the predictions log file
    predictions_log_file_name = f"{mode}-predictions-logfile.csv"
    predictions_log_file_path = os.path.join(os.path.dirname(__file__), LOG_DIR, predictions_log_file_name)
    print('\npredictions_log_file_path:\n\n',predictions_log_file_path,'\n')

    # Read CSV predictions log file
    predictions_log_df = pd.read_csv(predictions_log_file_path, 
                     index_col = False, 
                     usecols = ['country', 'date_range', 'target_date', 'y_pred', 'test_set_rmse', 'version'],
                     sep='|'
                     ).query(f"country == '{country}'")
    
    predictions_log_df.drop(columns=['country'], inplace=True)

    # Acceding to the target_next_30_days_revenue actual
    actual_data_file_name = f'{country}_preprocessed_data.csv'
    actual_data_file_path = os.path.join(os.path.dirname(__file__), PREPROCESSED_DATA_DIR, actual_data_file_name)
    print('\nactual_data_file_path:\n\n',actual_data_file_path,'\n')

    # Read CSV preprocessed file
    actual_df = pd.read_csv(actual_data_file_path,
                                  index_col = False, 
                                  usecols = ['date','revenue','target_next_30_days_revenue'],
                                  sep=',')
    
    actual_df['revenue_actual'] = actual_df['revenue']
    actual_df['target_next_30_days_revenue_actual'] = actual_df['target_next_30_days_revenue']
    actual_df.drop(columns=['revenue', 'target_next_30_days_revenue'], inplace=True)

    # Merge DataFrames on target_date (from predictions_log_df) and date (from actual_df)
    merged_df = pd.merge(predictions_log_df, actual_df, left_on='target_date', right_on='date', how='left')

    # Drop the redundant 'date' column
    merged_df = merged_df.drop(columns=['date'])
    # Drop the rows with Nan (no actuals yet)
    merged_df = merged_df.dropna(subset=['target_next_30_days_revenue_actual', 'y_pred'])

    # Calculate error metrics
    merged_df['predictions_mse'] = mean_squared_error(merged_df['target_next_30_days_revenue_actual'], merged_df['y_pred'])
    merged_df['predictions_rmse'] = merged_df['predictions_mse']**0.5
    merged_df['predictions_mape'] = (abs((merged_df['target_next_30_days_revenue_actual'] - merged_df['y_pred']) / merged_df['target_next_30_days_revenue_actual']).mean()) * 100  # Mean Absolute Percentage Error
    
    merged_df.drop(columns=['predictions_mse'], inplace=True)

    # Compare predictions RMSE to the threshold and create an alert column
    merged_df.rename(columns={'test_set_rmse': 'training_rmse',
                              'target_next_30_days_revenue_actual': 'actual_next_30_days_revenue',
                              'y_pred': 'predicted_next_30_days_revenue',
                              'version': 'model_version'},
                     inplace=True)
    
    merged_df['rmse_threshold'] = TRAINING_RMSE_THRESHOLD_MULTIPLIER * merged_df['training_rmse']
    merged_df['alert'] = merged_df['predictions_rmse'] > merged_df['rmse_threshold'] 
    
    # Prepare the return merged_df
    reordered_columns = ['target_date', 'date_range',
                         'actual_next_30_days_revenue', 'predicted_next_30_days_revenue',
                         'model_version', 'training_rmse', 'predictions_rmse', 'alert']
    merged_df = merged_df[reordered_columns]

    return merged_df


if __name__ == "__main__":
    
    ap = ArgumentParser()
    
    ap.add_argument('-m', '--mode', choices=['dev', 'prod'], default='prod')    
    ap.add_argument('-c', '--country', default='World')
    ap.add_argument('-t', '--task', choices=['train', 'predict'], default='train')
    args    = ap.parse_args()
    
    ################################################################
    
    TASK    = args.task
    MODE    = args.mode
    COUNTRY = args.country

    ################################################################

    # train models
    if TASK == 'train':
        training_log_df, drift_log_df = reading_training_metrics(MODE, COUNTRY)
        
        print('\nReading training metrics\n')
        print(MODE, COUNTRY)
        print('\ntraining logs\n')
        print(training_log_df)
        print('\n-------------------------------------------------------------------')

        print('\ndrift logs\n')
        pd.set_option('display.max_columns', None)
        print(drift_log_df)
        print('\n')

    
    if TASK == 'predict':
        predictions_metrics = reading_predictions_metrics(MODE, COUNTRY)
        
        print('\nReading predictions metrics')
        print('mode :', MODE, 'country :', COUNTRY, '\n')
        pd.set_option('display.max_columns', None)
        print(predictions_metrics)
        print('\n')

