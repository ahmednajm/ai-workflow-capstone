#!/usr/bin/env python3

## Import necessary libraries 
import os, pickle, glob
from argparse import ArgumentParser
import pandas as pd
from scipy.stats import wasserstein_distance
from umap import UMAP
from sklearn.pipeline      import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble      import IsolationForest
from server_side.logger_module import update_training_data_drift_log
import warnings
warnings.filterwarnings('ignore')

# Set global random seed for reproducibility
import numpy as np
np.random.seed(42)

PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data')
MODEL_DIR         = os.path.join(os.path.dirname(__file__), '.' , "models")
TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), '.' , "training_data")
LOG_DIR           = os.path.join(os.path.dirname(__file__), '.' , "server_logs")

def _get_latest_two_training_data_versions(mode, country):
    """
    Load the penultimate and last training data for the given mode and country.
    Returns: (penultimate version, X of penultimate version, y of penultimate version), 
             (last version, X of last version, y of last version)
    """
    # Define the file pattern
    file_pattern = os.path.join(TRAINING_DATA_DIR, 
                                f"{mode}-{country}-*-*-*-*.pickle")  # Wildcards for version

    # Find all files that match the pattern
    matching_files = glob.glob(file_pattern)
    if not matching_files:
        print(f"\nCannot find training data for {mode}, {country}")
        print("Have you trained the model?")
        return None

    # Sorting files by version and RMSE
    matching_files.sort(key=lambda k: (
        -int(k.split("-v")[1].split("_")[1].split('.')[0]), 
        -int(k.split("-v")[1].split("_")[0]),  
        int(k.split("{'RMSE': ")[1].split(",")[0]) 
    ))

    # Get the last two consecutive versions
    new_version_file = matching_files[-2]
    last_version_file = matching_files[-1]
    
    # Extract the version from the filenames 
    new_version  = new_version_file.split('-')[-1].split('.')[0]  
    last_version = last_version_file.split('-')[-1].split('.')[0]  

    # Load the training data for both versions
    with open(new_version_file, 'rb') as tmp:
        new_training_data = pickle.load(tmp)

    with open(last_version_file, 'rb') as tmp:
        last_training_data = pickle.load(tmp)

    # Return (penultimate version, X, y) and (last version, X, y)
    return (
        (last_version,last_training_data['X'], last_training_data['y']),
        (new_version, new_training_data['X'], new_training_data['y'])
    )
    
    
def _get_training_monitoring_tools(X,y):
    """
    Outlier analysis and Wasserstein distance evaluation
    """

    # Set Up Anomaly Detection Pipeline
    pipe = Pipeline([
                ('scaler'            , RobustScaler()),              
                ('dim_reduction'     , UMAP(n_components=2, random_state=42)),   
                ('outliers_detection', IsolationForest(random_state=42, contamination=0.01))       
                    ])
    
    pipe.fit(X)
    
    # Number of bootstrap samples to use in the process
    nbr_bs_samples = 100  

    # Initialize Variables for Bootstrapping
    outliers_X_percentage = pd.Series([0.0] * nbr_bs_samples)
    wasserstein_X         = pd.Series([0.0] * nbr_bs_samples)
    wasserstein_y         = pd.Series([0.0] * nbr_bs_samples)

    ## Bootstrap Sampling and Outlier Detection Loop    
    for bs in range(nbr_bs_samples):

        # Generate bootstrap sample
        n_samples = round(0.80 * len(X))

        # Sampling Subset Indices
        subset_indices = pd.Series(X.index).sample(n_samples, random_state=42, replace=True).tolist()

        # Bootstrap Sample for Features and Target 
        X_bs = X.values[subset_indices, :]
        y_bs = y[subset_indices]

        # Predict outliers using IsolationForest
        outliers_X = pipe.predict(X)

        # Calculate outliers percentage
        outliers_X_percentage[bs] = 100 * (1.0 - ( outliers_X[outliers_X == 1].size / outliers_X.size ) )

        # Calculate Wasserstein distances between originals and bootstrap samples
        wasserstein_X[bs] = wasserstein_distance(X.values.flatten(), X_bs.flatten())
        wasserstein_y[bs] = wasserstein_distance(y.values.flatten(), y_bs.values.flatten())
    
    # Determine thresholds based on confidence intervals (upper bound as threshold)

    outliers_X_percentage_threshold  = outliers_X_percentage.quantile(0.975)  
    wasserstein_X_threshold          =         wasserstein_X.quantile(0.975)  
    wasserstein_y_threshold          =         wasserstein_y.quantile(0.975)

    # Prepare Results for Return
    monitoring_tools = {"outliers_pipeline"        : pipe,
                        "outliers_X_threshold"     : round(outliers_X_percentage_threshold, 2),
                        "wasserstein_X_threshold"  : round(wasserstein_X_threshold, 2),
                        "wasserstein_y_threshold"  : round(wasserstein_y_threshold, 2),
                       }
    
    return monitoring_tools

def _monitor_new_training_data(X_new, y_new, X_last, y_last):
    """
    Check for drift in new data batch against reference thresholds.
    """
    monitoring_tools = _get_training_monitoring_tools(X_last, y_last)

    # Calculate outlier percentage for new data using IsolationForest
    outliers_X_new            = monitoring_tools['outliers_pipeline'].predict(X_new)
    outliers_X_new_percentage = 100 * (1.0 - ( outliers_X_new[outliers_X_new == 1].size / outliers_X_new.size ))

    # Calculate Wasserstein distances for new data
    wasserstein_X_new = wasserstein_distance(X_last.values.flatten(), X_new.values.flatten())
    wasserstein_y_new = wasserstein_distance(y_last.values.flatten(), y_new.values.flatten())

    # Check for drift against thresholds
    drift_detected = {
        'outliers_X_percentage' : outliers_X_new_percentage > monitoring_tools['outliers_X_threshold']    ,
        'wasserstein_X'         : wasserstein_X_new         > monitoring_tools['wasserstein_X_threshold'],
        'wasserstein_y'         : wasserstein_y_new         > monitoring_tools['wasserstein_y_threshold']
    }

    return drift_detected


def detect_drift_between_training_data_versions(mode, country):
    """
    Detect drift between the penultimate and last version of training data for the specified mode and country.
    This function combines loading the data, setting up the monitoring tools, and detecting drift.
    """
    # Get the new and last training data versions
    (last_version, X_last, y_last), (new_version, X_new, y_new) = _get_latest_two_training_data_versions(mode, country)
    
    # Print the versions being compared
    print(mode, country)
    print(f"Detecting drift between training data versions: {last_version} and {new_version}")
    
    # Extract monitoring tools using the last version as reference
    monitoring_tools = _get_training_monitoring_tools(X_last, y_last)
    outliers_X_threshold = monitoring_tools["outliers_X_threshold"]
    wasserstein_X_threshold =  monitoring_tools["wasserstein_X_threshold"]
    wasserstein_y_threshold = monitoring_tools["wasserstein_y_threshold"]
                        
    # Monitor drift for the last version against the reference
    drift_results = _monitor_new_training_data(X_new, y_new, X_last, y_last)
    outliers_X_percentage = drift_results["outliers_X_percentage"]
    wasserstein_X = drift_results["wasserstein_X"]
    wasserstein_y = drift_results["wasserstein_y"]

    # Check drift results
    if any(drift_results.values()):
        print("Warning: Potential drift detected!")
        print("Drift details:", drift_results)
    else:
        print("No significant drift detected.")

    update_training_data_drift_log(mode, country, new_version, 
                                   outliers_X_threshold, wasserstein_X_threshold, wasserstein_y_threshold,
                                   outliers_X_percentage, wasserstein_X, wasserstein_y)

    return  outliers_X_threshold, wasserstein_X_threshold, wasserstein_y_threshold, outliers_X_percentage, wasserstein_X, wasserstein_y


if __name__ == "__main__":
    
    ap = ArgumentParser()
    
    ap.add_argument('-m', '--mode', choices=['dev', 'prod'], default='prod')
    
    ap.add_argument('-c', '--country', default='World')

    args    = ap.parse_args()
    
    ################################################################
    
    MODE    = args.mode
    COUNTRY = args.country


    # Detect drift between the last two versions
    outliers_X_threshold, wasserstein_X_threshold, wasserstein_y_threshold, outliers_X_percentage, wasserstein_X, wasserstein_y = detect_drift_between_training_data_versions(MODE, COUNTRY)
    
    print("outliers_X_threshold   ", outliers_X_threshold)
    print("wasserstein_X_threshold", wasserstein_X_threshold)
    print("wasserstein_y_threshold", wasserstein_y_threshold)
    print("outliers_X_percentage  ", outliers_X_percentage)
    print("wasserstein_X          ", wasserstein_X)
    print("wasserstein_y          ", wasserstein_y)