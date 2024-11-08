
import numpy as np
import pandas as pd
import scipy.stats as stats                                 # General statistical functions
from sklearn.ensemble import IsolationForest                # Anomaly detection
import statsmodels.tsa.stattools as ts


def check_missing_values(df):
    """
    Check for missing values in the dataframe and summarize them.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.Series: A summary of columns with missing values.
    """
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print("\nThere are some missing values in the dataframe\n")
        print("\nMissing values Summary\n{}".format("-"*25))
    else:
        print("\nThere are no missing values in the dataframe")
    
    # Filter columns with missing values
    missing_data = missing_values[missing_values > 0]
    for column, missing_count in missing_data.items():
        print(f"\nThere are {missing_count:,.0f} missing values in the {column} column.\n")



def drop_duplicate_data(df):
    """
    Removes duplicate rows from the DataFrame and prints a summary of the operation.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with duplicates removed.
    """
    print("\nDuplicate rows Summary\n{}".format("-"*22))
    size_before = len(df)
    df_dropped_duplicate_data = df.drop_duplicates(keep="first")
    size_after = len(df_dropped_duplicate_data)
    print(f"\nThe total number of rows before dropping duplicates is {size_before:,.0f}")
    print(f"\n... Removed {size_before - size_after:,.0f} duplicate rows.")
    print(f"\nThe total number of rows after dropping duplicates is {size_after:,.0f}\n")
    return df_dropped_duplicate_data



def drop_invalid_data(df):
    """
    Exclude rows with invalid data based on price and times viewed.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    pd.DataFrame: Dataframe with invalid rows excluded.
    """

    # Select numerical columns
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    print(f'\nThe numerical columns are : {" and ".join(numerical_cols)}\n')
    size_before_invalid_data = len(df)
    print(f"The total number of rows before dropping any invalid data is {size_before_invalid_data:,.0f}.\n")

    
    print("\nData Quality Summary\n{}".format("-" * 26))

    # Start with the full dataframe and iteratively filter out invalid rows
    df_dropped_invalid_data = df.copy()

    # Print the number of invalid data points for each numerical column
    for col in numerical_cols:
        invalid_data_count = (df[col] <= 0).sum()
        print(f"\nThe total number of invalid data points in {col} is {(invalid_data_count):,.0f}")

        # Exclude rows with invalid data in the current column
        df_dropped_invalid_data = df_dropped_invalid_data[df_dropped_invalid_data[col] > 0]

    size_after_invalid_data = len(df_dropped_invalid_data)
    print(f"\n... Removed {(size_before_invalid_data - size_after_invalid_data):,.0f} rows with invalid data.")
    print(f"\nThe total number of rows after dropping all invalid data is {size_after_invalid_data:,.0f}.\n")

    return df_dropped_invalid_data
    


def detect_outliers(df, column, z_threshold, contamination):
    """
    Detect outliers in the specified column of the DataFrame for each country.
    
    Parameters:
    - df: DataFrame to detect outliers in.
    - columns: a specific column to check for outliers.
    - z_threshold: Z-score threshold for identifying outliers.
    - contamination: Fraction of outliers in the data (for Isolation Forest and LOF).
    
    Returns:
    - outlier_mask: Boolean mask indicating which rows are outliers.
    """
    
    outlier_mask = pd.Series(False, index=df.index)  # Initialize the mask with False

    # Iterate over each country in the DataFrame
    for country, group in df.groupby('country'):
        
        # Initialize mask for the current group
        group_mask = pd.Series(False, index=group.index)  
        
        # Z-Score Method
        z_scores = stats.zscore(group[column])
        abs_z_scores = abs(z_scores)
        z_score_outliers = abs_z_scores > z_threshold
        
        # IQR Method
        Q1 = group[column].quantile(0.25)
        Q3 = group[column].quantile(0.75)
        IQR = Q3 - Q1
        iqr_outliers = (group[column] < (Q1 - 1.5 * IQR)) | (group[column] > (Q3 + 1.5 * IQR))
        
        # Isolation Forest
        isolation_model = IsolationForest(contamination=contamination)
        isolation_outliers = isolation_model.fit_predict(group[[column]]) == -1
                
        # Combine outlier masks for the current country
        group_mask = z_score_outliers | iqr_outliers | isolation_outliers 
        
        # Update the overall outlier mask
        outlier_mask.update(group_mask)

    return outlier_mask



def drop_outliers(df):
    
    combined_outliers = pd.Series(False, index=df.index)
    size_before_handling_outliers = len(df)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    print(f'\nThe numerical columns are : {" and ".join(numerical_cols)}\n')

    for col in numerical_cols:
        col_outliers = detect_outliers(df, column=col, z_threshold=3, contamination=0.01)
        df_with_col_outliers = df[col_outliers]

        print(f"\nThere are {(len(df[col_outliers])):,.0f} rows identified with outliers in {col} data")
        
        # Update combined outliers
        combined_outliers |= col_outliers
        
    print(f"\nOverall, there are {(len(df[combined_outliers])):,.0f} rows identified with outliers\n")
    
    print("\nOutliers Summary\n{}".format("-"*26))
    print(f"\nThe total number of rows before dropping outliers is {(size_before_handling_outliers):,.0f}.")

    df_dropped_outliers = df[~combined_outliers]
    
    size_after_handling_outliers  = len(df_dropped_outliers)

    print(f"\n... Removed {(size_before_handling_outliers-size_after_handling_outliers):,.0f} rows with outliers in the loaded_df_without_invalid_data.")
    print(f"\nThe total number of rows after dropping outliers is {(size_after_handling_outliers):,.0f}.\n")

    return df_dropped_outliers



def data_cleaning_pipeline(df):
    """
    Apply a pipeline of data cleaning functions: 
    drop_duplicate_data, drop_invalid_data, drop_outliers.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: Cleaned DataFrame after applying all functions.
    """
    print("\nStarting data cleaning pipeline...\n")
    
    # Step 1: Drop duplicate data
    df = drop_duplicate_data(df)
    
    # Step 2: Drop invalid data
    df = drop_invalid_data(df)
    
    # Step 3: Drop outliers
    df = drop_outliers(df)
    
    print("\n\nData cleaning pipeline completed.\n")
    
    return df