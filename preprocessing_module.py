
import time, os
from collections import defaultdict

import pandas as pd
import scipy.stats as stats                                

import warnings
warnings.filterwarnings('ignore')

from ingestion_module import load_json_data


def check_missing_values(df, printing):
    """
    Check for missing values in the dataframe and summarize them.
    Parameters:
    - df (pd.DataFrame) : The input dataframe.
    - printing (bool)   : Flag to print results.
    Returns: pd.Series  : A summary of columns with missing values.
    """
        
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        if printing:
            print("\nThere are some missing values in the dataframe\n")
            print("\nMissing values Summary\n{}".format("-"*25))
    else:
        if printing:
            print("\nThere are no missing values in the dataframe")
    
    # Filter columns with missing values
    missing_data = missing_values[missing_values > 0]
    for column, missing_count in missing_data.items():
            if printing:
                print(f"\nThere are {missing_count:,.0f} missing values in the {column} column.\n")


def drop_duplicate_data(df, printing):
    """
    Removes duplicate rows from the DataFrame and prints a summary of the operation.
    Parameters:
    - df (pd.DataFrame)   : The input DataFrame.
    - printing (bool)     : Flag to print results.
    Returns: pd.DataFrame : DataFrame with duplicates removed.
    """
    if printing:
        print("\nDuplicate rows Summary\n{}".format("-"*22))

    size_before = len(df)
    df_dropped_duplicate_data = df.drop_duplicates(keep="first")
    size_after = len(df_dropped_duplicate_data)
    if printing:
        print(f"\nThe total number of rows before dropping duplicates is {size_before:,.0f}")
        print(f"\n... Removed {size_before - size_after:,.0f} duplicate rows.")
        print(f"\nThe total number of rows after dropping duplicates is {size_after:,.0f}\n")
    return df_dropped_duplicate_data


def drop_invalid_data(df, printing):
    """
    Exclude rows with invalid data based on price and times viewed.
    Parameters:
    - df (pd.DataFrame)   : The input dataframe.
    - printing (bool)     : Flag to print results.
    Returns: pd.DataFrame : Dataframe with invalid rows excluded.
    """

    # Select numerical columns
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    size_before_invalid_data = len(df)

    if printing:
        print(f'\nThe numerical columns are : {" and ".join(numerical_cols)}\n')
        print(f"The total number of rows before dropping any invalid data is {size_before_invalid_data:,.0f}.\n")
        print("\nData Quality Summary\n{}".format("-" * 26))

    # Start with the full dataframe and iteratively filter out invalid rows
    df_dropped_invalid_data = df.copy()

    # Print the number of invalid data points for each numerical column
    for col in numerical_cols:
        invalid_data_count = (df[col] <= 0).sum()
        if printing:
            print(f"\nThe total number of invalid data points in {col} is {(invalid_data_count):,.0f}")

        # Exclude rows with invalid data in the current column
        df_dropped_invalid_data = df_dropped_invalid_data[df_dropped_invalid_data[col] > 0]

    size_after_invalid_data = len(df_dropped_invalid_data)
    if printing:
        print(f"\n... Removed {(size_before_invalid_data - size_after_invalid_data):,.0f} rows with invalid data.")
        print(f"\nThe total number of rows after dropping all invalid data is {size_after_invalid_data:,.0f}.\n")

    return df_dropped_invalid_data
    

def detect_outliers(df, column, z_threshold):
    """
    Detects outliers in a specified column based on Z-score and IQR methods.
    Parameters:
    - df (pd.DataFrame)   : DataFrame to detect outliers in.
    - column (str)        : Column to check for outliers.
    - z_threshold (float) : Z-score threshold for identifying outliers.
    Returns: pd.Series    : Boolean mask indicating outliers.
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
                
        # Combine outlier masks for the current country
        group_mask = z_score_outliers | iqr_outliers 
        
        # Update the overall outlier mask
        outlier_mask.update(group_mask)

    return outlier_mask



def drop_outliers(df, printing):
    """
    Drops rows with outliers in numerical columns.
    Parameters:
    - df (pd.DataFrame) : DataFrame to remove outliers from.
    - printing (bool)   : Flag to print results.
    Returns             : pd.DataFrame: DataFrame with outliers removed.
    """
    
    combined_outliers = pd.Series(False, index=df.index)
    size_before_handling_outliers = len(df)
    
    ## Select numerical columns
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    # Define columns to exclude
    exclude_cols = ['year', 'month', 'day']
    # Filter out the excluded columns
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    if printing:
        print("\nOutliers Summary\n{}".format("-" * 26))
        print(f'\nThe numerical columns are : {" and ".join(numerical_cols)}\n')

    for col in numerical_cols:
        col_outliers = detect_outliers(df, column=col, z_threshold=3)
        df_with_col_outliers = df[col_outliers]
        if printing:
            print(f"\nThere are {(len(df_with_col_outliers)):,.0f} rows identified with outliers in {col} data")
        
        # Update combined outliers
        combined_outliers |= col_outliers
        
    if printing:
        print(f"\nOverall, there are {(len(df[combined_outliers])):,.0f} rows identified with outliers\n")
        print(f"\nThe total number of rows before dropping outliers is {(size_before_handling_outliers):,.0f}.")

    df_dropped_outliers = df[~combined_outliers]
    
    size_after_handling_outliers  = len(df_dropped_outliers)

    if printing:
        print(f"\n... Removed {(size_before_handling_outliers-size_after_handling_outliers):,.0f} rows with outliers in the loaded_df_without_invalid_data.")
        print(f"\nThe total number of rows after dropping outliers is {(size_after_handling_outliers):,.0f}.\n")

    return df_dropped_outliers


def data_cleaning_pipeline(original_loaded_df, clean_data_dir ,country, printing):
    """
    Applies data cleaning functions to remove duplicates, invalid data, and outliers.
    Parameters:
    - df (pd.DataFrame)    : The input DataFrame.
    - clean_data_dir (str) : Directory to save cleaned data.
    - country (str)        : Country filter for the data.
    - printing (bool)      : Flag to print results.
    Returns: pd.DataFrame  : Cleaned DataFrame.
    """
    
    print(f"\n... Starting data cleaning pipeline for {country}")
    
    # Filter by country if a specific country is provided
    if country != 'all_countries':
        original_loaded_df = original_loaded_df[original_loaded_df['country'] == country]

    # Step 1: Drop duplicate data
    clean_df = drop_duplicate_data(original_loaded_df, printing)

    # Step 2: Drop invalid data
    clean_df = drop_invalid_data(clean_df, printing)

    # Step 3: Drop outliers
    clean_df = drop_outliers(clean_df, printing)

    # Save the cleaned DataFrame to a CSV file   
    output_filename = os.path.join(clean_data_dir, f'{country}_cleaned_data.csv')
    clean_df.to_csv(output_filename, index=False)
    print(f"cleaned data saved to {output_filename}")

    print(f"Data cleaning pipeline completed for {country}.\n")
    
    return clean_df


def time_series_df(clean_data_dir, country):
    """
    Converts cleaned data into a time-series DataFrame with aggregated daily metrics.
    Parameters:
    - clean_data_dir (str) : Directory of cleaned data.
    - country (str)        : Country for which to create the time-series DataFrame.
    Returns: pd.DataFrame  : Time-series DataFrame.
    """
        
    try:
        clean_df = pd.read_csv(f"{clean_data_dir}/{country}_cleaned_data.csv")
    except FileNotFoundError:
        raise Exception("Country not found")

    ## Ensure dates are in datetime format
    clean_df['date'] = pd.to_datetime(clean_df['date'])
    
    ## Sort the DataFrame by date to fit the logic below
    clean_df.sort_values(by='date',inplace=True)

    # Create a date range from the minimum date to the maximum date
    start_date = clean_df['date'].min()
    end_date = clean_df['date'].max()
    days = pd.date_range(start=start_date, end=end_date, freq='D')

    # Aggregate the dataframe by 'date'
    time_series_df = (clean_df.groupby('date')
                      .agg(purchases=('date', 'size'),                   # Transaction count
                           unique_invoices=('invoice_id', 'nunique'),    # Unique invoice count
                           unique_streams=('stream_id', 'nunique'),      # Unique stream count
                           total_views=('times_viewed', 'sum'),          # Sum of views
                           revenue=('price', 'sum')                      # Sum of revenue
                          )
                      .reindex(days, fill_value=0)                       # Reindexing to include all days
                      .reset_index()                                     # Reset index
                     )

    # Rename the date column
    time_series_df.rename(columns={'index': 'date'}, inplace=True)

    # Add a 'year-month' column
    time_series_df['year-month'] = time_series_df['date'].dt.to_period('M').astype(str)

    return time_series_df


def engineer_features(data_dir, country, training):
    """
    Creates features for modeling based on time-series data.
    Parameters:
    - data_dir (str)      : Directory containing cleaned data.
    - country (str)       : Country to engineer features for.
    - training (bool)     : Flag for training to exclude recent data.
    Returns: pd.DataFrame : Feature matrix X, target vector y, and dates.
    """
    engineer_features_time_start = time.time()

    ts_df = time_series_df(data_dir, country=country)
    ts_df = ts_df[['date', 'revenue', 'purchases', 'total_views']]
    ts_df['date'] = pd.to_datetime(ts_df['date'], errors='coerce')

    # Initialize dictionaries to store features and target values
    eng_features = defaultdict(list)
    y = []

    # Define the look-back periods (in days) for feature engineering
    previous_days = [7, 14, 28, 70]

    # Calculate rolling sums for revenue for each period and shift to align with target
    for num_days in previous_days:
        ts_df[f'revenue_{num_days}'] = ts_df['revenue'].rolling(window=num_days, min_periods=1).sum().shift(1)

    # Iterate over each row in the DataFrame
    for idx, row in ts_df.iterrows():
        current_date = row['date']

        # Append engineered features
        for num_days in previous_days:
            eng_features[f'previous_{num_days}'].append(row[f'revenue_{num_days}'])

        # Target: Sum revenue for the next 30 days
        target_sum = ts_df[(ts_df['date'] >= current_date) & (ts_df['date'] < current_date + pd.Timedelta(days=30))]['revenue'].sum()
        y.append(target_sum)
        
        # Previous year revenue for trend analysis
        prev_year_start = current_date - pd.DateOffset(years=1)
        prev_year_revenue = ts_df[(ts_df['date'] >= prev_year_start) & (ts_df['date'] < prev_year_start + pd.DateOffset(days=30))]['revenue'].sum()
        eng_features['previous_year'].append(prev_year_revenue)
        
        # Non-revenue features: Average invoices and views over the last 30 days
        recent_data = ts_df[(ts_df['date'] >= current_date - pd.Timedelta(days=30)) & (ts_df['date'] < current_date)]
        eng_features['recent_views'].append(recent_data['total_views'].mean() if not recent_data.empty else 0)
    
    # Convert the features dictionary to a DataFrame
    X = pd.DataFrame(eng_features)
    y = pd.Series(y, name='target')
    dates = ts_df['date']

    # Remove rows with all zeros (in cases where no data exists for look-back periods)
    X = X[(X != 0).any(axis=1)]
    y = y[X.index]
    dates = dates[X.index]

    # If training, exclude the last 30 days to ensure target reliability
    if training:
        X = X.iloc[:-30]
        y = y.iloc[:-30]
        dates = dates.iloc[:-30]
        
    # Reset index for neatness
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    dates.reset_index(drop=True, inplace=True)

    # Calculate runtime duration
    m, s = divmod(time.time() - engineer_features_time_start, 60)
    h, m = divmod(m, 60)
    engineer_features_runtime = "%02d:%02d" % (m, s)
    
    return X, y, dates


if __name__ == "__main__":
    
    #SOURCE_DATA_DIR = 'data/source_data/cs-train'
    SOURCE_DATA_DIR = os.path.join(os.path.dirname(__file__), '.', 'data', 'source_data', 'cs-train')
    CLEAN_DATA_DIR = os.path.join(os.path.dirname(__file__), '.', 'data', 'clean_data')

    ## Loading data
    print(f"\n... Loading source data")
    original_loaded_df = load_json_data(SOURCE_DATA_DIR) 
    print(f"\nSource data loaded successfully \n")
    
    ## Cleaning data and saving it as csv for each country in original_loaded_df
    for country in original_loaded_df.country.unique() :
        data_cleaning_pipeline(original_loaded_df, CLEAN_DATA_DIR, country, printing=False)

    ## Cleaning data and saving it as csv for all_countries in original_loaded_df
    data_cleaning_pipeline(original_loaded_df, CLEAN_DATA_DIR, 'all_countries', printing=True)
    

    
