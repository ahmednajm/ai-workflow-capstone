#!/usr/bin/env python3

import time, os
import pandas as pd
import scipy.stats as stats                                

import warnings
warnings.filterwarnings('ignore')

from server_side.ingestion_module import load_json_data


def _check_missing_values(df, printing):
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


def _drop_duplicate_data(df, printing):
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


def _drop_invalid_data(df, printing):
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
    

def _detect_outliers(df, column, z_threshold):
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


def _drop_outliers(df, printing):
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
        col_outliers = _detect_outliers(df, column=col, z_threshold=3)
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



def data_cleaning_pipeline(df, country, printing):
    """
    Applies data cleaning functions to remove duplicates, invalid data, and outliers.
    Parameters:
    - df (pd.DataFrame)    : The input DataFrame.
    - clean_data_dir (str) : Directory to save cleaned data.
    - country (str)        : Country filter for the data.
    - printing (bool)      : Flag to print results.
    Returns: pd.DataFrame  : Cleaned DataFrame.
    """
        
    # Filter by country if a specific country is provided
    if country != 'World':
        df = df[df['country'] == country]

    clean_df = _drop_duplicate_data(df, printing)
    clean_df = _drop_invalid_data(clean_df, printing)
    clean_df = _drop_outliers(clean_df, printing)
    clean_df = clean_df.reset_index(drop=True)
    
    return clean_df

def create_time_series(df):
    """
    Converts cleaned data into a time-series DataFrame with aggregated daily metrics.
    Parameters:
    - clean_data_dir (str) : Directory of cleaned data.
    - country (str)        : Country for which to create the time-series DataFrame.
    - ts_data_dir (str)    : Directory to store time series data.
    Returns: pd.DataFrame  : Time-series DataFrame.
    """
    
    ## Ensure dates are in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    ## Sort the DataFrame by date to fit the logic below
    df.sort_values(by='date',inplace=True)

    # Create a date range from the minimum date to the maximum date
    start_date = df['date'].min()
    end_date = df['date'].max()
    days = pd.date_range(start=start_date, end=end_date, freq='D')

    # Aggregate the dataframe by 'date'
    time_series_df = (df.groupby('date')
                      .agg(purchases=('date', 'size'),                   # Transaction count
                           unique_invoices=('invoice_id', 'nunique'),    # Unique invoice count
                           unique_streams=('stream_id', 'nunique'),      # Unique stream count
                           total_views=('times_viewed', 'sum'),          # Sum of views
                           revenue=('price', 'sum')                      # Sum of revenue
                          )
                      .reindex(days, fill_value=0)                       # Re-indexing to include all days
                      .reset_index()                                     # Reset index
                     )

    # Rename the date column
    time_series_df.rename(columns={'index': 'date'}, inplace=True)

    # Add a 'year-month' column
    time_series_df['year-month'] = time_series_df['date'].dt.to_period('M').astype(str)
    
    return time_series_df


def _get_dynamic_look_ahead_days(date):
    """
    Function to get dynamic look-ahead days based on month and year
    """
    month_days = (pd.Timestamp(year=date.year,
                               month=date.month,
                               day=1)
                  .days_in_month)  # Days in the current month
    
    # Limit to 30 days for months longer than 30 days
    return pd.Timedelta(days=min(month_days, 30))  


def _get_dynamic_look_back_days(date):
    """ 
    Function to get dynamic look-back days based on month and year
    """
    previous_month = date - pd.DateOffset(months=1)
    prev_month_days = (pd.Timestamp(year=previous_month.year,
                                    month=previous_month.month,
                                    day=1)
                       .days_in_month)
    
    return pd.Timedelta(days=min(prev_month_days, 30))


def _get_30_day_revenue(df, date):
    """
    Calculate 30-day target revenue with dynamic look-ahead
    """
    look_ahead = _get_dynamic_look_ahead_days(date)
    filter_cond = (df['date'] >= date) & (df['date'] < date + look_ahead)
    
    return df.loc[filter_cond, 'revenue'].sum()


def _get_previous_year_revenue(df, date):
    """
    Calculate revenue from the same period in the previous year with dynamic look-ahead
    """
    look_ahead = _get_dynamic_look_ahead_days(date - pd.DateOffset(years=1))
    start = date - pd.DateOffset(years=1)
    filter_cond = (df['date'] >= start) & (df['date'] < start + look_ahead)
    
    return df.loc[filter_cond, 'revenue'].sum()


def _get_recent_views(df, date):
    """
    Calculate average views over the last 30 days with dynamic look-back
    """
    look_back = _get_dynamic_look_back_days(date)
    filter_cond = (df['date'] >= date - look_back) & (df['date'] < date)
    recent_views = df.loc[filter_cond, 'total_views']
    return recent_views.mean() if not recent_views.empty else 0


def preprocessing(df, country, preprocessed_data_dir):
    """
    Creates features for modeling based on time-series data.
    Parameters:
    - data_dir (str)                      : Directory containing cleaned data.
    - country (str)                       : Country to engineer features for.
    - training (bool)                     : Flag for training to exclude recent data.
    - features_engineering_data_dir (str) : Directory to store time series data.

    Returns: pd.DataFrame : Feature matrix X, target vector y, and dates.
    """
        
    # Define the look-back periods (in days) for feature engineering
    previous_days = [7, 14, 28, 70]

    # Calculate rolling sums for revenue for each period and shift to align with target
    for num_days in previous_days:
        df[f'previous_{num_days}_days_revenue'] = df['revenue'].rolling(window=num_days, min_periods=1).sum().shift(1)
        
    # Calculate 30-day target revenue
    df['target_next_30_days_revenue'] = df['date'].apply(lambda date: _get_30_day_revenue(df, date))

    # Calculate revenue from the same period in the previous year
    df['previous_year_revenue'] = df['date'].apply(lambda date: _get_previous_year_revenue(df, date))

    # Calculate average views over the last 30 days
    df['previous_30_days_views'] = df['date'].apply(lambda date: _get_recent_views(df, date))

    # Drop 'total_views'
    del df['total_views']

    # Drop rows with missing values for required features and reset index
    df.dropna( subset = [f'previous_{num_days}_days_revenue' for num_days in previous_days] + \
                        ['target_next_30_days_revenue', 'previous_30_days_views'],
               inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Reorder the columns
    columns_order = ['date', 'previous_30_days_views', 'previous_year_revenue'] + \
                    [f'previous_{num_days}_days_revenue' for num_days in previous_days] + \
                    ['revenue', 'target_next_30_days_revenue']
    preprocessed_df = df[columns_order]

    # Save the time series DataFrame to a CSV file   
    preprocessed_data_filename = os.path.join(preprocessed_data_dir, f'{country}_preprocessed_data.csv')
    preprocessed_df.to_csv(preprocessed_data_filename)
    
    return preprocessed_df

    
########################################################################################################

if __name__ == "__main__":
    
    SOURCE_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'source_data', 'cs-train')
                
    PREPROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data')

########################################################################################################
    
    time_start = time.time()

    if not os.path.exists(PREPROCESSED_DATA_DIR):
        os.makedirs(PREPROCESSED_DATA_DIR)

    ## Loading source data
    print(f"\n... Loading source data")
    original_loaded_df = load_json_data(SOURCE_DATA_DIR) 
    print(f"Source data loaded successfully")
        
    #for country in list(original_loaded_df.country.unique()) + ['World']:
    for country in ['United Kingdom', 'World']:

        print(f"\n... Cleaning source data for {country}")
        clean_df = data_cleaning_pipeline(original_loaded_df, country, printing=False)
        print(f"Source data cleaned successfully for {country}")

        print(f"... Creating time series data for {country}")
        ts_df    = create_time_series(clean_df)
        print(f"Time series data created successfully for {country}")

        print(f"... Preprocessing data for {country}")
        preprocessing(ts_df, country, PREPROCESSED_DATA_DIR)
        print(f"Data preprocessed successfully for {country}")

    # Calculate runtime duration
    m, s = divmod(time.time() - time_start, 60)
    h, m = divmod(m, 60)
    time_start = "%02d:%02d" % (m, s)
    
    print('\nPreprocessing runtime', time_start)
