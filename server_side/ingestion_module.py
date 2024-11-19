#!/usr/bin/env python3

import os, re
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def load_json_data(data_dir: str) -> pd.DataFrame:
    """
    Load all JSON formatted files from the specified directory into a single DataFrame.

    This function iterates over all JSON files in the given directory, reads the content into 
    individual DataFrames, and concatenates them into a single DataFrame. The resulting DataFrame 
    includes specific columns with appropriate data types, along with a normalized date column.

    Parameters:
    ----------
    data_dir : str
        The directory path containing JSON files to be loaded.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing concatenated data from all JSON files with specified columns:
        ['country', 'date', 'invoice_id', 'customer_id', 'stream_id', 'times_viewed', 
         'price', 'year', 'month', 'day'].

    Raises:
    ------
    FileNotFoundError
        If the specified directory does not exist.
    ValueError
        If any JSON file cannot be parsed or if required columns are missing in the JSON data.
    """
    
    ## Initialize an empty list to hold the DataFrames
    dataframes = []
    
    ## Loop through each file in the directory
    for file_name in os.listdir(data_dir):
        
        ## Check if the file is a JSON file
        if file_name.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(data_dir, file_name)
            
            # Read the JSON data into a DataFrame and append to the list
            splited_df = pd.read_json(file_path)
            splited_df = pd.DataFrame(splited_df.values)
            dataframes.append(splited_df)
    
    ## Merge all DataFrames into a single DataFrame
    original_loaded_df = pd.concat(dataframes, ignore_index=True)

    ## Define the column names of the ingested dataframe
    columns = ['country', 'customer_id', 'invoice', 'price', 'stream_id', 'times_viewed', 'year', 'month', 'day']

    ## Adding the columns headers
    original_loaded_df.columns = columns
    
    ## Convert each column to its appropriate data types
    original_loaded_df['country']      = original_loaded_df['country'].astype('category')             # Convert to category for better memory usage
    original_loaded_df['customer_id']  = original_loaded_df['customer_id'].astype('Int64')            # Convert to nullable integer
    original_loaded_df['customer_id']  = original_loaded_df['customer_id'].astype('category')         # reConvert to category for better memory usage
    original_loaded_df['invoice']      = original_loaded_df['invoice'].astype('category')             # Convert to category for better memory usage
    original_loaded_df['price']        = pd.to_numeric(original_loaded_df['price'], errors='coerce')  # Convert to float, coerce errors to NaN
    original_loaded_df['stream_id']    = original_loaded_df['stream_id'].astype('category')           # Convert to category for better memory usage
    original_loaded_df['times_viewed'] = original_loaded_df['times_viewed'].astype(int)               # Convert to integer
    original_loaded_df['year']         = original_loaded_df['year'].astype(int)                       # Convert to integer
    original_loaded_df['month']        = original_loaded_df['month'].astype(int)                      # Convert to integer
    original_loaded_df['day']          = original_loaded_df['day'].astype(int)                        # Convert to integer
    
    ## Adding a date column to the dataframe & convert it to the appropriate data type
    years,months,days = original_loaded_df['year'].values,original_loaded_df['month'].values,original_loaded_df['day'].values 
    dates = [ "{}-{}-{}".format( years[i] , str(months[i]).zfill(2) , str(days[i]).zfill(2) ) for i in range(original_loaded_df.shape[0] ) ]
    original_loaded_df['date'] = pd.to_datetime(dates).normalize()
    original_loaded_df['date'] = pd.to_datetime(dates).date

    ## Rename the 'invoice' column to 'invoice_original'
    original_loaded_df.rename(columns={'invoice': 'invoice_original'}, inplace=True)
    ## Remove letters from invoice_original to improve matching and store the result in a new 'invoice' column
    original_loaded_df['invoice_id'] = [re.sub("\D+","",i) for i in original_loaded_df['invoice_original'].values]
    ## Convert to category for better memory usage
    original_loaded_df['invoice_id'] = original_loaded_df['invoice_id'].astype('category')          

    ## Change the column positions
    columns_reordered = ['country', 'date', 'invoice_id', 'customer_id',
                        'stream_id', 'times_viewed', 'price','year', 'month', 'day']
    original_loaded_df = original_loaded_df[columns_reordered]

    ## Sort the DataFrame by date to fit the logique of the next function (i.e time_series_df)
    original_loaded_df.sort_values(by=['country', 'date'],inplace=True)

    ## Reset the index
    original_loaded_df.reset_index(drop=True,inplace=True)
    
    return original_loaded_df