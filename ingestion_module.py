#!/usr/bin/env python3

"""
A collection of functions for the AVAIL case study solution
"""

import os
import re
import numpy as np
import pandas as pd


## Laod all json formatted files into a dataframe
def load_json_data(data_dir) :
    """
    laod all json formatted files into a dataframe
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
    df = pd.concat(dataframes, ignore_index=True)

    ## Define the column names of the ingested dataframe
    columns = ['country', 'customer_id', 'invoice', 'price', 'stream_id', 'times_viewed', 'year', 'month', 'day']

    ## Adding the clumns headers
    df.columns = columns
    
    ## Convert each column to its appropriate data types
    df['country'] = df['country'].astype('category')           # Convert to category for better memory usage
    df['customer_id'] = df['customer_id'].astype('Int64')      # Convert to nullable integer
    df['customer_id'] = df['customer_id'].astype('category')   # reConvert to category for better memory usage
    df['invoice'] = df['invoice'].astype('category')           # Convert to category for better memory usage
    df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to float, coerce errors to NaN
    df['stream_id'] = df['stream_id'].astype('category')       # Convert to category for better memory usage
    df['times_viewed'] = df['times_viewed'].astype(int)        # Convert to integer
    df['year'] = df['year'].astype(int)                        # Convert to integer
    df['month'] = df['month'].astype(int)                      # Convert to integer
    df['day'] = df['day'].astype(int)                          # Convert to integer
    
    ## Adding a date column to the dataframe & convert it to the appropriate data type
    years,months,days = df['year'].values,df['month'].values,df['day'].values 
    dates = [ "{}-{}-{}".format( years[i] , str(months[i]).zfill(2) , str(days[i]).zfill(2) ) for i in range(df.shape[0] ) ]
    df['date'] = pd.to_datetime(dates).normalize()
    df['date'] = pd.to_datetime(dates).date

    
    ## Rename the 'invoice' column to 'invoice_original'
    df.rename(columns={'invoice': 'invoice_original'}, inplace=True)
    ## Remove letters from invoice_original to improve matching and store the result in a new 'invoice' column
    df['invoice_id'] = [re.sub("\D+","",i) for i in df['invoice_original'].values]
    ## Convert to category for better memory usage
    df['invoice_id'] = df['invoice_id'].astype('category')          

    ## Change the column positions
    columns_reordered = ['country', 'date', 'invoice_id', 'customer_id',
                        'stream_id', 'times_viewed', 'price', 'invoice_original', 'year', 'month', 'day']
    df = df[columns_reordered]

    
    ## Sort the DataFrame by date to fit the logique of the next function (i.e time_series_df)
    df.sort_values(by=['country', 'date'],inplace=True)

    ## Reset the index
    df.reset_index(drop=True,inplace=True)
    
    return df
    
    
## Create a time series DataFrame filtered by a specific country or not.
def time_series_df(df_, country=None):
    """
    Create a time series DataFrame filtered by a specific country or not.

    Parameters:
    - df_: the input DataFrame containing the necessary columns.
    - country: the country to filter on. Must be in the list of unique countries in the DataFrame.
    
    Returns:
    - A time series DataFrame grouped by date.
    """
    
    if country:
        if country not in df_['country'].unique().tolist():
            raise Exception("Country not found")
        df = df_[df_['country'] == country]
    else:
        df = df_

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
                      .reindex(days, fill_value=0)                       # Reindexing to include all days
                      .reset_index()                                     # Reset index
                     )

    # Rename the date column
    time_series_df.rename(columns={'index': 'date'}, inplace=True)

    # Add a 'year-month' column
    time_series_df['year-month'] = time_series_df['date'].dt.to_period('M').astype(str)

    return time_series_df



if __name__ == "__main__":

    # Specify the directory
    data_dir = 'cs-train'  
    dataframe = load_json_data(data_dir)
    
    # Initializing the output print
    print('fetching data ...')
    
    # Print the first few rows of the DataFrame
    print(dataframe.head())  