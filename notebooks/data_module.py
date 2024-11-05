#!/usr/bin/env python3

import os
import re
import pandas as pd

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
    original_loaded_df = pd.concat(dataframes, ignore_index=True)

    ## Define the column names of the ingested dataframe
    columns = ['country', 'customer_id', 'invoice', 'price', 'stream_id', 'times_viewed', 'year', 'month', 'day']

    ## Adding the columns headers
    original_loaded_df.columns = columns
    
    ## Convert each column to its appropriate data types
    original_loaded_df['country'] = original_loaded_df['country'].astype('category')           # Convert to category for better memory usage
    original_loaded_df['customer_id'] = original_loaded_df['customer_id'].astype('Int64')      # Convert to nullable integer
    original_loaded_df['customer_id'] = original_loaded_df['customer_id'].astype('category')   # reConvert to category for better memory usage
    original_loaded_df['invoice'] = original_loaded_df['invoice'].astype('category')           # Convert to category for better memory usage
    original_loaded_df['price'] = pd.to_numeric(original_loaded_df['price'], errors='coerce')  # Convert to float, coerce errors to NaN
    original_loaded_df['stream_id'] = original_loaded_df['stream_id'].astype('category')       # Convert to category for better memory usage
    original_loaded_df['times_viewed'] = original_loaded_df['times_viewed'].astype(int)        # Convert to integer
    original_loaded_df['year'] = original_loaded_df['year'].astype(int)                        # Convert to integer
    original_loaded_df['month'] = original_loaded_df['month'].astype(int)                      # Convert to integer
    original_loaded_df['day'] = original_loaded_df['day'].astype(int)                          # Convert to integer
    
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

def time_series_df(clean_df, country):
    """
    Create a time series DataFrame filtered by a specific country or not.

    Parameters:
    - df: the input DataFrame containing the necessary columns.
    - country: the country to filter on. Must be in the list of unique countries in the DataFrame.
    
    Returns:
    - A time series DataFrame grouped by date.
    """

    if country != None:
        if country not in clean_df['country'].unique().tolist():
            raise Exception("Country not found")
        clean_df_ = clean_df[clean_df['country'] == country]
    else:
        clean_df_ = clean_df

    ## Ensure dates are in datetime format
    clean_df_['date'] = pd.to_datetime(clean_df_['date'])
    
    ## Sort the DataFrame by date to fit the logic below
    clean_df_.sort_values(by='date',inplace=True)

    # Create a date range from the minimum date to the maximum date
    start_date = clean_df_['date'].min()
    end_date = clean_df_['date'].max()
    days = pd.date_range(start=start_date, end=end_date, freq='D')

    # Aggregate the dataframe by 'date'
    time_series_df = (clean_df_.groupby('date')
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

def anova_df(df, countries):
    """
    Create a DataFrame for ANOVA analysis with monthly revenue aggregated by country.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing time series data.
    countries (list or str): List of countries or a single country as a string.

    Returns:
    pd.DataFrame: A long-format DataFrame with year-month, country, and sum of revenue for ANOVA analysis.
    """

    
    # Initialize an empty DataFrame to store combined data for all countries
    anova_df = None

    for country in countries:
        # Create time series DataFrame for the specific country
        df_ = time_series_df(df, country=country)

        # Aggregate data by year-month and sum of revenue
        df_aggregated = (df_.groupby('year-month')
                         .agg(revenue=('revenue', 'sum'))
                         .round(2)
                         .reset_index()
                        )
        df_aggregated['country'] = country  # Add country column

        # Concatenate into the long DataFrame
        if anova_df is None:
            anova_df = df_aggregated.copy()
        else:
            anova_df = pd.concat([anova_df, df_aggregated], ignore_index=True)

    # Fill NaN values with 0 (in case there are any missing data)
    anova_df = anova_df.fillna(0)
    
    return anova_df