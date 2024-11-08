import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

from data_module import anova_df
from scipy.stats import spearmanr, pearsonr
from statsmodels.tsa.stattools import acf, pacf             


def top_revenue_by_country(df, top_n):
    """
    Calculate total revenue by country, determine percentage contributions, and visualize the top N revenue-generating countries.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing country and revenue data.
    top_n (int): The number of top revenue-generating countries to display.
    
    Returns:
    pd.DataFrame: A DataFrame containing the top N revenue-generating countries with their revenue and percentage.
    """
    # Group by country and calculate total revenue for each country
    revenue_country = (df[['country', 'price']]
                       .groupby('country', as_index=False)
                       .sum()
                       .rename(columns={'price': 'revenue'})
                       .sort_values(by='revenue', ascending=False)
                       .reset_index(drop=True)
                       )
    
    # Calculate percentage contribution for each country
    revenue_country['percentage'] = (revenue_country['revenue'] / revenue_country['revenue'].sum()) * 100
    
    # Format revenue for display on the bars
    revenue_country['revenue_label'] = revenue_country['revenue'].apply(lambda x: f"{x:,.2f}")
    
    # Create an interactive bar plot for the top N revenue-generating countries
    fig = px.bar(revenue_country.head(top_n),
                 x='revenue', 
                 y='country', 
                 text='revenue_label',  # Show revenue inside the bars
                 labels={'revenue': 'Log Revenue', 'country': ''}, 
                 title=f'Top {top_n} Revenue-Generating Countries',
                 orientation='h',  # horizontal bars
                 )
    
    # Sort the bars by descending revenue
    fig.update_yaxes(categoryorder='total ascending') 
    
    # Adjust layout
    fig.update_layout(xaxis_type="log",
                      width=1200, height=500)
                    
    # Show the plot
    fig.show()

    # Return only the top N revenue-generating countries with the required columns
    return revenue_country[['country', 'revenue', 'percentage']].head(top_n)



def violin_plot(df, countries):
    """
    Plot a violin plot of daily revenues for the top revenue-generating countries.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    countries (list or str): List of top revenue-generating countries to include in the plot or a single country as a string.
    """

    # Ensure countries is a list, even if a single string is provided
    if isinstance(countries, str):
        countries = [countries]

    from data_module import anova_df
    df_ = anova_df(df, countries)

    # Create a violin plot using Plotly Graph Objects
    fig = go.Figure()

    for country in countries:
        fig.add_trace(go.Violin(
                                 y=df_[df_['country'] == country]['revenue'],
                                 name=country,
                                 box_visible=True,
                                 points='all',
                                 # Dynamic color assignment
                                 line_color=px.colors.qualitative.Plotly[countries.index(country) % len(px.colors.qualitative.Plotly)],  
                                 text=df_[df_['country'] == country]['year-month'].astype(str),  # Add year-month for hover
                                 hoverinfo="y+text",  # Display revenue and year-month on hover
        ))

    # Update layout
    fig.update_layout(
        title='Violin Plot of Top Revenue-Generating Countries',
        xaxis_title='',
        yaxis_title='Revenue',
        showlegend=False,  # Remove the legend
        width=1000,  
        height=800,  
        margin=dict(l=40, r=40, t=40, b=120),  # Set margins with more space at the bottom
        yaxis=dict(showgrid=True),  # Add gridlines for y-axis
        xaxis_tickangle=-45,  # Rotate x-axis labels for readability
    )

    # Show the figure
    fig.show()


def pair_plot(df, target_column):
    """
    Creates scatter plots of numerical variables against the specified target column
    with dates in hover text.

    Parameters:
    - df (DataFrame): DataFrame containing engagement data.
    - target_column (str): Name of the target column in the DataFrame.
    """
    # Select numerical columns excluding the target column
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    numerical_columns.remove(target_column)

    # Create a subplot grid
    fig = make_subplots(rows=1, cols=len(numerical_columns))

    # Get hover data (assumed to be in the first column)
    hover_title = df.columns[0]  
    hover_data = df[hover_title]

    # Generate scatter plots
    for i, col in enumerate(numerical_columns):
        fig.add_trace(
            go.Scatter(
                x=df[col],
                y=df[target_column],
                mode='markers',
                name=col,
                marker=dict(opacity=0.7),
                hovertemplate=f'{hover_title}: %{{customdata}}<extra></extra>',
                customdata=hover_data
            ),
            row=1, col=i + 1
        )

    # Update layout
    fig.update_layout(
        title='Scatter Plots of REvenue vs User Engagement Variables ',
        yaxis_title=target_column.title(),
        height=400,
        showlegend=False
    )

    # Add subplot titles
    for i, col in enumerate(numerical_columns):
        fig.add_annotation(
            text=col,
            xref="paper",
            yref="paper",
            x=(i + 0.5) / len(numerical_columns),
            y=-0.1,
            showarrow=False,
            font=dict(size=12),
            xanchor='center',
            yanchor='top'
        )
        
    fig.show()



def correlation_matrix(dataframe, target_column='revenue', method='spearman'):
    # Select numerical columns
    numerical_columns = dataframe.select_dtypes(include=['number']).columns.tolist()
    
    # Calculate the correlation matrix based on the specified method
    corr_matrix = dataframe[numerical_columns].corr(method=method)

    # Create a heatmap for the correlation matrix
    fig, ax = plt.subplots(figsize=(7, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix,
                annot=True, fmt='.2f', ax=ax,
                linewidth=0.5, vmin=-1, vmax=1, mask=mask,
                cmap='BrBG', cbar=True)
    ax.set_title('The correlation heatmap')
    plt.show()

    # Assess the significance of the correlation between the target column and other numerical columns
    print("\n")
    for column in numerical_columns:
        if column != target_column:
            if method == 'spearman':
                # Calculate the Spearman correlation coefficient and p-value
                correlation_coefficient, p_value = spearmanr(dataframe[target_column], dataframe[column])
            elif method == 'pearson':
                # Calculate the Pearson correlation coefficient and p-value
                correlation_coefficient, p_value = pearsonr(dataframe[target_column], dataframe[column])
            else:
                raise ValueError("Unsupported method. Please use 'spearman' or 'pearson'.")

            # Interpret the results and print
            significance = "significant" if p_value < 0.05 else "no significant"
            print(f"Statisticaly, there is a {significance} correlation between '{target_column}' and '{column}'.")
    print("\n")
            
            

def plot_moving_average(df, column, window_size):
    """
    Create a plot comparing original revenue with a moving average.

    Parameters:
    df (DataFrame): DataFrame containing the revenue data with a 'revenue' column.
    window_size (int): The size of the window for the moving average calculation.

    Returns:
    None: Displays the plot.
    """
    # Calculate the moving average
    rolling_mean = df[column].rolling(window=window_size).mean()
    rolling_std = df[column].rolling(window=window_size).std()

    # Create a figure using Plotly
    fig = go.Figure()

    # Add original revenue line
    fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name='Original', line=dict(color='green')))

    # Add smoothed revenue line
    fig.add_trace(go.Scatter(x=df.index, y=rolling_mean , mode='lines', name=f'{window_size}-days Moving Average', line=dict(color='red')))

    # Add smoothed revenue line
    fig.add_trace(go.Scatter(x=df.index, y=rolling_std, mode='lines', name=f'{window_size}-days Moving STD', line=dict(color='blue')))

    # Update layout with larger dimensions
    fig.update_layout(title=f'{window_size}-days Smoothed {column}' ,
                      xaxis_title='Date',
                      yaxis_title=f'{column}',
                      legend_title='Legend',
                      width=1130,  
                      height=600)  

    # Show the plot
    fig.show()


def plot_moving_average(df, column, window_size):
    """
    Create a plot comparing original revenue with a moving average.

    Parameters:
    df (DataFrame): DataFrame containing the revenue data with a 'revenue' column.
    window_size (int): The size of the window for the moving average calculation.

    Returns:
    None: Displays the plot.
    """
    # Calculate the moving average
    df['smooth'] = df[column].rolling(window=window_size).mean()

    # Create a figure using Plotly
    fig = go.Figure()

    # Add original revenue line
    fig.add_trace(go.Scatter(x=df.index, y=df[df.columns[0]], mode='lines', name='Original', line=dict(color='green')))

    # Add smoothed revenue line
    fig.add_trace(go.Scatter(x=df.index, y=df[df.columns[1]], mode='lines', name=f'{window_size}-day Moving Average', line=dict(color='red')))

    # Update layout with larger dimensions
    fig.update_layout(title=f'original vs smoothed {df.columns[0]}' ,
                      xaxis_title='Date',
                      yaxis_title='Revenue',
                      legend_title='Legend',
                      width=1130,  
                      height=600)  

    # Show the plot
    fig.show()


def plot_time_series_decomposition(df, column_to_decompose, model, period):
    """
    Plots the time series decomposition of the specified column in the given DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the time series data.
    column_to_decompose (str): The name of the column to decompose.
    model (str): The type of decomposition model ('additive' or 'multiplicative').
    period (int): The number of observations per season.

    Returns:
    None
    """

    # Check if model is 'multiplicative' and if the data contains zero or negative values
    if model == 'multiplicative' and (df[column_to_decompose] <= 0).any():
        # Add a constant to avoid issues with zero or negative values
        df['revenue_transformed'] = df[column_to_decompose] + 1  
        column_to_decompose = 'revenue_transformed'

    # Decompose the time series
    decomposition = seasonal_decompose(df[column_to_decompose], model=model, period=period)

    # Create a figure with subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, vertical_spacing=0.05)

    # Plot Observed Revenue
    fig.add_trace(go.Scatter(x=decomposition.observed.index, 
                             y=decomposition.observed, mode='lines', 
                             line=dict(color='blue')), row=1, col=1)

    # Plot Trend Component
    fig.add_trace(go.Scatter(x=decomposition.trend.index, 
                             y=decomposition.trend, mode='lines', 
                             line=dict(color='orange')), row=2, col=1)

    # Plot Seasonal Component
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, 
                             y=decomposition.seasonal, mode='lines', 
                             line=dict(color='green')), row=3, col=1)

    # Plot Residual Component
    fig.add_trace(go.Scatter(x=decomposition.resid.index, 
                             y=decomposition.resid, mode='markers', 
                             line=dict(color='gold')), row=4, col=1)

    # Update layout with titles and horizontal line at 0
    fig.update_layout(title=f'{model} time series decomposition of {column_to_decompose}' , 
                      margin=dict(t=100, b=50, l=50, r=50),
                      width=1100,  
                      height=1500,
                      showlegend=False
                     )
    # Add horizontal line at y=0 for each subplot
    for i in range(1, 5):
        fig.add_shape(type='line', x0=decomposition.observed.index.min(), x1=decomposition.observed.index.max(), y0=0, y1=0,
                      line=dict(color='red', dash='dash'), row=i, col=1)

    # Set subplot titles with adjusted positions
    fig.add_annotation(text=column_to_decompose, x=0.5, y=1.02, xref='paper', yref='paper', showarrow=False, font=dict(size=16))
    fig.add_annotation(text='Trend Component', x=0.5, y=0.76, xref='paper', yref='paper', showarrow=False, font=dict(size=16))
    fig.add_annotation(text='Seasonal Component', x=0.5, y=0.49, xref='paper', yref='paper', showarrow=False, font=dict(size=16))
    fig.add_annotation(text='Residual Component', x=0.5, y=0.22, xref='paper', yref='paper', showarrow=False, font=dict(size=16))

    # Show the plot
    fig.show()


def plot_mstl_time_series_decomposition(df, column_name, periods):
    """
    Plots the time series decomposition using MSTL.
    
    Parameters:
    - ts_df: DataFrame containing the time series data
    - column_name: str, the name of the column to decompose
    - title: str, title of the plot
    - periods: tuple, the periods for MSTL decomposition
    """

    # Ensure periods is a tuple
    if isinstance(periods, (int, float)):
        periods = (periods,)
        
    # Perform MSTL decomposition
    model = MSTL(df[column_name], periods=periods)
    decomp = model.fit()

    # Create the subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, subplot_titles=(column_name, "Trend", "Seasonal", "Residuals"))

    # Add a horizontal line at y=0 across all rows
    fig.add_hline(y=0, line_dash="dash", row='all', col=1, line_color="red")

    # Plot the original time series
    fig.add_trace(go.Scatter(x=df.index, y=df[column_name], mode='lines', name=column_name), row=1, col=1)

    # Plot the decomposed components: Trend, Seasonal, and Residuals
    fig.add_trace(go.Scatter(x=df.index, y=decomp.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomp.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=decomp.resid, mode='markers', name='Residuals'), row=4, col=1)

    # Add a green horizontal line at y=0 for each subplot
    for i in range(1, 5):
        fig.add_shape(type='line',
                      x0=df.index.min(), x1=df.index.max(), 
                      y0=0, y1=0, 
                      line=dict(color='green', width=1.5),
                      row=i, col=1)

    # Update layout with the title and dimensions
    fig.update_layout(title=f'MSTL Decomposition of {column_name}',
                      height=800, width=1100,
                      showlegend=False,
                      xaxis=dict(title='Date'))

    # Show the plot
    fig.show()




def pacf_plot(serie, nlags):
    """
    Plots the Partial Autocorrelation Function (PACF) of the residuals and performs the Shapiro-Wilk test for normality.

    Parameters:
    - serie: A serie
    """

    # Calculate PACF and confidence intervals
    pacf_values, pacf_conf = sm.tsa.pacf(serie, alpha=0.05, nlags=nlags)

    # Calculate confidence intervals for PACF
    pacf_lower_y = pacf_conf[:, 0] - pacf_values
    pacf_upper_y = pacf_conf[:, 1] - pacf_values

    # Create a Plotly figure
    fig = go.Figure()

    # Plot PACF lines
    for x in range(len(pacf_values)):
        fig.add_trace(go.Scatter(x=[x, x], y=[0, pacf_values[x]], mode='lines', line=dict(color='#3f3f3f'), showlegend=False))

    # Markers for PACF values
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_values)), y=pacf_values, mode='markers', marker=dict(color='#ff7f0e', size=12), showlegend=False))

    # Add confidence interval lines
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_values)), y=pacf_upper_y, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_values)), y=pacf_lower_y, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False, 
                              fill='tonexty', fillcolor='rgba(255, 127, 14, 0.3)'))

    # Update layout
    fig.update_layout(title='Partial Autocorrelation Function (PACF) of',
                      xaxis_title='Lags', yaxis_title='', height=500, width=800)

    # Show the figure
    fig.show()


def acf_plot(serie, nlags):
    """
    Plots the Partial Autocorrelation Function (PACF) of the residuals and performs the Shapiro-Wilk test for normality.

    Parameters:
    - serie: A serie
    """

    # Calculate ACF and confidence intervals
    acf_values, acf_conf = sm.tsa.acf(serie, alpha=0.05, nlags=nlags)

    # Calculate confidence intervals for ACF
    acf_lower_y = acf_conf[:, 0] - acf_values
    acf_upper_y = acf_conf[:, 1] - acf_values

    # Create a Plotly figure
    fig = go.Figure()

    # Plot PACF lines
    for x in range(len(acf_values)):
        fig.add_trace(go.Scatter(x=[x, x], y=[0, acf_values[x]], mode='lines', line=dict(color='#3f3f3f'), showlegend=False))

    # Markers for PACF values
    fig.add_trace(go.Scatter(x=np.arange(len(acf_values)), y=acf_values, mode='markers', marker=dict(color='#ff7f0e', size=12), showlegend=False))

    # Add confidence interval lines
    fig.add_trace(go.Scatter(x=np.arange(len(acf_values)), y=acf_upper_y, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(len(acf_values)), y=acf_lower_y, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False, 
                              fill='tonexty', fillcolor='rgba(255, 127, 14, 0.3)'))

    # Update layout
    fig.update_layout(title='Autocorrelation Function (ACF)',
                      xaxis_title='Lags', yaxis_title='', height=500, width=800)

    # Show the figure
    fig.show()


def acf_and_pacf_plots(df, column_to_decompose, nlags):
    """
    Plots both the Autocorrelation (ACF) and Partial Autocorrelation (PACF) of the specified column.
    
    Parameters:
    df (DataFrame): The DataFrame containing the time series data.
    column_to_decompose (str): The name of the column to analyze.
    nlags (int): The number of lags to plot.
    
    Returns:
    None
    """
    # Calculate ACF and PACF
    acf_values, acf_conf = sm.tsa.acf(df[column_to_decompose].dropna(), alpha=0.05, nlags=nlags)
    pacf_values, pacf_conf = sm.tsa.pacf(df[column_to_decompose].dropna(), alpha=0.05, nlags=nlags)
    
    # Calculate confidence intervals for ACF and PACF
    acf_lower_y = acf_conf[:, 0] - acf_values
    acf_upper_y = acf_conf[:, 1] - acf_values
    pacf_lower_y = pacf_conf[:, 0] - pacf_values
    pacf_upper_y = pacf_conf[:, 1] - pacf_values

    # Create figure with 2 subplots (ACF and PACF)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.14,
                    subplot_titles=(f'Autocorrelation (ACF) of {column_to_decompose}', f'Partial Autocorrelation (PACF) of {column_to_decompose}')
                    )

    # Plot ACF
    for x in range(len(acf_values)):
        fig.add_scatter(x=(x, x), y=(0, acf_values[x]), mode='lines', line_color='#3f3f3f', row=1, col=1)
    fig.add_scatter(x=np.arange(len(acf_values)), y=acf_values, mode='markers',
                    marker_color='#1f77b4', marker_size=12, row=1, col=1)
    fig.add_scatter(x=np.arange(len(acf_values)), y=acf_upper_y, mode='lines', line_color='rgba(255,255,255,0)', row=1, col=1)
    fig.add_scatter(x=np.arange(len(acf_values)), y=acf_lower_y, mode='lines', fillcolor='rgba(32, 146, 230, 0.3)', 
                    fill='tonexty', line_color='rgba(255,255,255,0)', row=1, col=1)

    # Plot PACF
    for x in range(len(pacf_values)):
        fig.add_scatter(x=(x, x), y=(0, pacf_values[x]), mode='lines', line_color='#3f3f3f', row=2, col=1)
    fig.add_scatter(x=np.arange(len(pacf_values)), y=pacf_values, mode='markers',
                    marker_color='#ff7f0e', marker_size=12, row=2, col=1)
    fig.add_scatter(x=np.arange(len(pacf_values)), y=pacf_upper_y, mode='lines', line_color='rgba(255,255,255,0)', row=2, col=1)
    fig.add_scatter(x=np.arange(len(pacf_values)), y=pacf_lower_y, mode='lines', fillcolor='rgba(255, 127, 14, 0.3)', 
                    fill='tonexty', line_color='rgba(255,255,255,0)', row=2, col=1)

    # Update layout
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, nlags + 1])
    fig.update_yaxes(zerolinecolor='#000000')

    # Move subplot titles higher by updating annotations
    annotations = [dict(
        text='Autocorrelation (ACF)', x=0.1, y=1.02, xref='paper', yref='paper', showarrow=False, font=dict(size=15)),
        dict(text='Partial Autocorrelation (PACF)', x=0.13, y=0.45, xref='paper', yref='paper', showarrow=False, font=dict(size=15))
                  ]

    fig.update_layout(
        annotations=annotations,
        margin=dict(t=50, b=10, l=50, r=50), 
        width=1000, height=600
                    )

    # Show the figure
    fig.show()

def residuals_pacf_and_normality(model):
    """
    Plots the Partial Autocorrelation Function (PACF) of the residuals and performs the Shapiro-Wilk test for normality.

    Parameters:
    - model: A fitted SARIMAX model
    """
    # Extract the residuals from the model
    residuals = model.resid

    # Calculate PACF and confidence intervals
    pacf_values, pacf_conf = sm.tsa.pacf(residuals[model.loglikelihood_burn:], alpha=0.05, nlags=14)

    # Calculate confidence intervals for PACF
    pacf_lower_y = pacf_conf[:, 0] - pacf_values
    pacf_upper_y = pacf_conf[:, 1] - pacf_values

    # Create a Plotly figure
    fig = go.Figure()

    # Plot PACF lines
    for x in range(len(pacf_values)):
        fig.add_trace(go.Scatter(x=[x, x], y=[0, pacf_values[x]], mode='lines', line=dict(color='#3f3f3f'), showlegend=False))

    # Markers for PACF values
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_values)), y=pacf_values, mode='markers', marker=dict(color='#ff7f0e', size=12), showlegend=False))

    # Add confidence interval lines
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_values)), y=pacf_upper_y, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False))
    fig.add_trace(go.Scatter(x=np.arange(len(pacf_values)), y=pacf_lower_y, mode='lines', line=dict(color='rgba(255,255,255,0)'), showlegend=False, 
                              fill='tonexty', fillcolor='rgba(255, 127, 14, 0.3)'))

    # Update layout
    fig.update_layout(title='Partial Autocorrelation Function (PACF) of Residuals (after burn-in)', 
                      xaxis_title='Lags', yaxis_title='', height=500, width=800)

    # Show the figure
    fig.show()

    # Perform the Shapiro-Wilk test for normality on the residuals after burn-in
    stat, p_value = shapiro(residuals[model.loglikelihood_burn:])

    # Interpret the result of the Shapiro-Wilk test
    print("\nShapiro-Wilk test\n{}".format("-"*22))
    print(f'Statistic={stat:.4f}, p-value={p_value:.4f}')
    if p_value > 0.05:
        print("\nResiduals appear to be normally distributed. The model is adequate\n")
    else:
        print("\nResiduals do not appear to be normally distributed.The model may be inadequate\n")


def plot_sarimax_diagnostics(model):
    """
    Plots diagnostics for a SARIMAX model.
    
    Parameters:
    - model: A fitted SARIMAX model
    """
    # Extract the residuals from the model
    residuals = model.resid

    # Create subplots
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.1,
                        subplot_titles=['Residuals', 'Histogram of Residuals', 'Q-Q Plot of Residuals', 'ACF of Residuals'])

    # Residuals plot
    fig.add_trace(
        go.Scatter(x=np.arange(len(residuals)), y=residuals, mode='lines', name='Residuals'),
        row=1, col=1
    )

    # Histogram of Residuals with KDE and N(0,1)
    x_vals = np.linspace(np.min(residuals), np.max(residuals), 100)
    kde = norm.pdf(x_vals, loc=np.mean(residuals), scale=np.std(residuals))  # KDE approximation

    # Add histogram
    fig.add_trace(
        go.Histogram(x=residuals, nbinsx=30, name='Histogram', histnorm='probability density'),
        row=1, col=2
    )
    
    # Add KDE line
    fig.add_trace(
        go.Scatter(x=x_vals, y=kde, mode='lines', name='KDE', line=dict(color='blue')),
        row=1, col=2
    )

    # Add standard normal distribution N(0,1)
    std_normal = norm.pdf(x_vals, loc=0, scale=1)
    fig.add_trace(
        go.Scatter(x=x_vals, y=std_normal, mode='lines', name='N(0,1)', line=dict(color='green', dash='dash')),
        row=1, col=2
    )

    # Q-Q Plot
    qq_theoretical = np.linspace(np.min(residuals), np.max(residuals), len(residuals))
    qq_sample = np.sort(residuals)
    fig.add_trace(
        go.Scatter(x=qq_theoretical, y=qq_sample, mode='markers', name='Q-Q Plot'),
        row=2, col=1
    )

    # Add diagonal line to Q-Q Plot
    fig.add_trace(
        go.Scatter(x=qq_theoretical, y=qq_theoretical, mode='lines', name='45-degree Line', line=dict(color='green', dash='dash')),
        row=2, col=1
    )

    # ACF of Residuals
    acf_values, acf_conf = sm.tsa.acf(residuals, alpha=0.05, nlags=14)
    
    # Calculate confidence intervals for ACF
    acf_lower_y = acf_conf[:, 0] - acf_values
    acf_upper_y = acf_conf[:, 1] - acf_values
    
    # Plot ACF with confidence intervals
    for x in range(len(acf_values)):
        # Vertical line for ACF values
        fig.add_trace(
            go.Scatter(x=[x, x], y=[0, acf_values[x]], mode='lines', line_color='#3f3f3f'),
            row=2, col=2
        )  
    
    # Markers for ACF values
    fig.add_trace(
        go.Scatter(x=np.arange(len(acf_values)), y=acf_values, mode='markers',
                    marker_color='#1f77b4', marker_size=12, name='ACF'),
        row=2, col=2
    )

    # Fill between the confidence intervals
    fig.add_trace(
        go.Scatter(x=np.arange(len(acf_values)), y=acf_upper_y, mode='lines', line_color='rgba(255,255,255,0)', name='Upper CI'),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=np.arange(len(acf_values)), y=acf_lower_y, fill='tonexty', 
                    fillcolor='rgba(32, 146, 230, 0.3)', line_color='rgba(255,255,255,0)', name='Lower CI'),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title='SARIMAX Model Diagnostics',
        height=700,
        width=1100,
        showlegend=False
    )

    # Show the plot
    fig.show()

def plot_autocorrelation(df, column_name, nlags):
    """
    Plots the autocorrelation of a time series with confidence intervals.
    
    Parameters:
    - ts_df: DataFrame containing the time series data
    - column_name: str, the name of the column to compute autocorrelation for
    - nlags: int, number of lags to compute autocorrelation (default is 600)
    - conf_level: float, confidence level for the intervals (default is 3.291 for 99.9%)
    - title: str, title of the plot
    """
    # Compute autocorrelation values
    autocorr_values = acf(df[column_name], nlags=nlags)
    lags = list(range(len(autocorr_values)))

    # Calculate the confidence interval
    n = len(df[column_name])
    conf_interval = 3.291 / np.sqrt(n)  # alpha equal to 0,001 and not the classic 0.05

    # Create the plot
    fig = go.Figure()

    # Plot the autocorrelation values as bars
    fig.add_trace(go.Bar(x=lags, y=autocorr_values, marker_color='blue', name='Autocorrelation'))

    # Add confidence interval lines
    fig.add_trace(go.Scatter(x=lags, y=[conf_interval] * len(lags), mode='lines', 
                             line=dict(color='red', dash='dash'), name='Upper Confidence Interval'))
    fig.add_trace(go.Scatter(x=lags, y=[-conf_interval] * len(lags), mode='lines', 
                             line=dict(color='red', dash='dash'), name='Lower Confidence Interval'))

    # Update layout
    fig.update_layout(
        title=f'Autocorrelation Plot of {column_name}',
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        yaxis_range=[-1, 1],
        width=1100,
        height=600,
        showlegend=False
    )

    # Show the plot
    fig.show()


def plot_predictions(x, y, x_pred, y_pred):
    """
    Plot actual and predicted time series data using Plotly Express.

    Parameters:
    x (array-like): Dates or time steps for the true values.
    y (array-like): Actual values corresponding to x.
    x_pred (array-like): Dates or time steps for the predicted values.
    y_pred (array-like): Predicted values corresponding to x_pred.
    """
    # Create DataFrames for true and predicted values
    df_true = pd.DataFrame({"Time": x, "Revenue": y, "Type": "True"})
    df_pred = pd.DataFrame({"Time": x_pred, "Revenue": y_pred, "Type": "Predicted"})
    
    # Concatenate true and predicted DataFrames
    df_combined = pd.concat([df_true, df_pred])
    
    # Generate the plot
    fig = px.line(df_combined, x="Time", y="Revenue", color="Type",
                  title="True vs Predicted Revenue Over Time")
    
    # Customize layout (remove legend title)
    fig.update_layout(width=1100, height=600,
                      legend_title_text=None,  # Removes legend title
                      legend=dict(x=0.9, y=1),
                      xaxis_title="",
                      yaxis_title="")
    
    # Display the plot
    fig.show()


def plot_models_comparison(models, rmse_values, runtime_values):
    """
    Plots a grouped bar chart to compare RMSE and runtime for different models.

    Parameters:
    - models (list): A list of model names.
    - rmse_values (list): A list of RMSE values corresponding to the models.
    - runtime_values (list): A list of runtime values corresponding to the models.
    - title (str): The title of the plot.
    """
    # Create a comparison dictionary
    comparison = {
        'Model': models,
        'RMSE': rmse_values,
        'Runtime': [runtime_value*10 for runtime_value in runtime_values]
    }

    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison)

    # Melt the DataFrame to get a long format for Plotly
    comparison_df_melted = comparison_df.melt(id_vars='Model', value_vars=['RMSE', 'Runtime'], 
                                               var_name='Metric', value_name='Value')

    # Create a grouped bar chart
    fig = px.bar(comparison_df_melted, x='Model', y='Value', color='Metric', barmode='group',
                 title='Comparison of RMSE and Runtime for Different Models')

    # Customize layout
    fig.update_layout(height=600, width=800, showlegend=True)
    fig.update_xaxes(title='', showticklabels=True)
    fig.update_yaxes(title='Value', showticklabels=True)

    # Show the figure
    fig.show()