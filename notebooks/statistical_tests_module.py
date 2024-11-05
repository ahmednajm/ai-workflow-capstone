
import numpy as np
import pandas as pd

import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import t
from statsmodels.tsa.stattools import adfuller



def shapiro_test(dataframe, columns):
    """
    Check for normality of specified numerical columns in a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    columns (list): List of numerical column names to analyze.
    
    Returns:
    str: A formatted string indicating the normality of the data.
    """
    
    if isinstance(columns, str):
        columns = [columns]

    results = []  # List to store results

    for col in columns:
        # Shapiro-Wilk Test for normality
        stat, p_value = stats.shapiro(dataframe[col].dropna())
        
        # Check if p-value is significant
        if p_value > 0.05:
            result = f"The '{col}' data look normally distributed (p-value: {p_value:.3f})"
        else:
            result = f"The '{col}' data is not normally distributed (p-value: {p_value:.3f})"
            # Add message regarding the Central Limit Theorem
            if dataframe[col].dropna().shape[0] > 21:  # Check if sample size is greater than 21
                result += "\nBut since the sample size is greater than 20, the CLT helps to mitigate normality concerns."
        
        results.append(result)  # Store the result

    return "\n".join(results)  # Return all results as a single string



def levenes_test(dataframe, col_to_group_by, col_to_analyze):
    """
    Perform Levene's test on the specified group and value columns.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    col_to_group_by (str): The column name for grouping (e.g., 'country').
    col_to_analyze (str): The numerical column name to analyze (e.g., 'revenue').

    Returns:
    an interpretation message.
    """
    # Create a list of groups for the value column
    groups = [group[col_to_analyze].dropna() for name, group in dataframe.groupby(col_to_group_by)]
    
    # Perform Levene's test
    W_stat, p_value = stats.levene(*groups)
    
    # Interpretation
    if p_value > 0.05:
        print(f"\nThe variances of {col_to_analyze} across {col_to_group_by}s are approximately equal (p-value = {round(p_value,4)})\n")
    else : 
        print(f"\nThe variances of {col_to_analyze} across {col_to_group_by}s are significantly different (p-value = {round(p_value,4)})\n")



def welchs_anova_test(dataframe, col_to_group_by, col_to_analyze):
    """
    Perform Welch's ANOVA on the specified group and value columns.

     Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    col_to_group_by (str): The column name for grouping (e.g., 'country').
    col_to_analyze (str): The numerical column name to analyze (e.g., 'revenue').

    Returns:
    an interpretation message.
    """
    groups = [group[col_to_analyze].dropna() for name, group in dataframe.groupby(col_to_group_by)]
    
    # Perform Welch's ANOVA
    F_stat, p_value = stats.f_oneway(*groups)  # Bartlett's test can be a good preliminary check for homogeneity
    
    # Interpretation of results

    if p_value < 0.05:
        print(f"\nThere are significant differences in {col_to_analyze} among {col_to_group_by}.\n")
    else:
        print(f"\nThere are no significant differences in {col_to_analyze} among {col_to_group_by}.\n")
        


def Games_Howell_test(dataframe, col_to_group_by, col_to_analyze):
    # Step 1: Calculate group statistics
    group_stats = dataframe.groupby(col_to_group_by).agg(
        mean=(col_to_analyze, 'mean'),
        var=(col_to_analyze, 'var'),
        n=(col_to_analyze, 'size')
    ).reset_index()

    # Step 2: Prepare pairwise comparisons
    results = []
    for i in range(len(group_stats)):
        for j in range(i + 1, len(group_stats)):
            group1 = group_stats.iloc[i]
            group2 = group_stats.iloc[j]

            # Mean difference between the two groups
            mean_diff = group1['mean'] - group2['mean']

            # Step 3: Calculate the standard error
            se = ((group1['var'] / group1['n']) + (group2['var'] / group2['n'])) ** 0.5

            # Step 4: Calculate the t-statistic
            t_stat = abs(mean_diff) / se

            # Step 5: Degrees of freedom calculation for Welch-Satterthwaite equation
            df = ((group1['var'] / group1['n']) + (group2['var'] / group2['n'])) ** 2 / (
                (group1['var'] ** 2 / ((group1['n'] ** 2) * (group1['n'] - 1))) +
                (group2['var'] ** 2 / ((group2['n'] ** 2) * (group2['n'] - 1)))
            )

            # Step 6: Calculate the p-value
            p_value = 2 * (1 - t.cdf(t_stat, df))

            # Confidence interval calculation (95%)
            ci_low = mean_diff - t.ppf(0.975, df) * se
            ci_high = mean_diff + t.ppf(0.975, df) * se

            # Collect the result
            results.append({
                'Group 1': group1[col_to_group_by],
                'Group 2': group2[col_to_group_by],
                'Mean Difference': mean_diff,
                'p-value': p_value,
                'CI Lower Bound': ci_low,
                'CI Upper Bound': ci_high
            })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
            
    # Step 7: Interpret results
    #interpretations = []
    for index, row in results_df.iterrows():
        if row['p-value'] < 0.05:
            interpretation = (
                f"There is a significant difference in the means between {row['Group 1']} "
                f"and {row['Group 2']} (Mean Difference = {row['Mean Difference']:.2f}, "
                f"p-value = {row['p-value']:.3f})."
            )
        else:
            interpretation = (
                f"There is no significant difference in the means between {row['Group 1']} "
                f"and {row['Group 2']} (Mean Difference = {row['Mean Difference']:.2f}, "
                f"p-value = {row['p-value']:.3f})."
            )
        print(interpretation)  


def augmented_dickey_fuller_test(time_series):
    """
    Perform the Augmented Dickey-Fuller test to check for stationarity in a time series.
    
    Parameters:
    time_series (pd.Series): The time series data to test.
    
    Prints:
    ADF test statistics, p-value, critical values, and a conclusion on whether the series is stationary.
    """
    # Run the Augmented Dickey-Fuller test
    adf_result = adfuller(time_series)
    
    # Create a summary of the results
    adf_output = pd.Series(
        adf_result[:4], 
        index=['Test Statistic', 'p-value', 'Lags Used', 'Observations Used']
    )
    
    # Add critical values to the output
    for key, value in adf_result[4].items():
        adf_output[f'Critical Value ({key})'] = value
    
    # Print the test results
    print(adf_output)
    
    # Conclude based on the p-value
    if adf_result[1] <= 0.05:
        print("\nConclusion: The data is likely stationary.\n")
    else:
        print("\nConclusion: The data is non-stationary (We fail to reject the null hypothesis).\n")



def breusch_pagan_test(time_series):
    """
    Performs the Breusch-Pagan test for heteroscedasticity.

    Parameters:
    serie (array-like): The dependent variable 

    Returns:
    float: The p-value of the Breusch-Pagan test.
    """
    # Add a constant to the independent variables
    exog = sm.add_constant(np.arange(len(time_series)))
    
    # Perform the Breusch-Pagan test
    bp_test = sms.het_breuschpagan(time_series, exog)
    p_value = bp_test[1]  # The second value is the p-value

    # Check if heteroscedasticity is present based on the p-value
    if p_value < 0.05:
        print("\nHeteroscedasticity is present.\n")
    else:
        print("\nNo significant heteroscedasticity detected.\n")



def ljungbox_test(residuals):
    """
    Performs the Ljung-Box test for autocorrelation in residuals.

    Parameters:
    residuals : the residuals of a time series fitted model.
    """


    # Perform the Ljung-Box test
    ljungbox_results = acorr_ljungbox(residuals, lags=[14], return_df=True)
    
    # Extract the test statistic and p-value
    lb_stat = ljungbox_results['lb_stat'].iloc[0]
    lb_pvalue = ljungbox_results['lb_pvalue'].iloc[0]

    # Interpret the results
    print("\nLjung-Box test\n{}".format("-" * 22))
    print(f'Statistic={lb_stat:.4f}, p-value={lb_pvalue:.4f}')
    
    if lb_pvalue > 0.05:
        print("\nResiduals appear to be independent.\n")
    else:
        print("\nResiduals exhibit significant autocorrelation.\n")