
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def distr_plot(df_column, year):
    '''
    Distribution of the Realized Market Return

    Parameters:
    df_column (Pandas.Dataframe): dataframe column with index return
    year (String): Estimation year
   
    Returns:
    Histogram of the return distribution.
    '''
    plt.figure(figsize=(8, 6))
    # Plot distribution of 'ln_expected_change'
    #plt.subplot(1, 2, 1)
    sns.histplot(df_column, bins=15, kde=True, color='blue')
    plt.title(f'Distribution of the Realized Market Return, {year}')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    # Adjust layout for better readability
    plt.tight_layout()
    # Show the plots
    plt.show()


def add_sanction_line(sanctions, df):
    '''
    Adding highlighted regions
    
    Parameters:
    sanctions (Pandas.core.indexes.datetimes.DatetimeIndex): list of sanctions in yyyy-mm-dd format
    df (Pandas.Dataframe): estimated dataframe
   
    Returns:
    Adding hilighted vertical area on a plot.
    '''
    for date in sanctions:
        date = pd.Timestamp(date)  # Ensure it's a Timestamp
        for offset in range(3):  # Check date, date+1, date+2
            shifted_date = date + pd.Timedelta(days=offset)
            if shifted_date in df.index:
                idx = df.index.get_loc(shifted_date)  # Get index
                start_idx = max(0, idx - 5)
                end_idx = min(len(df) - 1, idx + 5)

                start_date = df.index[start_idx]
                end_date = df.index[end_idx]

                plt.axvspan(start_date, end_date, color="gray", alpha=0.3)
                plt.axvline(x=shifted_date, color='red', linewidth=2.5, linestyle='--')
                break  # Stop checking once a valid date is found

            '''# Annotate specific date
            value_at_date = df.loc[date, "cond_cov"]
            plt.text(date, value_at_date, f"{date.strftime('%Y-%m-%d')}", 
                    fontsize=10, color="red", ha="center", va="bottom", fontweight="bold")'''


def cond_volatility_plot(condit_volatility, year, sanctions):
    '''
    Conditional volatilities Plot setup

    Parameters:
    condit_volatility (Pandas.Dataframe): dataframe with two columns (condtional volatilities)
    year (String): Estimation year
    sanctions (Pandas.core.indexes.datetimes.DatetimeIndex): list of sanctions in yyyy-mm-dd format

    Returns:
    Plot of two conditional volatilities with hilighted sanctions dates.
    '''
    plt.figure(figsize=(8, 5))
    for col in condit_volatility.columns:
        plt.plot(condit_volatility.index, condit_volatility[col], label=col)
    plt.title(f'Conditional Volatilities, {year}', fontsize=14)
    plt.xlabel("", fontsize=10) #year
    plt.ylabel('Volatility, b.p.', fontsize=10)
    plt.grid(True)
    if str(year) in ['2022', '2023']:
        add_sanction_line(sanctions, condit_volatility)
        # Add custom red dotted line (not actually plotted)
        custom_line, = plt.plot([], [], 'r--', label='sanctions')  # 'r--' = red dotted line
    plt.legend(loc='upper right', fontsize=14, framealpha=0.5) #upper lower left
    # Get the current axis
    ax = plt.gca()
    # Set x-axis ticks to three times per month at valid trading days
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # Format labels as '03/10'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    # Rotate labels for readability
    plt.xticks(rotation=45)
    # Show the plot
    plt.tight_layout()
    plt.show()


def cond_covariance_plot(cond_covariances, cond_cov_name, year, sanctions):
    '''
    Conditional Covariance Plot setup

    Parameters:
    cond_covariances (Pandas.Dataframe): dataframe with conditional covariance
    cond_cov_name (String): plot title
    year (String): Estimation year
    sanctions (Pandas.core.indexes.datetimes.DatetimeIndex): list of sanctions in yyyy-mm-dd format

    Returns:
    Plot of Conditional Covariance with hilighted sanctions dates.
    '''

    # Plot setup
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=cond_covariances.index, y=cond_covariances, label=f"Conditional Covariance", color="blue")
    plt.title(f'Conditional Covariance, {cond_cov_name}, {year}', fontsize=14)
    # Formatting
    plt.xlabel("")#Date
    plt.ylabel("")# Covariance, b.p.
    plt.xticks(rotation=45)
    plt.grid(True)
    if str(year) in ['2022', '2023']:
        add_sanction_line(sanctions, cond_covariances)
        # Add custom red dotted line (not actually plotted)
        custom_line, = plt.plot([], [], 'r--', label='sanctions')  # 'r--' = red dotted line
    plt.legend(loc='upper left', fontsize=14) #upper right lower left
    ax = plt.gca()
    # Set x-axis ticks to three times per month at valid trading days
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # Format labels as '03/10'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    # Rotate labels for readability
    plt.xticks(rotation=45)
    # Show the plot
    plt.tight_layout()
    plt.show()
