import pandas as pd
import numpy as np
from scipy.stats import jarque_bera
from arch.unitroot import ADF, PhillipsPerron


def obtain_return(folder, file_name, column_name, return_type = 'log'):
    '''
    Obtaining dataframe with daily Index returns

    Parameters:
    folder (String): foler name
    file_name (String): name of .xlsx file with daily Index values
    column_name (String): name of the output column
    return_type (String): type of the returns
        log: logarithmic (Default)
        sim: simple return

    Returns:
    condit_volatility (Pandas.Dataframe): dataframe with two columns (condtional volatilities)
    '''

    #Market data
    df = pd.read_excel(folder+file_name+'.xlsx', index_col=0, parse_dates=True) #rtsi, imoex, eurusd_tom, usd_cad-(fx)_02, usd_eur-(fx)
    df = df.sort_values(by='Дата')
    df.columns = ['level']
    # Return calculation:
    if return_type == 'log':
         df[column_name] = 100*(np.log(df.level) - np.log(df.level.shift(1))) # asset log return
    elif return_type == 'sim':
        df[column_name] = (df.level.pct_change()) # asset simple return
    df = df.dropna()
    return df[column_name]

def get_stats(series):
    """Calculate descriptive statistics, ADF and PP tests, and Jarque-Bera test for a given series."""
    """JB's p-value=0 ->  variable does not appear to be normally distributed."""
    descr_stats = series.describe().round(3)
    jb_stats = []
    adf = []
    pp = []
    for column in series:
        jb_test = jarque_bera(series[column]) # JB test
        jb_stats.append(f'{round(jb_test.statistic, 3)} (p-value {round(jb_test.pvalue, 3)})')
        adf_test = ADF(series[column]) # ADF
        adf.append(f'{round(adf_test.stat, 3)} (p-value {round(adf_test.pvalue, 3)})')
        pp_test = PhillipsPerron(series[column]) # PP
        pp.append(f'{round(pp_test.stat, 3)} (p-value {round(pp_test.pvalue,3)})')
    print(len(jb_stats), len(adf), len(pp))

    timeseries_stats_df = pd.DataFrame({
        'JB' : jb_stats,
        'ADF': adf,
        'PP': pp
        })
    timeseries_stats_df = timeseries_stats_df.T
    timeseries_stats_df.columns = descr_stats.columns.tolist()
    descr_stats = pd.concat([descr_stats, timeseries_stats_df], ignore_index=False)

    return descr_stats
