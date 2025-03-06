import pandas as pd
import numpy as np
from arch import arch_model
import statsmodels.api as sm
import matplotlib.pyplot as plt

f = "stock_data/"

def estimate_garch(filename, two_parts=False, r='simple', plot_show=False):
    '''
    Obtain conditional volatility and standardized residuals

    Parameters:
    filename (String): .xlsx filename of initial data set with dates and asset pricesData
    two_parts (Boolean): True if data consists of two separate files
    r (String): Type of retutn. 'siple' for simple return and 'log' for logarithmic retutn
    plot_show (Boolean): False if no plot for GARCH parameters is required
   
    Returns:
    asset_df (pandas.DataFrame): DataFrame with 'gm_std' and 'gm_std_resid' columns: Conditional Volatility and Standardized Residuals DataFrame
    '''
    
    # File pre-procesing:
    if two_parts:
        df_first = pd.read_excel(f+filename+'_01.xlsx', index_col=0, parse_dates=True)
        df_second = pd.read_excel(f+filename+'_02.xlsx', index_col=0, parse_dates=True)
        df = pd.concat([df_first, df_second])
    else:
        df = pd.read_excel(f+filename+'.xlsx', index_col=0, parse_dates=True)
    df = df.sort_values(by='Дата')
    df.columns = ['level']

    # Return calculation:
    if r == 'simple':
        df['return'] = (df.level.pct_change()) # asset simple return
    elif r == 'log':
         df['return'] = (np.log(df.level) - np.log(df.level.shift(1))) # asset log return
    df = df.dropna()
    #df = df[df.index<="2019-11-26"]

    # Specify GARCH model assumptions
    basic_gm = arch_model(df['return'], p = 1, q = 1,
                        mean = 'constant', vol = 'GARCH', dist = 'normal')
    # Fit the model
    gm_result = basic_gm.fit(disp = 'off')
    
    # Display model fitting summary
    print(gm_result.summary())

    # Plot fitted results
    if plot_show:
        gm_result.plot()
        plt.show()

    #-----------------
    # Obtain model estimated residuals and volatility
    gm_resid = gm_result.resid 
    gm_std = gm_result.conditional_volatility
    gm_std.name = 'gm_std'

    # Calculate the standardized residuals
    gm_std_resid = gm_resid /gm_std
    gm_std_resid.name = 'gm_std_resid'

    asset_df = pd.merge(gm_std, gm_std_resid, left_index=True, right_index=True, how='left')
    asset_df = asset_df.dropna()

    basic_gm = None
    gm_result = None

    return asset_df


def garch_covariance (my_list, plot_show=False):
    '''Calculate GARCH covariance'''
    # Create an empty DataFrame with the date index
    date_range = pd.date_range(start="2009-01-01", end="2025-12-31", freq="D")
    result_df = pd.DataFrame(index=date_range)

    for i in my_list:
        asset_df = estimate_garch(i, r='log', plot_show=True) #, r='log', two_parts=True,
        result_df = pd.merge(result_df, asset_df, left_index=True, right_index=True, how='left')
    result_df = result_df.dropna()

    # Calculate correlation (with standardized residuals)
    corr = np.corrcoef(result_df.iloc[:, 1], result_df.iloc[:, 3])[0,1] # Correlation between standardized residuals
    print('Correlation: ', corr)
    # Calculate GARCH covariance (with estimated volatilities)
    covariance =  corr * result_df.iloc[:, 0] * result_df.iloc[:, 2]

    # Plot fitted results
    if plot_show:
        plt.plot(covariance, color = 'gold')
        plt.title('GARCH Covariance')
        plt.show()

    return covariance

print("====================Call of function====================")
first_asset = 'rtsi' #rtsi, imoex, eurusd_tom, eur_usd-(fx)
#estimate_garch(first_asset, r="simple", plot_show=True)
second_asset = 's-p-500' #usd_cad-(fx), s-p-500, usd_cad-(fx)
two_assets = [first_asset, second_asset]
covariance = garch_covariance(two_assets, plot_show=True)
print(covariance.head())
