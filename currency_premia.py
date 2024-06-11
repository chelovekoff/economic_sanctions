import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import jarque_bera

# Get USD/RUB exchange rate DF
def get_exchange_rate (cbr):
    cbr_data = pd.read_excel(f+cbr, index_col=0, parse_dates=True)
    cbr_data.columns = ['cbr']
    cbr_data = cbr_data.sort_values(by='Дата')
    return cbr_data

def currency_return (forward, spot, cbr):#, return_type
    try:
        # for a forward USD/RUB rate
        forward_data = pd.read_excel(f+forward, index_col=0, parse_dates=True)
        forward_data.columns = ['fwd']
        forward_data = forward_data.sort_values(by='Дата')
        
        # for a spot USD/RUB rate
        spot_data = pd.read_excel(f+spot, index_col=0, parse_dates=True)
        spot_data.columns = ['spt']
        spot_data = spot_data.sort_values(by='Дата')

        # for a CBR's USD/RUB rate
        cbr_data = get_exchange_rate (cbr)
        cbr_data['ln_cbr_return'] = np.log(cbr_data.cbr) - np.log(cbr_data.cbr.shift(1))
        cbr_data['mean_ln_cbr_return'] = cbr_data['ln_cbr_return'].rolling(window=250).mean()
        rub_return =  pd.DataFrame(cbr_data['mean_ln_cbr_return'])
        rub_return = rub_return.dropna(subset=['mean_ln_cbr_return'])

        #print("the first 16 rows:\n", cbr_data.mean_ln_cbr_return.head(16))

        currency_data = pd.merge(forward_data, spot_data, left_index=True, right_index=True, how='left')
        currency_data["expected_change"] = currency_data["fwd"]/currency_data["spt"]-1
        currency_data["ln_expected_change"] = np.log(currency_data["fwd"]) - np.log(currency_data["spt"])
        currency_data = pd.merge(currency_data, cbr_data['mean_ln_cbr_return'], left_index=True, right_index=True, how='left')

        #print(forward_data.tail(), "\n", spot_data.tail())
        print(rub_return)
        return currency_data, rub_return
    except Exception as e:
        print("Error: ", str(e))



f = "stock_data/"
forward = "usd_rub-o_n-fx-(outright)-mid.xlsx"
'''S/N (Spot/Next): This refers to transactions that are settled one business day after the spot date
(usually two days after the transaction date).
One b/day: usd_rub-s_n-fx-(outright)-mid (just from 28.02.2022)
O/N (Overnight): This refers to transactions that are settled on the next business day. It's the closest to a 1-day forward rate.
Overnight: usd_rub-o_n-fx-(outright)-mid
One month: usd_rub-1m-fx-(outright)-mid
One year: usd_rub-1y-fx-(outright)-mid'''
spot = "средневзвешенный-курс-usdrub_tom.xlsx"
cbr = "usd_rub-(банк-россии).xlsx"


currency_data, ___ = currency_return(forward, spot, cbr)
print("Final data:\n", currency_data.head())

# Expected Exchange Rate Return
plt.figure(figsize=(12, 4))
plt.plot(currency_data["ln_expected_change"], color='green', label='CIRP concepte')
plt.plot(currency_data['mean_ln_cbr_return'], color='blue', label='CBR Exchange Rate')
#plt.axhline(x=0, color='red', linestyle='--')
plt.title('') #'Cumulative Average Abnormal Return: ' + 
plt.xlabel('') #'Date', fontsize=14
plt.ylabel('Return', fontsize=14)
plt.grid(True)
plt.legend(loc='lower right')
plt.show()

# Distribution of the Expected Exchange Rate Return
plt.figure(figsize=(12, 6))
# Plot distribution of 'ln_expected_change'
plt.subplot(1, 2, 1)
sns.histplot(currency_data["ln_expected_change"], bins=15, kde=True, color='blue')
plt.title('Distribution of the Expected Return (the CIRP)')
plt.xlabel('Expected Return')
plt.ylabel('Frequency')
plt.grid(True)
# Plot distribution of 'mean_ln_cbr_return'
plt.subplot(1, 2, 2)
sns.histplot(currency_data['mean_ln_cbr_return'], bins=15, kde=True, color='green')
plt.title('Distribution of the Expected Return (CBR historical data)')
plt.xlabel('Expected Return')
plt.ylabel('Frequency')
plt.grid(True)
# Adjust layout for better readability
plt.tight_layout()
# Show the plots
plt.show()

descr_stats_cirp = currency_data["ln_expected_change"].dropna().describe()
descr_stats_cbr = currency_data["mean_ln_cbr_return"].dropna().describe()

print(descr_stats_cirp, "\n", descr_stats_cbr)


#DESCRIPTIVE STATISTIC

def get_stats(series):
    """Calculate descriptive statistics and Jarque-Bera test for a given series."""
    descr_stats = series.describe()
    jb_test = jarque_bera(series)
    return descr_stats, jb_test

# Get statistics for both series
descr_stats_cirp, jb_test_cirp = get_stats(currency_data["ln_expected_change"].dropna())
descr_stats_cbr, jb_test_cbr = get_stats(currency_data["mean_ln_cbr_return"].dropna())

# Create a DataFrame for descriptive statistics
descr_stats = pd.DataFrame({
    "ln_expected_change": descr_stats_cirp,
    "mean_ln_cbr_return": descr_stats_cbr
})

# Add Jarque-Bera test results to the DataFrame
jb_results = pd.DataFrame({
    "ln_expected_change": [jb_test_cirp.statistic, jb_test_cirp.pvalue],
    "mean_ln_cbr_return": [jb_test_cbr.statistic, jb_test_cbr.pvalue]
}, index=["Jarque-Bera statistic", "p-value"])

# Concatenate the descriptive statistics and Jarque-Bera results
combined_stats = pd.concat([descr_stats, jb_results])
combined_stats.columns = ['CIRP', 'CBR']
print(combined_stats)