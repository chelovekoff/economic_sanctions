import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from arch.univariate import GARCH, ConstantMean
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from arch.unitroot import ADF, PhillipsPerron
from scipy.stats import jarque_bera



def obtain_rf(filename):
    # Risk-free rate: RUB Yield Curve 1Y
    risk_free = pd.read_excel(f+filename, index_col=0, parse_dates=True)
    risk_free = risk_free.sort_values(by='Дата')

    # Calculation of the day risk-free rate:
    risk_free.columns = ['r_f']
    #risk_free['r_f'] = risk_free['r_f']/100
    #risk_free = 100*((1 + risk_free) ** (1/12) - 1) #Monthly risk-free rate, in % #1/365 - daily
    #print("=======\nRF:\n", risk_free.head(3))
    #risk_free.plot()
    #plt.show()

    return risk_free

f = "stock_data/"

# Load and preprocess data

sanction_list = [
    '2022-06-03', #6th package - oil import, SWIFT
    '2022-09-02', #8th package - announcment
    '2022-10-06', #8th package - imposition
    '2023-02-04', #!!! 8th package - oil price ceiling imposition
    '2023-06-23' # 11th package - tanker fleet
]
sanctions = pd.to_datetime(sanction_list)

def add_sanction_line(sanctions, df):
    '''Add highlighted regions'''
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

f = "stock_data/"

# Risk free rate:
filename_rf = 'rub-yield-curve-1y.xlsx' #rub-yield-curve-1y,  s-p-500, rgbitr-ru000a0jqv87
risk_free = pd.read_excel(f+filename_rf, index_col=0, parse_dates=True)
risk_free = risk_free.sort_values(by='Дата')
risk_free.columns = ['r_f']
r_f = 1*((risk_free.r_f) - (risk_free.r_f.shift(1)))# (risk_free['r_f'] - risk_free['r_f'].shift(1))
r_f.name = 'r_f'
r_f = r_f.dropna()

#r_f.plot()
#plt.show()


#Market data
market = pd.read_excel(f+'oil/MOEXOG.xlsx', index_col=0, parse_dates=True) #rtsi, imoex, eur_usd-(fx)_02, usd_cad-(fx)_02, oil/MOEXOG
market = market.sort_values(by='Дата')
market.columns = ['m']
market['r_m'] = 100*(np.log(market.m) - np.log(market.m.shift(1))) #(market.m.pct_change()) # market log return #100 * (sp_price['Close'].pct_change())
#market['r_m'] = market['r_m'].shift(1)

market = market.dropna()

#SP500
sp_name = 's-p-500.xlsx' #usd_cad-(fx), s-p-500, eur_usd-(fx)_01, eur_usd-(fx)_02, usd_rub-(банк-россии)
us_market = pd.read_excel(f+sp_name, index_col=0, parse_dates=True)
us_market = us_market.sort_values(by='Дата')
us_market.columns = ['m']
us_market['r_sp500'] = 100*(np.log(us_market.m) - np.log(us_market.m.shift(1))) #(us_market.m.pct_change()) # market log return
#us_market['r_sp500'] = us_market['r_sp500']**2

#Assamption that a lag could be presented between MOEX and SP500:
#us_market['r_sp500'] = us_market['r_sp500'].shift(1)

# Currency data
cur_name = 'usd_rub-(банк-россии).xlsx' #usd_cad-(fx), s-p-500, eur_usd-(fx)_01, eur_usd-(fx)_02, usd_rub-(банк-россии), usd_eur-(fx)
cur_market = pd.read_excel(f+cur_name, index_col=0, parse_dates=True)
cur_market = cur_market.sort_values(by='Дата')
cur_market.columns = ['cur']
cur_r = 100*(np.log(cur_market.cur) - np.log(cur_market.cur.shift(1))) #(us_market.m.pct_change()) # market log return
cur_r.name = 'r_cur'


df = pd.merge(market['r_m'], r_f, left_index=True, right_index=True, how='left')
df = pd.merge(df, cur_r, left_index=True, right_index=True, how='left')
df = df.dropna()

year = '2021'
to_year = str(int(year) + 1)
df = df[(df.index>=f"{year}-01-01") & (df.index<=f"{to_year}-01-01")]

# Distribution of the Realized Market Return
plt.figure(figsize=(8, 6))
# Plot distribution of 'ln_expected_change'
#plt.subplot(1, 2, 1)
sns.histplot(df["r_m"], bins=15, kde=True, color='blue')
plt.title(f'Distribution of the Realized Market Return, {year}')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.grid(True)
# Adjust layout for better readability
plt.tight_layout()
# Show the plots
plt.show()


# Descriptive statistics of returns:
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

# Get statistics for both series
df_stats = get_stats(df)
print(df_stats)
df_stats.to_excel(f'{year}_descr_stats_return_garch.xlsx', index=True)

'''
#Plot of the Returns
plt.plot(df, label = 'Returns')
plt.legend(loc='upper right')
plt.show()
'''

print(df.head())
print(df.tail())


# Correct Pandas syntax to select multiple columns
returns = df[['r_m', 'r_f', 'r_cur']]

# Step 1: Fit VAR(1) Model
var_model = VAR(returns)
var_result = var_model.fit(1)  # VAR(1)
print(var_result.summary())

# Extract residuals from VAR model
residuals = var_result.resid

# Step 2: Fit GARCH(1,1) models separately for each return series
garch_models = {}
for col in residuals.columns:
    garch = ConstantMean(residuals[col])
    garch.volatility = GARCH(1, 1)  # GARCH(1,1) process
    garch_models[col] = garch.fit(disp="off")
    print(f"\nGARCH(1,1) Model Summary for {col}:\n", garch_models[col].summary())

# Step 3: Compute Conditional Volatilities
h_1 = garch_models['r_m'].conditional_volatility
h_2 = garch_models['r_f'].conditional_volatility
h_3 = garch_models['r_cur'].conditional_volatility

# Plots of estimated conditional volatilities
condit_volatility = pd.merge(h_1, h_2, left_index=True, right_index=True, how='left')
condit_volatility = pd.merge(condit_volatility, h_3, left_index=True, right_index=True, how='left')
condit_volatility.columns = returns.columns.tolist()
print("Conditional volatilities DF:\n", condit_volatility.head())

#==================Conditional volatilities==================
plt.figure(figsize=(8, 5))
for col in condit_volatility.columns:
    plt.plot(condit_volatility.index, condit_volatility[col], label=col)
#plt.axvline(x=0, color='grey', linestyle='--')
#plt.axhline(y=0, color='grey', linestyle='-', linewidth=1)
plt.title(f'Conditional Volatilities, {year}')
plt.xlabel("", fontsize=10) #year
plt.ylabel('Volatility, b.p.', fontsize=10)
plt.grid(True)
add_sanction_line(sanctions, condit_volatility)
# Add custom red dotted line (not actually plotted)
custom_line, = plt.plot([], [], 'r--', label='sanctions')  # 'r--' = red dotted line
plt.legend(loc='lower left') #upper left
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

'''
# Assuming your DataFrame has a DateTime index
fig, ax = plt.subplots()
# Plot your time-series DataFrame
ax.plot(df.index, df['your_column'])
# Set major ticks every week
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MONDAY))  # Adjust as needed
# Format the ticks as 'Mar.27'
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b.%d'))
# Rotate labels for better readability
plt.xticks(rotation=45)
'''

# Step 4: Compute Constant Conditional Correlation
rho_12 = pearsonr(residuals['r_m'], residuals['r_cur'])[0]

# Step 5: Compute Conditional Covariances
cond_covariances = rho_12 * h_1 * h_2
cond_covariances.index.name = "Date"
cond_cov_name = f'{returns.columns[0]} vs {returns.columns[1]}'
cond_covariances.name = cond_cov_name
print(type(cond_covariances))


# Display results
print("\nConstant Conditional Correlation (ρ_12):", rho_12)
print("\nFirst 5 Conditional Covariance Values:\n", cond_covariances.tail(3))

#================================Conditional Covariance Plot setup================================
# Plot setup
plt.figure(figsize=(8, 5))
sns.lineplot(x=cond_covariances.index, y=cond_covariances, label=f"Conditional Covariance", color="blue")
plt.title(f'Conditional Covariance, {cond_cov_name}, {year}')
# Formatting
plt.xlabel("")#Date
plt.ylabel("")# Covariance, b.p.
plt.xticks(rotation=45)
plt.grid(True)
add_sanction_line(sanctions, cond_covariances)
custom_line, = plt.plot([], [], 'r--', label='sanctions')  # 'r--' = red dotted line
plt.legend(loc='lower left') #upper right
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