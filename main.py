import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def returns_calc(stock, risk_free, market):
    
    # Calculation of the day risk-free rate:
    risk_free.columns = ['r_f']
    risk_free['r_f'] = risk_free['r_f']/100
    risk_free = (1 + risk_free) ** (1/365) - 1
    #print("=======\nRF:\n", risk_free.head())

    # Calclulation of the day market return:
    market = market.sort_values(by='Дата')
    market.columns = ['m']
    market['r_m'] = market['m'].pct_change()  
    #print("=======\nMR:\n", market.head())

    try:
        # for a single stock
        stock = pd.read_excel(f+stock+'.xlsx', header=1, index_col=0, parse_dates=True, usecols=['Дата', 'Закрытие'])
        stock = stock.sort_values(by='Дата')
        stock['r_i'] = stock['Закрытие'].pct_change() # Factual stock return
    except:
        # for a sector index
        stock = pd.read_excel(f+stock+'.xlsx', index_col=0, parse_dates=True)
        stock.columns = ['sector_index']
        stock = stock.sort_values(by='Дата')
        stock['r_i'] = stock['sector_index'].pct_change() # Factual stock return
    stock = pd.merge(stock, risk_free, left_index=True, right_index=True, how='left')
    stock = pd.merge(stock, market['r_m'], left_index=True, right_index=True, how='left')
    stock['ri_rf'] = stock['r_i']- stock['r_f'] # Excess risk-free stock return
    stock['rm_rf'] = stock['r_m']- stock['r_f'] # Excess risk-free market return
    #print(f"=======\n{stock}:\n", stock.head(10))
    #print(SIBN.isna().sum()) #How many NaN has each column
    stock = stock.dropna(subset=['r_i'])

    # Calculate rolling beta
    def rolling_beta(ri_rf, rm_rf, window_size):
        beta_values = [float('nan')] * (window_size - 1)  # to match the length of the input array
        for i in range(window_size, len(ri_rf) + 1):
            Y = ri_rf[i-window_size:i].values
            X = rm_rf[i-window_size:i].values
            # Add constant to the array to perform an OLS regression
            X = sm.add_constant(X)
            model = sm.OLS(Y, X).fit()
            # Extract just the beta coefficient
            beta = model.params[1]
            beta_values.append(beta)
        return beta_values

    stock['beta'] = rolling_beta(stock['ri_rf'], stock['rm_rf'], window_size) # Market Beta
    #print(stock.isna().sum()) #How many NaN has each column
    stock = stock.dropna(subset=['beta'])
    stock['r_e'] = stock['r_f'] + stock['beta'] * stock['rm_rf'] # Expected stock return
    stock['r_a'] = stock['r_i'] - stock ['r_e'] # Abnormal stock return
    return stock

def tau_df(sanction_date, df, tau):
    important_date = pd.to_datetime(sanction_date)
    sanction_index = df.index.searchsorted(important_date)
    #sanction_index = df.index.get_loc(important_date)
    start_index = max(sanction_index - tau, 0)  # Ensuring it doesn't go below 0
    end_index = min(sanction_index + tau, len(df) - 1)  # Ensuring it doesn't exceed df length
    filtered_df = df.iloc[start_index:end_index + 1].copy()  # +1 because upper bound is exclusive in iloc
    filtered_df = filtered_df.copy()
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.index = range(-tau, tau + 1)
    filtered_df['r_cum'] = (1 + filtered_df['r_a']).cumprod() - 1 # Cumulative abnormal return
    return filtered_df 

# Calcilating an asymptotic confidence interval for CAAR
def calculate_conf_intervals(df, conf_level):
    
    z_score = abs(stats.norm.ppf((1-conf_level)/2)) # Z-score for the conf.level
    n = len(df.columns) - 1 # Number of CAR values
    # Calculate the standard deviation for the first n columns for each row
    std_dev = df.iloc[:, :n].std(axis=1)
    # Calculate the margin of error
    margin_of_error = z_score * (std_dev / np.sqrt(n))*100
    # Calculate the lower and upper confidence interval bounds
    df['CI_lower'] = df.iloc[:, -1] - margin_of_error
    df['CI_upper'] = df.iloc[:, -2] + margin_of_error
    
    return df


f = "stock_data/"
window_size = 250
sanction_dates = ['2022-05-30',
                  '2022-06-03',
                  '2022-09-02',
                  '2022-10-06',
                  '2022-12-05',
                  '2023-02-04',
                  '2023-06-23',#+PR
                  '2023-12-18',
                  #'2023-06-24' #PR
                  ]
seven_pacage = '2022-07-21' #The EU implements G7 commitments with its seventh sanctions package by banning imports of gold from Russia, clarifying and expanding on existing export controls, and sanctioning an additional 54 individuals and 10 entities.
prigozhin = '2023-06-24'
treasury_ban = '2023-11-02' #The US Treasury sanctions entities in China, Turkey, and the United Arab Emirates for sending high-priority dual use goods to Russia, as well as 7 Russian banks and other individuals and entities. The US State Department issues sanctions that affect over 90 entities for sanctions evasion and also target Russia’s future energy capabilities.

blue_chips = ['CHMF',
              'GAZP',
              'GMKN',
              'IRAO',
              'LKOH',
              'MGNT',
              'MTSS',
              'NVTK',
              'PLZL',
              'ROSN',
              'RUAL',
              'SBER',
              'SNGS',
              'TATN',
              'YNDX'
              ]

moexog_chips = ['BANEP',
                'GAZP',
                'LKOH',
                'NVTK',
                'RNFT',
                'ROSN',
                'SNGS',
                'SNGSP',
                'TATN',
                'TATNP',
                'TRNFP',
                #'MOEXOG',
]

while True:
    try:
        tau = int(input("Input the 'tau' value: "))
        break  # Exit the loop if successfully converted to an integer.
    except ValueError:
        print("Please enter a valid integer.")
#estimated_stock = input("input the stock for estimation: ")

# Risk-free rate: RUB Yield Curve 1Y
risk_free = pd.read_excel(f+'rub-yield-curve-1y.xlsx', index_col=0, parse_dates=True)

# Market return: IMOEX
market = pd.read_excel(f+'IMOEX.xlsx', index_col=0, parse_dates=True)

"""
#Stock CAR
SR_tau = tau_df(sanction_dates[5], SR, tau)
print (SR_tau)
print(len(SR_tau))

plt.figure(figsize=(10, 6))
plt.plot(SR_tau['r_a'], label='Abnormal return', color='blue', linestyle=':')
plt.plot(SR_tau['r_cum'], label='CAR', color='green')
plt.axvline(x=0, color='red', linestyle='--')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.title('Abnormal stock return "LKOH": ' + str(sanction_dates[5]))
plt.xlabel('Days')
plt.ylabel('Return')
plt.grid(True)
#plt.gca().xaxis.set_major_formatter(DateFormatter('%m-%d'))
plt.show()
"""
cum_av_return = pd.DataFrame() # Dataframe for all CAAR stocks
for oilstock in moexog_chips:
    print("\n", oilstock, ": ")
    cum_return = pd.DataFrame() # Dataframe for all CARs and particular stock
    # Stock return
    SR = returns_calc(oilstock, risk_free, market)

    for sanction in sanction_dates:
        filtered_df = tau_df(sanction, SR, tau) # Only event window
        cum_return[sanction] = filtered_df['r_cum']
    cum_return['CAAR']= cum_return.mean(axis=1)*100 # Cumulative average abnormal return in percente
    calculate_conf_intervals(cum_return, 0.9) # CAAR asymptotic confidence interval
    cum_av_return['CAAR_'+oilstock] = cum_return['CAAR'].round(2)
    #cum_return = None
    print(cum_return)

    # CAAR plot
    plt.figure(figsize=(4, 3))
    plt.plot(cum_return.iloc[:,-3], label='Abnormal return', color='green')
    # Adding the confidence interval area
    plt.fill_between(cum_return.index, cum_return.iloc[:,-2], cum_return.iloc[:,-1], color='lightblue', alpha=0.4, label='Confidence Interval')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.title(oilstock) #'Cumulative Average Abnormal Return: ' + 
    plt.xlabel('Event window, days', fontsize=14)
    plt.ylabel('Return, %', fontsize=14)
    plt.grid(True)
    plt.show()

# Save the DataFrame with all CAARs to an Excel file
cum_av_return.to_excel(f'caar_oilgas_{tau}.xlsx', index=True)

print(cum_av_return)
descr_stats = cum_av_return.iloc[-1].describe()
print(descr_stats)
#descr_stats.to_excel(f'caar_stat.xlsx', index=True)