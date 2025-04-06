import os
import pandas as pd
from statsmodels.tsa.api import VAR
from arch.univariate import GARCH, ConstantMean
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from config import config

from functions import obtain_return, get_stats
from garch_plot import distr_plot, cond_volatility_plot, cond_covariance_plot

sanction_list = config.sanction_list
sanctions = pd.to_datetime(sanction_list)

data_files = config.data_files
vars_list = list(data_files.keys())
possible_vars = '/'.join(vars_list)

f = "stock_data/"

# Checking if there is a required data set in the project folder:
preloaded_returns = 'returns.csv'

if os.path.exists(preloaded_returns):
    # Load ready data with two indices returns if it exists
    returns = pd.read_csv(preloaded_returns)
    returns = returns.rename(columns={'Дата': 'Date'})
    returns['Date'] = pd.to_datetime(returns['Date'])
    returns = returns.set_index('Date')
    first_index_abb = returns.columns[0]
    second_index_abb = returns.columns[1]

else:
    # Otherwise, manual inputing the first required variable:
    while True:
        try:
            first_index_abb = str(input(f"Input the first variable for VAR-GARCH estimation ('{possible_vars}'): ")).strip().upper()
            if first_index_abb in vars_list:
                vars_list.remove(first_index_abb)
                possible_vars = '/'.join(vars_list)
                first_index_filename = data_files[first_index_abb]
                break
        except ValueError:
            print(f"Please input a valid variable ('{possible_vars}').")
    # manual inputing the second required variable:
    while True:
        try:
            second_index_abb = str(input(f"Input the second variable for VAR-GARCH estimation ('{possible_vars}'): ")).strip().upper()
            if second_index_abb in vars_list:
                second_index_filename = data_files[second_index_abb]
                break
        except ValueError:
            print(f"Please input a valid variable ('{possible_vars}').")

    # Load Indices
    return_first = obtain_return(f, first_index_filename, first_index_abb) # Loading the first index return
    return_second = obtain_return(f, second_index_filename, second_index_abb) # Loading the second index return
    returns = pd.merge(return_first, return_second, left_index=True, right_index=True, how='left')
    returns = returns.dropna()

# Input the year:
while True:
    try:
        year = int(input("Input the year of the calculation start (in the 'yyyy' format): "))
        break  # Exit the loop if successfully converted to an integer.
    except ValueError:
        print("Please enter a valid integer.")
to_year = str(int(year) + 1)
if str(year) == '2022':
    start_month = '03'
else:
    start_month = '01'
start_date = f"{year}-{start_month}-01"
returns = returns[(returns.index>=start_date) & (returns.index<=f"{to_year}-01-01")]
print(returns.head(3), "\n", returns.tail(3))


#Distribution of the Realized Market Return
distr_plot(returns.iloc[:,0], year)

#Plot of the Returns
returns.plot()
plt.show()

# Get descriptive statistics for both series
df_stats = get_stats(returns)
print(df_stats)
df_stats.to_excel(f'{year}_descr_stats_return_garch.xlsx', index=True)


# Step 1: Fit VAR(1) Model
var_model = VAR(returns)
var_result = var_model.fit(1)  # VAR(1)
print("================================VAR================================")
print(var_result.summary())

# Extract residuals from VAR model
residuals = var_result.resid

# Step 2: Fit GARCH(1,1) models separately for each return series
garch_models = {}
for col in returns.columns:
    garch = ConstantMean(residuals[col])
    garch.volatility = GARCH(p=1, q=1)  # GARCH(1,1) process, for GJR-GARCH include: o=1
    garch_models[col] = garch.fit(disp="off")
    print(f"================================GARCH ({col})================================\n",
          f"\nGARCH(1,1) Model Summary for {col}:\n",
          garch_models[col].summary())

# Step 3: Compute Conditional Volatilities
h_1 = garch_models[first_index_abb].conditional_volatility
h_2 = garch_models[second_index_abb].conditional_volatility
#h_3 = garch_models['r_cur'].conditional_volatility
print(type(h_1))

# Plots of estimated conditional volatilities
condit_volatility = pd.merge(h_1, h_2, left_index=True, right_index=True, how='left')
#condit_volatility = pd.merge(condit_volatility, h_3, left_index=True, right_index=True, how='left')
condit_volatility.columns = returns.columns.tolist()
print("Conditional volatilities DF:\n", condit_volatility.head(3))
# Conditional volatilities Plot setup
cond_volatility_plot(condit_volatility, year, sanctions)


# Step 4: Compute Constant Conditional Correlation
rho = pearsonr(residuals[first_index_abb], residuals[second_index_abb])[0] #for h_1 and h_2

# Step 5: Compute Conditional Covariances
cond_covariances = rho * h_1 * h_2
cond_covariances.index.name = "Date"
cond_cov_name = f'{returns.columns[0]} vs {returns.columns[1]}'
cond_covariances.name = cond_cov_name

# Display results
print("\nConstant Conditional Correlation (ρ_12):", rho)
print("\nFirst 3 Conditional Covariance Values:\n", cond_covariances.tail(3))

# Conditional Covariance Plot setup
cond_covariance_plot(cond_covariances, cond_cov_name, year, sanctions)
