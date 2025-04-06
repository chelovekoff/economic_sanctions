import os
import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from config import config

from functions import obtain_return


# Step 0: Loading data
#========================================================================

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

#Plot of the Returns
returns.plot()
plt.show()


# Step 1: Estimate the VAR(1) model for the mean
#========================================================================

# Fit VAR model
var_model = VAR(returns)
var_result = var_model.fit(maxlags=1)
#print(var_result.summary())

residuals = var_result.resid.to_numpy()  # Extract residuals for GARCH estimation


# Step 2: Define Bivariate GARCH(1,1) Model
#========================================================================
class BivariateGARCH:
    def __init__(self, residuals):
        self.residuals = residuals
        self.T = len(residuals)

    def log_likelihood(self, params):
        # Unpack parameters
        omega1, omega2 = params[0], params[1]
        alpha11, alpha12 = params[2], params[3]
        alpha21, alpha22 = params[4], params[5]
        beta11, beta12 = params[6], params[7]
        beta21, beta22 = params[8], params[9]

        # Initialize variance estimates
        h1 = np.zeros(self.T)
        h2 = np.zeros(self.T)

        # Set initial variance to sample variance
        h1[0] = np.var(self.residuals[:, 0])
        h2[0] = np.var(self.residuals[:, 1])

        # Compute conditional variances
        for t in range(1, self.T):
            h1[t] = (omega1 +
                     alpha11 * self.residuals[t - 1, 0] ** 2 +
                     alpha12 * self.residuals[t - 1, 1] ** 2 +
                     beta11 * h1[t - 1] +
                     beta12 * h2[t - 1])

            h2[t] = (omega2 +
                     alpha21 * self.residuals[t - 1, 0] ** 2 +
                     alpha22 * self.residuals[t - 1, 1] ** 2 +
                     beta21 * h1[t - 1] +
                     beta22 * h2[t - 1])

        # Ensure variances are positive (numerical stability)
        h1 = np.maximum(h1, 1e-6)
        h2 = np.maximum(h2, 1e-6)

        # Log-likelihood function (Gaussian assumption)
        log_likelihood = -0.5 * np.sum(np.log(h1) + self.residuals[:, 0] ** 2 / h1) \
                         -0.5 * np.sum(np.log(h2) + self.residuals[:, 1] ** 2 / h2)

        return -log_likelihood  # Minimize negative log-likelihood

# Step 3: Set better initial parameters and optimize
#========================================================================
initial_params = np.array([
    np.var(residuals[:, 0]) * 0.1,  # omega1
    np.var(residuals[:, 1]) * 0.1,  # omega2
    0.3, 0.3,  # alpha11, alpha12
    0.1, 0.1,  # alpha21, alpha22
    0.7, 0.1,  # beta11, beta12
    0.1, 0.7   # beta21, beta22
])

# Constraints: all params must be >= 0, and sum(alpha+beta) < 1 for stationarity
bounds = [(1e-6, None)] * 10  # Ensure positivity

garch_model = BivariateGARCH(residuals)
opt_result = minimize(garch_model.log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)
print(opt_result)


# Step 4: Compute Standard Errors, t-statistics, and p-values
#========================================================================
if opt_result.success:
    estimated_params = opt_result.x

    # Compute Hessian (approximate)
    hessian_inv = opt_result.hess_inv.todense()  # Get inverse Hessian
    std_errors = np.sqrt(np.diag(hessian_inv))   # Standard errors
    t_stats = estimated_params / std_errors      # t-statistics
    p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))  # p-values

    # Create DataFrame
    index_names = [
        "omega_1", "omega_2",
        "alpha_11", "alpha_12", "alpha21", "alpha_22",
        "beta_11", "beta_12", "beta_21", "beta_22"
        ]
    output_df = pd.DataFrame({
        "Optimized Parameters": estimated_params,
        "Standard Errors": std_errors,
        "T-Statistics": t_stats,
        "P-Values": p_values
        
        }, index=index_names)

    # Convert to scientific notation
    #pd.options.display.float_format = '{:.3e}'.format

    print(output_df)
    print("Log-Likelihood: ", -opt_result.fun)
    
else:
    print("Optimization failed:", opt_result.message)

