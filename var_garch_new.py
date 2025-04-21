import os
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import config

from garch_plot import cond_volatility_plot, cond_covariance_plot
from index_plot import performance_plot
from functions import obtain_return


sanction_list = config.fomc_list
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
    first_index_filename = data_files[first_index_abb]
    second_index_filename = data_files[second_index_abb]

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

# Plot of two indices performance:
performance_plot(f, first_index_filename, second_index_filename, first_index_abb, second_index_abb, year, start_date, to_year, sanctions, eng=False)

returns.plot()
plt.show()


# Prepare data (pandas DataFrame `df` with two return series):
y1 = returns.iloc[:,0].values  # series 1 returns
y2 = returns.iloc[:,1].values  # series 2 returns
T = len(y1)

# Define the negative log-likelihood function for the VAR(1)-GARCH(1,1)-CCC model
def neg_loglik(params):
    # Unpack parameters
    mu1, mu2 = params[0], params[1]
    phi11, phi12, phi21, phi22 = params[2], params[3], params[4], params[5]
    c1, c2 = params[6], params[7]
    a11, a21, a12, a22 = params[8], params[9], params[10], params[11]
    b11, b21, b12, b22 = params[12], params[13], params[14], params[15]
    rho = params[16]
    # Enforce correlation constraint
    if rho <= -0.999 or rho >= 0.999:
        return 1e8  # penalty for invalid rho
    # Initialize conditional variances
    h1 = np.var(y1)  # start with sample variance
    h2 = np.var(y2)
    # Initial residuals (at t=1, no lagged data, so using mean only)
    e1_prev = y1[0] - mu1
    e2_prev = y2[0] - mu2
    # Compute log-likelihood
    loglik = 0.0
    # Contribution of first observation (t=1)
    z1 = e1_prev / np.sqrt(h1)
    z2 = e2_prev / np.sqrt(h2)
    log_det = np.log(h1) + np.log(h2) + np.log(1 - rho**2)   # log |H_1|
    quad_form = (z1**2 - 2*rho*z1*z2 + z2**2) / (1 - rho**2)  # Mahalanobis term
    loglik += -0.5 * (np.log(2*np.pi)*2 + log_det + quad_form)
    # Recursively compute for t=2,...,T
    for t in range(1, T):
        # Update variances h1_t and h2_t using previous shocks and variances
        h1 = c1**2 + a11**2 * e1_prev**2 + a21**2 * e2_prev**2 + b11**2 * h1 + b21**2 * h2
        h2 = c2**2 + a12**2 * e1_prev**2 + a22**2 * e2_prev**2 + b12**2 * h1 + b22**2 * h2
        # Ensure positivity (penalize if any variance is non-positive)
        if h1 <= 0 or h2 <= 0:
            return 1e8
        # Compute current residuals based on VAR(1) mean prediction
        e1 = y1[t] - (mu1 + phi11*y1[t-1] + phi12*y2[t-1])
        e2 = y2[t] - (mu2 + phi21*y1[t-1] + phi22*y2[t-1])
        # Log-likelihood contribution at time t
        z1 = e1 / np.sqrt(h1)
        z2 = e2 / np.sqrt(h2)
        log_det = np.log(h1) + np.log(h2) + np.log(1 - rho**2)
        quad_form = (z1**2 - 2*rho*z1*z2 + z2**2) / (1 - rho**2)
        loglik += -0.5 * (np.log(2*np.pi)*2 + log_det + quad_form)
        # Roll over residuals and variances for next iteration
        e1_prev, e2_prev = e1, e2
    # Return negative log-likelihood for minimization
    return -loglik

# Initial guess for parameters (e.g., zeros for AR, small positives for GARCH terms, sample corr for rho)
init_params = np.array([
    y1.mean(), y2.mean(),   # mu1, mu2
    #Initial parameters according to Syriopoulos et al. (2015)
    #0.14, 0.027, 0.045, -0.04,     # phi11, phi12, phi21, phi22
    #0.52, 0.02,               # c1, c2
    #0.25, 0.03, 0.07, 0.02,     # a11, a21, a12, a22
    #0.62, 0.25, 0.007, 0.91,     # b11, b21, b12, b22
    0.1, 0.05, 0.05, 0.1,     # phi11, phi12, phi21, phi22
    0.2, 0.2,               # c1, c2
    0.3, 0.1, 0.1, 0.3,     # a11, a21, a12, a22
    0.8, 0.1, 0.1, 0.8,     # b11, b21, b12, b22
    np.corrcoef(y1, y2)[0,1] * 0.5  # rho (shrink initial corr)
])

# Set parameter bounds for constraints: (None means no bound, or use 0 for non-negativity)
bounds = [(None,None), (None,None),    # mu1, mu2
          (None,None), (None,None), (None,None), (None,None),  # phi11, phi12, phi21, phi22
          (1e-6, None), (1e-6, None),  # c1, c2 >= 0
          (0, None), (0, None), (0, None), (0, None),          # a11, a21, a12, a22 >= 0
          (0, None), (0, None), (0, None), (0, None),          # b11, b21, b12, b22 >= 0
          (-0.99, 0.99)  # rho
]

# Optimize the negative log-likelihood
result = minimize(neg_loglik, init_params, method='L-BFGS-B', bounds=bounds)
opt_params = result.x
max_loglik = -result.fun  # maximized log-likelihood value

# Compute standard errors from inverse Hessian at optimum
# (Here we use a numerical approximation of the Hessian)
from numpy.linalg import inv
eps = 1e-5
n = len(opt_params)
hessian = np.zeros((n, n))
f0 = neg_loglik(opt_params)
for i in range(n):
    theta_i_plus = opt_params.copy();  theta_i_minus = opt_params.copy()
    theta_i_plus[i] += eps;           theta_i_minus[i] -= eps
    f_plus = neg_loglik(theta_i_plus); f_minus = neg_loglik(theta_i_minus)
    hessian[i,i] = (f_plus - 2*f0 + f_minus) / (eps**2)
    for j in range(i+1, n):
        theta_ij_pp = opt_params.copy(); theta_ij_pm = opt_params.copy()
        theta_ij_mp = opt_params.copy(); theta_ij_mm = opt_params.copy()
        theta_ij_pp[i] += eps; theta_ij_pp[j] += eps
        theta_ij_pm[i] += eps; theta_ij_pm[j] -= eps
        theta_ij_mp[i] -= eps; theta_ij_mp[j] += eps
        theta_ij_mm[i] -= eps; theta_ij_mm[j] -= eps
        f_pp = neg_loglik(theta_ij_pp)
        f_pm = neg_loglik(theta_ij_pm)
        f_mp = neg_loglik(theta_ij_mp)
        f_mm = neg_loglik(theta_ij_mm)
        hessian[i,j] = hessian[j,i] = (f_pp - f_pm - f_mp + f_mm) / (4*eps**2)

cov_matrix = inv(hessian)              # inverse Hessian
std_errors = np.sqrt(np.diag(cov_matrix))

param_names = [
    'mu1', 'mu2',
    'phi11', 'phi12', 'phi21', 'phi22',
    'c1', 'c2',
    'a11', 'a21', 'a12', 'a22',
    'b11', 'b21', 'b12', 'b22',
    'rho'
]

estimates = opt_params
t_stats = estimates / std_errors
p_values = 2 * (1 - norm.cdf(abs(t_stats)))  # two-tailed

results_df = pd.DataFrame({
    'Estimate': (np.round(estimates, 5)),
    'Std. Error': (np.round(std_errors, 5)),
    't-stat': (np.round(t_stats, 5)),
    'p-value': (np.round(p_values, 5))
}, index=param_names)

print("===== Estimation Results =====")
print(results_df)
print("\nMaximized Log-Likelihood =", round(max_loglik, 5))


# Svaing the results of VAR-GARCH estimation in .CSV
results_df.to_csv(f'results/{year}_var-garch_{first_index_abb}-{second_index_abb}.csv', index=True)



def compute_conditional_variances(y1, y2, params):
    T = len(y1)
    mu1, mu2 = params[0], params[1]
    phi11, phi12, phi21, phi22 = params[2], params[3], params[4], params[5]
    c1, c2 = params[6], params[7]
    a11, a21, a12, a22 = params[8], params[9], params[10], params[11]
    b11, b21, b12, b22 = params[12], params[13], params[14], params[15]
    # rho = params[16]  # not needed for volatility

    h1_vals = np.zeros(T)
    h2_vals = np.zeros(T)

    e1_vals = np.zeros(T)
    e2_vals = np.zeros(T)

    # Initial variance (e.g., sample variance or unconditional variance)
    h1_vals[0] = np.var(y1)
    h2_vals[0] = np.var(y2)

    # Initial residuals (simple mean prediction)
    e1_vals[0] = y1[0] - mu1
    e2_vals[0] = y2[0] - mu2

    for t in range(1, T):
        h1_vals[t] = (
            c1**2
            + a11**2 * e1_vals[t - 1] ** 2
            + a21**2 * e2_vals[t - 1] ** 2
            + b11**2 * h1_vals[t - 1]
            + b21**2 * h2_vals[t - 1]
        )

        h2_vals[t] = (
            c2**2
            + a12**2 * e1_vals[t - 1] ** 2
            + a22**2 * e2_vals[t - 1] ** 2
            + b12**2 * h1_vals[t - 1]
            + b22**2 * h2_vals[t - 1]
        )

        # Compute residuals from VAR(1)
        e1_vals[t] = y1[t] - (mu1 + phi11 * y1[t - 1] + phi12 * y2[t - 1])
        e2_vals[t] = y2[t] - (mu2 + phi21 * y1[t - 1] + phi22 * y2[t - 1])

    return h1_vals, h2_vals

h1, h2 = compute_conditional_variances(y1, y2, result.x)
h1 = np.sqrt(h1)
h2 = np.sqrt(h2)
h1 = pd.DataFrame(h1, columns=[first_index_abb])
h2 = pd.DataFrame(h2, columns=[second_index_abb])
h1.index = returns.index
h2.index = returns.index

condit_volatility = pd.merge(h1, h2, left_index=True, right_index=True, how='left')

# Conditional volatilities Plot
cond_volatility_plot(condit_volatility, year, sanctions)

# Extract Constant Conditional Correlation
rho = result.x[16]

# Step 5: Compute Conditional Covariances
cond_covariances = rho * h1.iloc[:, 0]  * h2.iloc[:, 0] 
cond_covariances.index.name = "Date"
cond_cov_name = f'{returns.columns[0]} vs {returns.columns[1]}'
cond_covariances.name = cond_cov_name

# Display results
print("\nConstant Conditional Correlation (ρ_12):", rho)
print("\nFirst 3 Conditional Covariance Values:\n", cond_covariances.tail(3))

# Conditional Covariance Plot setup
cond_covariance_plot(cond_covariances, cond_cov_name, year, sanctions)

