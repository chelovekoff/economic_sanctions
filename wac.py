import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pycwt as wavelet
#from pycwt.helpers import find
from pycwt.wavelet import Morlet
#from pycwt.helpers import fftconv
from scipy.ndimage import uniform_filter1d  # for consistent output shape
import matplotlib.dates as mdates

from config import config
from functions import obtain_return
from garch_plot import add_sanction_line

sanction_list = config.sanction_list # fomc_list, sanction_list, us_fin_sanction_dates
sanctions = pd.to_datetime(sanction_list)


def wavelet_smooth(power, scales, dt):
    """Smooth wavelet power spectra in time and scale."""
    m, n = power.shape
    smooth_power = np.zeros_like(power)
    # Time smoothing
    for i in range(m):
        win_size = max(int(np.ceil(2 * scales[i] / dt)), 1)
        smooth_power[i, :] = uniform_filter1d(power[i, :], size=win_size, mode='nearest')
    # Scale smoothing
    for j in range(n):
        smooth_power[:, j] = uniform_filter1d(smooth_power[:, j], size=3, mode='nearest')
    return smooth_power

def wavelet_coherence_analysis(df, sanctions, eng=True):
    '''
    Obtain Wavelet Coherence spectrogram of two signals

    Parameters:
    df (Pandas.Dataframe): dataframe with two variables (voatilities)
    sanctions (list): list with dates in string format ('yyyy-mm-dd')
    eng (Boolean): The legend language , True if English, False if Russian
   
    Returns:
    WCA spectrogram with highlighted dates.
    '''

    # Extract log returns and time step (for daily data)
    dt = (df.index[1] - df.index[0]).days  
    #time = np.arange(len(df)) * dt  

    x1 = df.iloc[:,0].values
    x2 = df.iloc[:,1].values
    wca_name = f'{df.columns[0]} и {df.columns[1]}' # vs

    # Plot legend settings
    if eng:
        coherence_label = 'Coherence'
        ci_line = 'Cone of Influence'
        oy_line = 'Time Horizon (Days)'
        plot_title = f"Wavelet Coherence Analysis: {wca_name}"
        vertical_name = 'Sanctions'
    else:
        coherence_label = f'Когерентность'
        ci_line = 'Конус влияния'
        oy_line = 'Временной горизонт, в днях'
        plot_title = f'{wca_name}, {year}'
        vertical_name = 'Санкции EC' #'Оъявления ФРС' , 'Санкции США', 'ЕС'


    # Set wavelet parameters
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Smallest scale
    dj = 1 / 12  # Scale spacing

    # Calculate J to ensure maximum scale covers at least 60 days
    max_period = 128  # Target maximum period in days
    J = int(np.ceil(np.log2(max_period / s0) / dj))  # Number of scales

    # Compute wavelet coherence using pycwt.wct
    WCT, _, coi, freqs, _ = wavelet.wct(x1, x2, dt, dj=dj, s0=s0, J=J, wavelet=mother)

    ''' Another approach to estimate wavelets'''
    
    # Compute continuous wavelet transforms
    wave1, scales, freqs, coi, _, _ = wavelet.cwt(x1, dt, dj, s0, J, mother)
    wave2, _, _, _, _, _ = wavelet.cwt(x2, dt, dj, s0, J, mother)

    # Cross-wavelet transform
    W12 = wave1 * np.conj(wave2)
    S_W12 = wavelet_smooth(W12.real, scales, dt) + 1j * wavelet_smooth(W12.imag, scales, dt)
    
    phase = np.angle(S_W12)  # Phase difference

    # Convert frequency to period in days
    period_days = 1 / freqs

    # Plot coherence
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(left=0.1, right=1, top=0.93, bottom=0.08)
    T, P = np.meshgrid(df.index, period_days)
    
    c = ax.contourf(T, P, WCT, levels=np.linspace(0, 1, 11), cmap="jet") #cmap='jet' cmap="coolwarm" , extend='both'
    #plt.colorbar(c, label=coherence_label)
    cbar = plt.colorbar(c)
    cbar.set_label(coherence_label, fontsize=14)
    
    # Add uniform phase arrows across the grid
    step_time = 10  # Interval for arrows along time axis
    step_scale = 5  # Interval for arrows along scale axis
    time_indices = np.arange(0, len(df.index), step_time)
    scale_indices = np.arange(0, len(period_days), step_scale)

    # Meshgrid for arrow positions
    T_arrow, P_arrow = np.meshgrid(time_indices, scale_indices)
    U = np.cos(phase[P_arrow, T_arrow])/3  # X component of arrow
    V = np.sin(phase[P_arrow, T_arrow])/3  # Y component of arrow

    # Convert df.index to numpy array for indexing
    time_array = df.index.to_numpy()  # Fixes the indexing error

    # Plot arrows: bold, big, and uniform
    ax.quiver(time_array[T_arrow], period_days[P_arrow], U, V,
              scale=15, width=0.005, headwidth=3, headlength=5, color='black')

    # Plot Cone of Influence (coi in scale units, approx. equal to period for Morlet)
    ax.plot(df.index, coi, "--k", linewidth=1.5, label=ci_line)

    # Create a mask for areas outside the COI
    coi_mask = P > coi[np.newaxis, :]  # Broadcasting COI to match P's shape
    # Apply blurring effect outside the COI
    ax.contourf(T, P, coi_mask, levels=[0.5, 1], colors='darkblue', alpha=0.5)

    # Configure y-axis (time horizon in days)
    ax.set_yscale("log")
    ax.set_ylim([2, 128])  # Limit to 2-22 days as specified
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels([f"{int(t)}" for t in ax.get_yticks()])

    # Labels and title
    ax.set_xlabel("", fontsize=10) # f"Months, {year}"
    ax.set_ylabel(oy_line, fontsize=14)
    ax.set_title(plot_title, fontsize=14) # f"Wavelet Coherence Analysis: {wca_name}"

    # Customize x-axis: set month abbreviation and vertical labels
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # Show ticks every month
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))  # Format as "Jan", "Feb", etc.
    plt.xticks(rotation=90)  # Rotate labels to vertical

    # Rotate labels for better readability (optional)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    '''
    # Optional: adding grey area for absence of tradings in 2022:
    plt.axvspan("2022-02-24", "2022-03-28", color="gray", alpha=0.9)
    '''

    if str(year) in ['2020','2021','2022', '2023']:
        add_sanction_line(sanctions, df)
        # Add custom red dotted line (not actually plotted)
        custom_line, = plt.plot([], [], 'r--', label=vertical_name)  # 'r--' = red dotted line # 'k:' = k dotted line

    plt.legend()
    plt.legend(loc='upper right', fontsize=14) #upper right lower left
    plt.show()


# USING
#==========================================================================================
#Input required data: time-series and a year of estimation
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

# Realized volatility 
volatilities = returns**2
print(volatilities.head(3), "\n", volatilities.tail(3))

#Realized volatilities plot
#volatilities.plot()
#plt.show()

wavelet_coherence_analysis(volatilities, sanction_list, eng=False)
