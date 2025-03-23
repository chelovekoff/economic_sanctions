import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pycwt as wavelet
#from pycwt.helpers import find
from pycwt.wavelet import Morlet
#from pycwt.helpers import fftconv
from scipy.ndimage import uniform_filter1d  # ✅ Fix for consistent output shape


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

def wavelet_coherence_analysis(df, sanctions):
    # Extract log returns and time step (assuming daily data)
    dt = (df.index[1] - df.index[0]).days  
    #time = np.arange(len(df)) * dt  

    x1 = df['r_m'].values
    x2 = df['r_f'].values

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
    T, P = np.meshgrid(df.index, period_days)
    
    c = ax.contourf(T, P, WCT, levels=np.linspace(0, 1, 11), cmap="jet") #cmap='jet' cmap="coolwarm" , extend='both'
    plt.colorbar(c, label="Coherence")


    '''# Add phase arrows
    step = 10  # Controls arrow density
    for i in range(0, len(period_days), step):
        for j in range(0, len(df.index), step):
            #if WCT[i, j] > 0.5:  # Threshold for significant coherence
            angle = phase[i, j] 
            arrow_length = 0.2
            dx = arrow_length * np.cos(angle)
            dy = arrow_length * np.sin(angle)
            y = period_days[i] #current period
            dy_visual = dy * (np.log10(y*1.1) - np.log10(y)) / arrow_length
            ax.arrow(df.index[j], y,
                    dx, dy_visual,
                    head_width=0.1, head_length=0.3, fc='k', ec='k')'''
    
    # Add uniform phase arrows across the grid
    step_time = 20  # Interval for arrows along time axis
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
    ax.plot(df.index, coi, "--k", linewidth=1.5, label="Cone of Influence")

    # Configure y-axis (time horizon in days)
    ax.set_yscale("log")
    ax.set_ylim([2, 128])  # Limit to 2-22 days as specified
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels([f"{int(t)}" for t in ax.get_yticks()])

    # Labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Time Horizon (Days)")
    ax.set_title("Wavelet Coherence Analysis")



    add_sanction_line(sanctions, df)
    # Add custom red dotted line (not actually plotted)
    custom_line, = plt.plot([], [], 'r--', label='sanctions')  # 'r--' = red dotted line


    plt.legend()
    plt.show()



# USING
    
sanction_list = [
    #'2022-05-30',
    '2022-06-03', #6th package - oil import, SWIFT
    '2022-09-02', #8th package - announcment
    '2022-10-06', #8th package - imposition
    '2022-12-05',
    '2023-02-04', #!!! 8th package - oil price ceiling imposition
    '2023-06-23', # 11th package - tanker fleet +PR
    '2023-12-18'
]


f = "stock_data/"

# Risk free rate:
filename_rf = 'rgbitr-ru000a0jqv87.xlsx' #rub-yield-curve-1y,  s-p-500, rgbitr-ru000a0jqv87
risk_free = pd.read_excel(f+filename_rf, index_col=0, parse_dates=True)
risk_free = risk_free.sort_values(by='Дата')
risk_free.columns = ['r_f']
#First Difference:
#r_f = 1*((risk_free.r_f) - (risk_free.r_f.shift(1)))# (risk_free['r_f'] - risk_free['r_f'].shift(1))
#Log-Return:
r_f = 100*(np.log(risk_free.r_f) - np.log(risk_free.r_f.shift(1)))# (risk_free['r_f'] - risk_free['r_f'].shift(1))
r_f.name = 'r_f'
print(r_f.head())
# Realized volatility 
r_f = r_f**2
print(r_f.head())
r_f = r_f.dropna()

#Market data
market = pd.read_excel(f+'oil/MOEXOG.xlsx', index_col=0, parse_dates=True) #rtsi, imoex, eur_usd-(fx)_02, usd_cad-(fx)_02, oil/MOEXOG
market = market.sort_values(by='Дата')
market.columns = ['m']
market['r_m'] = 100*(np.log(market.m) - np.log(market.m.shift(1))) #(market.m.pct_change()) # market log return #100 * (sp_price['Close'].pct_change())
# Realized volatility 
market['r_m'] = market['r_m']**2

#market['r_m'] = market['r_m'].shift(1)

market = market.dropna()

#SP500
sp_name = 's-p-500.xlsx' #usd_cad-(fx), s-p-500, eur_usd-(fx)_01, eur_usd-(fx)_02, usd_rub-(банк-россии)
us_market = pd.read_excel(f+sp_name, index_col=0, parse_dates=True)
us_market = us_market.sort_values(by='Дата')
us_market.columns = ['m']
us_market['r_sp500'] = 100*(np.log(us_market.m) - np.log(us_market.m.shift(1))) #(us_market.m.pct_change()) # market log return

# Currency data
cur_name = 'usd_rub-(банк-россии).xlsx' #usd_cad-(fx), s-p-500, eur_usd-(fx)_01, eur_usd-(fx)_02, usd_rub-(банк-россии), usd_eur-(fx)
cur_market = pd.read_excel(f+cur_name, index_col=0, parse_dates=True)
cur_market = cur_market.sort_values(by='Дата')
cur_market.columns = ['cur']
cur_r = 100*(np.log(cur_market.cur) - np.log(cur_market.cur.shift(1))) #(us_market.m.pct_change()) # market log return
cur_r.name = 'r_cur'
# Realized volatility 
cur_r = cur_r**2


#VIX
vix_name = 'vix-index.xlsx'
vix = pd.read_excel(f+vix_name, index_col=0, parse_dates=True)
vix = vix.sort_values(by='Дата')
vix.columns = ['vix_value']
vix['d_vix'] = vix.vix_value - vix.vix_value.shift(1)# different in VIX


df = pd.merge(market['r_m'], r_f, left_index=True, right_index=True, how='left')
df = pd.merge(df, cur_r, left_index=True, right_index=True, how='left')
df = df.dropna()

year = '2023'
to_year = str(int(year) + 1)
df = df[(df.index>=f"{year}-01-01") & (df.index<=f"{to_year}-01-01")]
print(len(df))
print(df.head(3), "\n", df.tail(3))


wavelet_coherence_analysis(df, sanction_list)
