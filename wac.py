import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pycwt as wavelet
#from pycwt.helpers import find
from pycwt.wavelet import Morlet
#from pycwt.helpers import fftconv
from scipy.ndimage import uniform_filter1d  # ✅ Fix for consistent output shape
from currency_premia import excange_return


def obtain_rf(filename):
    # Risk-free rate: RUB Yield Curve 1Y
    risk_free = pd.read_excel(f+filename, index_col=0, parse_dates=True)
    risk_free = risk_free.sort_values(by='Дата')

    # Calculation of the day risk-free rate:
    risk_free.columns = ['r_f']
    risk_free['r_f'] = risk_free['r_f']/100
    risk_free = ((1 + risk_free) ** (1/365) - 1)
    print("=======\nRF:\n", risk_free.head(30))
    return risk_free
    

def wavelet_smooth(power, scales, dt):
    """
    Smoothing function for wavelet power spectra.
    Uses a moving average in both time and scale dimensions.
    """
    m, n = power.shape
    smooth_power = np.zeros_like(power)

    # Time smoothing (moving average with adaptive window size)
    for i in range(m):
        win_size = int(np.ceil(2 * scales[i] / dt))  # Adaptive window
        win_size = max(win_size, 1)  # Ensure at least 1
        smooth_power[i, :] = uniform_filter1d(power[i, :], size=win_size, mode='nearest')

    # Scale smoothing (fixed window size)
    for j in range(n):
        smooth_power[:, j] = uniform_filter1d(smooth_power[:, j], size=3, mode='nearest')

    return smooth_power

def wavelet_coherence_analysis(df):
    # Extract log returns and time step (assuming daily data)
    dt = (df.index[1] - df.index[0]).days  
    time = np.arange(len(df)) * dt  

    x1 = df['r_m'].values
    x2 = df['r_sp500'].values

    # Set wavelet parameters
    mother = wavelet.Morlet(6)
    s0 = 2 * dt  # Smallest scale
    dj = 1 / 12  # Scale spacing
    J = int(1 / dj * np.log2(len(x1) * dt / s0))  # Number of scales

    # Compute CWT for both series
    wave1, scales, freqs, coi, _, _ = wavelet.cwt(x1, dt, dj, s0, J, mother)
    wave2, _, _, _, _, _ = wavelet.cwt(x2, dt, dj, s0, J, mother)

    # Compute cross-wavelet transform
    W12 = wave1 * np.conj(wave2)

    # Compute wavelet coherence (smoothed)
    S1 = wavelet_smooth(np.abs(wave1) ** 2, scales, dt)
    S2 = wavelet_smooth(np.abs(wave2) ** 2, scales, dt)
    S12 = wavelet_smooth(np.abs(W12) ** 2, scales, dt)
    WCT = S12 / (S1 * S2)  # Wavelet Coherence
    #WCT = np.clip(WCT, 0, 1)  # ✅ Fix: Ensure no NaN or extreme values


    # Convert frequency to time scale (in days)
    period_days = 1 / freqs

    # Plot coherence
    fig, ax = plt.subplots(figsize=(12, 6))

    T, P = np.meshgrid(df.index, period_days)
    
    c = ax.contourf(T, P, WCT, levels=np.linspace(0, 1, 11), cmap="jet") #cmap='jet' cmap="coolwarm" , extend='both'
    plt.colorbar(c, label="Coherence")

    # Cone of Influence
    ax.plot(df.index, coi * dt, "--k", linewidth=1.5, label="Cone of Influence")

    # ✅ Fix: Set correct y-axis labels (Time Horizon in Days)
    ax.set_yscale("log")
    ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256])  # Manually set days
    ax.set_yticklabels([f"{int(t)}" for t in ax.get_yticks()])  # Show as whole numbers

    ax.set_xlabel("Date")
    ax.set_ylabel("Time Horizon (Days)")
    ax.set_title("Wavelet Coherence Analysis")

    plt.legend()
    plt.show()



# USING

f = "stock_data/"
filename_rf = 'rub-yield-curve-1y.xlsx'
risk_free = obtain_rf(filename_rf)
print('Risk-free return:\n------\n', risk_free.head())
risk_free['change_in_rf'] = 100 * (risk_free['r_f'].pct_change())
risk_free = risk_free.dropna()

#Market data
market = pd.read_excel(f+'imoex.xlsx', index_col=0, parse_dates=True)
market = market.sort_values(by='Дата')
market.columns = ['m']
market['r_m'] = (np.log(market.m) - np.log(market.m.shift(1))) # market log return
market['r_m'] = market['r_m']**2
print("!!!!")
print(market.head())

#Excange rate
cbr = "usd_rub-(банк-россии).xlsx"
rubusd_return = excange_return(cbr)


#SP500
sp_name = 's-p-500.xlsx'
us_market = pd.read_excel(f+sp_name, index_col=0, parse_dates=True)
us_market = us_market.sort_values(by='Дата')
us_market.columns = ['m']
us_market['r_sp500'] = (np.log(us_market.m) - np.log(us_market.m.shift(1))) # market log return
us_market['r_sp500'] = us_market['r_sp500']**2

#VIX
vix_name = 'vix-index.xlsx'
vix = pd.read_excel(f+vix_name, index_col=0, parse_dates=True)
vix = vix.sort_values(by='Дата')
vix.columns = ['vix_value']
vix['r_vix'] = (np.log(vix.vix_value) - np.log(vix.vix_value.shift(1))) # market log return

'''
# Conditional volatility
rf_volatility_name = 'change_in_rf_std.xlsx'
rf_volatility = pd.read_excel(rf_volatility_name, index_col=0, parse_dates=True)
rf_volatility = rf_volatility.sort_values(by='Дата')
rf_volatility.columns = ['rf_volatility']
rf_volatility['rf_volatility'] = rf_volatility['rf_volatility']/100

moex_volatility_name = 'change_in_rf_std.xlsx'
moex_volatility = pd.read_excel(moex_volatility_name, index_col=0, parse_dates=True)
moex_volatility = moex_volatility.sort_values(by='Дата')
moex_volatility.columns = ['moex_volatility']
moex_volatility['moex_volatility'] = moex_volatility['moex_volatility']/100

moex_volatility = pd.merge(moex_volatility, rf_volatility, left_index=True, right_index=True, how='left')
moex_volatility = moex_volatility.dropna()
#print(moex_volatility.head())
#print(moex_volatility.tail())
#wavelet_coherence_analysis(moex_volatility)

'''



# Example usage
market = pd.merge(market, risk_free, left_index=True, right_index=True, how='left')
#market = pd.merge(market, rubusd_return, left_index=True, right_index=True, how='left')
market = pd.merge(market, us_market, left_index=True, right_index=True, how='left')
#market = pd.merge(us_market, vix, left_index=True, right_index=True, how='left')

market = market.dropna()
#market['rm_rf'] = (market['r_m']- market['r_f'])*100
print(market.isna().sum())
print(market.head())

print(len(market))
wavelet_coherence_analysis(market)
