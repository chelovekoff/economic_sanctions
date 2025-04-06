# economic_sanctions
The purpose of this project is to evaluate anti-Russian economic sanctions (2022-2023) on the stock market performance.

`main.py` - Event-Driven Analysys of the US/EU sanctions against Russia
1. Input the integral value of the event winow ('tau'): 3, 5, or 7
2. Input the sanctions' sender:
    * 'US' to estimate American sanctions against Russian Financial Sector;
    * 'EU' to estimate European Union sanctions against Russian Oil&Gas Industry.
3. Input the currency for returns in CAPM: USD/EUR
4. Input the method of sanction assessment: type of the Event-Driven Analysys:
    * 'U' - Univariate;
    * 'M' - Multivariate.
#===========================================================

`ccc_garch.py` - Univariate GARCH(1,1) model with constant conditional correlation estimated on residuals of VAR(1) process
1. Input two of the following indicies: MOEX/RTSI/USDRUB/USDEUR/SP500/RGBITR/RYC
2. Input the starting year of calculations (yyyy)

#===========================================================

`var_garch_new.py` - Bivariate VAR(1)-GARCH(1,1) model with constant conditional correlation (Ling & McAleer, 2003)
1. Input two of the following indicies: MOEX/RTSI/USDRUB/USDEUR/SP500/RGBITR/RYC
2. Input the starting year of calculations (yyyy)
#===========================================================

`wac.py` - Building a spectrogram of the Wavelet Coherence Analysis (Krieger & Freij, 2023)
