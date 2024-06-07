import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import statsmodels.api as sm

from arch import arch_model
from arch.unitroot import VarianceRatio


# Load the data
file_path_imoex = 'stock_data/индекс-мосбиржи-ru000a0jp7k5.xlsx'
file_path_rtsi = 'stock_data/ртс-ru000a0jpeb3.xlsx'
df = pd.read_excel(file_path_imoex, index_col=0, parse_dates=True)
df_rts = pd.read_excel(file_path_rtsi, index_col=0, parse_dates=True)
df.columns = ['index']
df_rts.columns = ['index']
df = df.sort_values(by='Дата')
df_rts = df_rts.sort_values(by='Дата')
df['imoex_return'] = df['index'].pct_change()  
df_rts['rts_return'] = df_rts['index'].pct_change()
df_merg =  pd.merge(df[['imoex_return']], df_rts[['rts_return']], left_index=True, right_index=True, how='left')
df_merg = df_merg.dropna(subset=['imoex_return', 'rts_return'])

df_before = df_merg[(df_merg.index >= '2020-01-01') & (df_merg.index <= '2022-01-01')]
print(df_before.head(10))
df_after= df_merg[(df_merg.index >= '2022-01-01') & (df_merg.index <= '2023-12-31')]
print(df_after.head(10))



#============================Comparison of two indexes====================================

#---------------Variance Ratio Test -------------------
# before 2022
vr_test_imoex_before = VarianceRatio(df_before['imoex_return'].dropna(), lags=2)
vr_test_rtsi_before = VarianceRatio(df_before['rts_return'].dropna(), lags=2)
# after 2022
vr_test_imoex_after = VarianceRatio(df_after['imoex_return'].dropna(), lags=2)
vr_test_rtsi_after = VarianceRatio(df_after['rts_return'].dropna(), lags=2)
#------------------------------------------------------

# Calculate Cross-Correlation
cross_corr_before = df_before[['imoex_return', 'rts_return']].corr().iloc[0, 1]
cross_corr_after = df_after[['imoex_return', 'rts_return']].corr().iloc[0, 1]

# Compile results into a DataFrame
results = pd.DataFrame({
    '2020-2021 IMOEX': [vr_test_imoex_before.stat, vr_test_imoex_before.pvalue],
    '2020-2021 RTSI': [vr_test_rtsi_before.stat, vr_test_rtsi_before.pvalue],
    '2022-2023 IMOEX': [vr_test_imoex_after.stat, vr_test_imoex_after.pvalue],
    '2022-2023 RTSI': [vr_test_rtsi_after.stat, vr_test_rtsi_after.pvalue]
}, index=['Variance Ratio Test Stat', 'Variance Ratio Test P-value'])
# Add Cross-Correlation results
results.loc['Cross-Correlation'] = [cross_corr_before, cross_corr_before, cross_corr_after, cross_corr_after]
results = results.round(3)

# Display the results
print("Combined Analysis Results:")
print(results)
#========================================================================================


# Calculate the minimum value of the index
min_value = df['index']["2022"].min()
min_date = df['index']["2022"].idxmin()


# -------------IMOEX plot 2016-2023--------------------
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y='index')

# Adding the horizontal red bold line for the minimum value
plt.axhline(y=min_value, color='red', linewidth=2.5, linestyle='--')
# Adding the minimum value text annotation
plt.text(x=min_date, y=min_value, s=f'{min_value:.2f}', color='black', fontsize=12, fontweight='bold', ha='center', va='top')

# Customizing the plot
plt.xlabel('')#2016 - 2022
plt.ylabel('')
plt.title('Индекс Мосбиржи')
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
plt.show()
#-----------------------------------------------------

#Plotting indexes return
plt.figure(figsize=(12, 6))
sns.lineplot(data=df_after, x=df_after.index, y='rts_return', color='orange', label='RTSI Return')
sns.lineplot(data=df_after, x=df_after.index, y='imoex_return', color='blue', label='IMOEX Return')
plt.xlabel('')#2016 - 2022
plt.ylabel('')
plt.title('Доходность индексов IMOEX и RTSI')
plt.xticks(rotation=0)
plt.grid(True)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()
