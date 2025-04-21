import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from garch_plot import add_sanction_line


def performance_plot(folder, file_name_01, file_name_02, column_name01, column_name02, year, start_date, to_year, verical_line, eng=True):
    '''
    Indices Performance Plot setup

    Parameters:
    ...

    Returns:
    Plot of two indices performance with hilighted sanctions dates.
    '''
    #1st Market data
    index01 = pd.read_excel(folder+file_name_01+'.xlsx', index_col=0, parse_dates=True) #rtsi, imoex, eurusd_tom, usd_cad-(fx)_02, usd_eur-(fx)
    index01 = index01.sort_values(by='Дата')
    index01.columns = [column_name01]

    #2nd Market data
    index02 = pd.read_excel(folder+file_name_02+'.xlsx', index_col=0, parse_dates=True) #rtsi, imoex, eurusd_tom, usd_cad-(fx)_02, usd_eur-(fx)
    index02 = index02.sort_values(by='Дата')
    index02.columns = [column_name02]

    # Joining 2 indices together
    df = pd.merge(index01, index02, left_index=True, right_index=True, how='left')
    df = df.dropna()
    df = df[(df.index>=start_date) & (df.index<=f"{to_year}-01-01")]
    
    # Normilized df
    df = df / df.iloc[0]

    if eng:
        plot_title = f'Indices Performance: {df.columns[0]} vs {df.columns[1]}, {year}'
        vertical_name = 'FOMC'
    else:
        plot_title = f'{year}'
        vertical_name = 'Оъявления ФРС'


    plt.figure(figsize=(5, 3))
    plt.plot(df.index, df.iloc[:,0], 'k-', label=df.columns[0], linewidth=1.5, color='grey')
    plt.plot(df.index, df.iloc[:,1], 'k--', label=df.columns[1], linewidth=2.5)
    # old defining:
    #for col in df.columns:
    #    plt.plot(df.index, df[col], 'r:', label=col, linewidth=2.0)
    
    plt.title(plot_title, fontsize=14)
    plt.xlabel("", fontsize=9) #year
    plt.ylabel("", fontsize=10) #
    plt.grid(True)
    if str(year) in ['2020','2021','2022', '2023']:
        add_sanction_line(verical_line, df)
        # Add custom red dotted line (not actually plotted)
        custom_line, = plt.plot([], [], 'k:', label=vertical_name)  # 'r--' = red dotted line
    plt.legend(loc='upper left', fontsize=10, framealpha=0.5) #upper lower left right
    # Get the current axis
    ax = plt.gca()
    # Set x-axis ticks to three times per month at valid trading days
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    # Format labels as '03/10'
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')) #%d/%m
    # Rotate labels for readability
    plt.xticks(rotation=45)
    # Show the plot
    plt.tight_layout()
    plt.show()
