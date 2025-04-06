from dotenv import load_dotenv
import os

load_dotenv()

class config:
    """common config constants"""


    sanction_list: list = [
        #'2022-05-30', #overlapping
        '2022-06-03', # 6th package - oil import, SWIFT
        '2022-09-02', # 8th package - announcment
        '2022-10-06', # 8th package - imposition
        '2022-12-05',
        '2023-02-04', # 8th package - oil price ceiling imposition
        '2023-06-23', # 11th package - tanker fleet +PR
        '2023-12-18'
    ]

    data_files: dict = {
        'MOEX': 'imoex',
        'RTSI': 'ртс-ru000a0jpeb3',
        'USDRUB': 'usd_rub-(банк-россии)',
        'USDEUR': 'usd_eur-(fx)',
        #'EURUSD': 'eurusd_tom',
        #'USDCAD': 'usd_cad-(fx)',
        #'VIX': 'vix-index',
        'SP500': 's-p-500',
        'RGBITR': 'rgbitr-ru000a0jqv87',
        'RYC': 'rub-yield-curve-1y'
    }