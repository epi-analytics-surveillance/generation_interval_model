import numpy as np
import pandas


# data.csv = https://data.humdata.org/dataset/oxford-covid-19-government-response-tracker 
# OxCGRT_CSVCSV (46.4M) Modified: 2 September 2025 
df = pandas.read_csv('data.csv')


def get_si(country, first_date, last_date, df=df):

    sis = []

    for i, row in df.iterrows():
        if row['CountryName'] == country:
            if first_date <= int(row['Date']) <= last_date:
                sis.append(row['StringencyIndex_Average'])
    
    return np.mean(sis)


get_si('Australia', 20200121, 20200208)
