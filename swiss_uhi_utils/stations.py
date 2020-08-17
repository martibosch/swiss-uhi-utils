import zipfile

import pandas as pd

WSL_T_COL = '91'


def df_from_meteoswiss_zip(zip_filepath, tair_column):
    with zipfile.ZipFile(zip_filepath) as zf:
        data_fn = next(fn for fn in zf.namelist() if fn.endswith('_data.txt'))
        # METEOSWISS_CSV_KWS = {'delim_whitespace': True, 'na_values': '-'}
        df = pd.read_csv(zf.open(data_fn), delim_whitespace=True,
                         na_values='-')

    # pivot and set datetime index
    df = df.drop(df[df['stn'] == 'stn'].index)
    df = df.pivot(index='time', columns='stn', values=tair_column)
    df.index = pd.to_datetime(df.index.astype(str))

    return df


def df_from_agrometeo(data_filepath):
    df = pd.read_csv(data_filepath, index_col=0, sep=';', na_values='?',
                     skiprows=[1, 2])

    # set datetime index
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y %H:%M')

    return df


def df_from_wsl(data_filepath, station_name=None):
    df = pd.read_csv(data_filepath, delim_whitespace=True)

    # set datetime index
    rename_dict = {
        'JAHR': 'year',
        'MO': 'month',
        'TG': 'day',
        'HH': 'hour',
        'MM': 'minute'
    }
    datetime_columns = list(rename_dict.keys())
    df.index = pd.to_datetime(df[datetime_columns].rename(columns=rename_dict))

    # drop all the columns except the temperature measurements
    df = df[[WSL_T_COL]]
    if station_name is not None:
        df = df.rename(columns={WSL_T_COL: station_name})

    return df
