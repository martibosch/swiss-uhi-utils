from os import environ, path

import fsspec
import pandas as pd
import pyeto
import xarray as xr

from . import geo_utils, settings

__all__ = ['open_meteoswiss_s3_ds', 'get_ref_et_da']

# meteoswiss utils
# constants useful for geo-operations
METEOSWISS_CRS = 'epsg:21781'
DROP_DIMS = ['lon', 'lat', 'dummy', 'swiss_coordinates']
RENAME_DIMS_MAP = {'chx': 'x', 'chy': 'y'}

# meteoswiss grid products needed to compute the reference evapotranspiration
METEOSWISS_GRID_PRODUCTS = ['TminD', 'TabsD', 'TmaxD']


def _preprocess_meteoswiss_ds(ds):
    # set crs attribute to dataset and all data variables individually
    ds.attrs['pyproj_srs'] = METEOSWISS_CRS
    for data_var in list(ds.data_vars):
        ds[data_var].attrs['pyproj_srs'] = METEOSWISS_CRS

    # drop unnecessary dimensions and rename the others so that salem can
    # understand the grid
    return ds.drop(DROP_DIMS).rename(RENAME_DIMS_MAP)


def open_meteoswiss_s3_ds(year, product, open_kws=None, shape=None, geom=None,
                          crs=None, preprocess=False, roi=True, prefix=None,
                          subset_kws=None, roi_kws=None):
    # prepare remote access to MeteoSwiss grid data
    fs = fsspec.filesystem(
        # 'filecache',
        'simplecache',
        target_protocol='s3',
        target_options=dict(
            profile=environ.get('S3_PROFILE_NAME',
                                settings.METEOSWISS_S3_PROFILE_NAME),
            client_kwargs=settings.METEOSWISS_S3_CLIENT_KWARGS),
        cache_storage=settings.METEOSWISS_CACHE_STORAGE_DIR)
    bucket_name = environ.get('S3_BUCKET_NAME',
                              settings.METEOSWISS_S3_BUCKET_NAME)

    if prefix is None:
        prefix = settings.METEOSWISS_GRID_DATA_PREFIX

    file_key = path.join(
        bucket_name, prefix, product,
        f'{product}_ch01r.swisscors_{year}01010000_{year}12310000.nc')
    with fs.open(file_key) as file_obj:
        if open_kws is None:
            open_kws = {}
        ds = xr.open_dataset(file_obj, **open_kws)

    if shape is not None or geom is not None:
        ds = geo_utils.clip_ds_to_extent(_preprocess_meteoswiss_ds(ds),
                                         shape=shape, geom=geom, crs=crs,
                                         roi=roi, subset_kws=subset_kws,
                                         roi_kws=roi_kws)
    elif preprocess:
        ds = _preprocess_meteoswiss_ds(ds)

    return ds


# other utils to compute the reference evapotranspiration
def _compute_solar_radiation(date, lat):
    day_of_year = date.timetuple().tm_yday
    sol_dec = pyeto.sol_dec(day_of_year)
    sha = pyeto.sunset_hour_angle(lat, sol_dec)
    ird = pyeto.inv_rel_dist_earth_sun(day_of_year)
    return pyeto.et_rad(lat, sol_dec, sha, ird)


def _compute_ref_eto(day_ds, lat):
    return pyeto.hargreaves(
        day_ds['TminD'], day_ds['TabsD'], day_ds['TmaxD'],
        _compute_solar_radiation(pd.to_datetime(day_ds.time.values), lat))


def get_ref_et_da(dates_ser, geom, lat, crs):
    # pre-compute the meteo inputs
    t_ds = xr.concat([
        xr.Dataset({
            meteoswiss_grid_product: open_meteoswiss_s3_ds(
                year_period.year,
                meteoswiss_grid_product,
                geom=geom,
                crs=crs,
                roi_kws={'all_touched': True},
            )[meteoswiss_grid_product]
            for meteoswiss_grid_product in METEOSWISS_GRID_PRODUCTS
        }).sel(time=year_ser.values)
        for year_period, year_ser in dates_ser.groupby(
            dates_ser.dt.to_period('Y'))
    ], dim='time')

    # reference evapotranspiration
    ref_eto_da = t_ds.groupby('time').map(_compute_ref_eto, args=(lat, ))

    # align the reference evapotranspiration data-array to the agglom. LULC
    ref_eto_da.name = 'ref_eto'
    ref_eto_da.attrs = dict(
        pyproj_srs=t_ds[list(t_ds.data_vars)[0]].attrs['pyproj_srs'],
        units='mm/day', long_name='$ET_o$')

    return ref_eto_da
