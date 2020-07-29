import numpy as np
import salem  # noqa: F401
import xarray as xr
from rasterio import transform

from . import settings

__all__ = ['get_ref_da', 'align_ds', 'clip_ds_to_extent']

# GEO-OPERATIONS


def _calculate_transform(geom, res):
    west, south, east, north = geom.bounds
    dst_height, dst_width = tuple(
        int(np.ceil(diff / res)) for diff in [north - south, east - west])
    dst_transform = transform.from_origin(west, north, res, res)

    return dst_transform, (dst_height, dst_width)


def get_ref_da(geom, res, fill=0, crs=None):
    if crs is None:
        crs = settings.CRS
    ref_transform, (ref_height, ref_width) = _calculate_transform(geom, res)
    rows = np.arange(ref_height)
    cols = np.arange(ref_width)
    xs, _ = transform.xy(ref_transform, cols, cols)
    _, ys = transform.xy(ref_transform, rows, rows)
    ref_da = xr.DataArray(fill, dims=('y', 'x'), coords={'y': ys, 'x': xs})
    ref_da.attrs['pyproj_srs'] = crs

    return ref_da


def align_ds(ds, ref_ds, interp='linear'):
    if ds.name is None:
        ds.name = ''  # salem needs some name to align the ds/da
    return ref_ds.salem.transform(ds, interp=interp)


def clip_ds_to_extent(ds, shape=None, geom=None, crs=None, roi=True,
                      subset_kws=None, roi_kws=None):
    if crs is None:
        crs = settings.CRS
    if subset_kws is None:
        subset_kws = {}
    if roi_kws is None:
        roi_kws = {}

    if shape is not None:
        subset_kws['shape'] = shape
        if roi:
            roi_kws['shape'] = shape
    elif geom is not None:
        subset_kws['geometry'] = geom
        subset_kws['crs'] = crs
        if roi:
            roi_kws['geometry'] = geom
            roi_kws['crs'] = crs
    subset_ds = ds.salem.subset(**subset_kws)
    if roi:
        return subset_ds.salem.roi(**roi_kws)
    return subset_ds
