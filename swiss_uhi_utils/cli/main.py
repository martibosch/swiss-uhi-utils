import logging

import click
import numpy as np
import rasterio as rio
from rasterio import transform, windows

import swiss_uhi_utils as suhi


# CLI
@click.group()
@click.version_option(version=suhi.__version__, message="%(version)s")
@click.pass_context
def cli(ctx):
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    ctx.ensure_object(dict)
    ctx.obj['LOGGER'] = logger


@cli.command()
@click.pass_context
@click.argument('lulc_filepath', type=click.Path(exists=True))
@click.argument('binary_feature_filepath', type=click.Path(exists=True))
@click.argument('dst_filepath', type=click.Path())
@click.option('--dst-dtype', default=None, required=False)
def compute_feature_cover(ctx, lulc_filepath, binary_feature_filepath,
                          dst_filepath, dst_dtype):
    logger = ctx.obj['LOGGER']

    # read the agglomeration extract raster
    with rio.open(lulc_filepath) as src:
        lulc_arr = src.read(1)
        res = src.res
        shape = src.shape
        t = src.transform
        nodata = src.nodata
        meta = src.meta

    # get a list of (x,y) pixel coordinates
    lulc_flat_arr = lulc_arr.flatten()
    cond = lulc_flat_arr != nodata
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                         indexing='ij')
    xys = np.array([
        transform.xy(t, i, j) for i, j in zip(ii.flatten()[cond],
                                              jj.flatten()[cond])
    ])
    logger.info("computed flat list of %d (x, y) pixel coordinates of %s",
                len(xys), lulc_filepath)

    # get the percentage of tree cover of each pixel
    # get_pixel_tree_cover(xy, src, x_inc, y_inc)
    x_inc, y_inc = res[0] / 2, res[1] / 2
    with rio.open(binary_feature_filepath) as src:

        def get_pixel_feature_cover(xy):
            arr = src.read(
                1, window=windows.from_bounds(xy[0] - x_inc, xy[1] - y_inc,
                                              xy[0] + x_inc, xy[1] + y_inc,
                                              transform=src.transform))
            # ACHTUNG: gdalmerge might have messed with `src.nodata`
            # TODO: avoid UGLY HARDCODED zero below (inspect gdalmerge or
            # accept extra click CLI arg for tree nodata value)
            # return np.sum(arr != src.nodata) / arr.size
            return np.sum(arr != 0) / arr.size

        feature_cover_flat_arr = np.apply_along_axis(get_pixel_feature_cover,
                                                     1, xys)
    logger.info("extracted per-pixel proportion of feature cover from %s",
                binary_feature_filepath)

    if dst_dtype is None:
        dst_dtype = feature_cover_flat_arr.dtype

    # use `nodata` (instead of `zeros_like`) because it allows distinguishing
    # actual zeros from nodata
    # tree_cover_arr = np.zeros_like(lulc_arr, dtype=tree_cover_flat_arr.dtype)
    tree_cover_arr = np.full_like(lulc_arr, nodata, dtype=dst_dtype)
    # use `ravel` instead of `flatten` because the former returns a view that
    # can be modified in-place (while the latter returns a copy)
    tree_cover_arr.ravel()[cond] = feature_cover_flat_arr

    # dump the tree cover raster
    meta.update(dtype=dst_dtype)
    with rio.open(dst_filepath, 'w', **meta) as dst:
        dst.write(tree_cover_arr, 1)
    logger.info("dumped raster of per-pixel proportion of feature cover to %s",
                dst_filepath)
