from os import environ, path

# GEO-UTILS
CRS = 'epsg:2056'

# METEOSWISS
# constants related to s3 remote file system
METEOSWISS_GRID_DATA_PREFIX = 'meteoswiss'
METEOSWISS_CACHE_STORAGE_DIR = path.join(path.expanduser('~'),
                                         '.meteoswiss-cache')

METEOSWISS_S3_PROFILE_NAME = None
METEOSWISS_S3_CLIENT_KWARGS = None
METEOSWISS_S3_BUCKET_NAME = None
